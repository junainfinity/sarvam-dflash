#!/usr/bin/env python3
"""
Quick end-to-end test of DFlash speculative decoding vs baseline Sarvam-30B.
Loads both models, generates 64 tokens on 2 prompts, reports tok/s and acceptance rate.

Usage:
    python3 quick_test.py
"""

import sys, time, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TARGET_PATH = "./sarvam-30b"
DRAFT_CKPT  = "./checkpoints/dflash_draft_best.pt"
MAX_TOKENS  = 64
BLOCK_SIZE  = 16

PROMPTS = [
    "The key insight behind speculative decoding is that",
    "In a mixture-of-experts model, each token is routed to",
]

# ─────────────────────────────────────────────────────────────────────────────
# Load target
# ─────────────────────────────────────────────────────────────────────────────
print("Loading Sarvam-30B...", flush=True)
from transformers import AutoTokenizer
from modeling_sarvam_moe_dflash import SarvamMoEForCausalLMWithKVInjection, SarvamMoEConfig
from modeling_sarvam_moe_dflash import KV_INJECTION_HIDDEN_STATE_INDICES

tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

cfg = SarvamMoEConfig.from_pretrained(TARGET_PATH)
target = SarvamMoEForCausalLMWithKVInjection.from_pretrained(
    TARGET_PATH, config=cfg, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True,
)
target.eval()
device = next(target.parameters()).device
print(f"Target on {device}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load draft
# ─────────────────────────────────────────────────────────────────────────────
print("Loading draft model...", flush=True)
from train_dflash_sarvam import load_draft_from_checkpoint
from dflash_draft import make_block_causal_mask

draft = load_draft_from_checkpoint(DRAFT_CKPT, target)
draft.kv_fusion  = draft.kv_fusion.to(device, dtype=torch.bfloat16)
draft.layers     = draft.layers.to(device, dtype=torch.bfloat16)
draft.norm       = draft.norm.to(device, dtype=torch.bfloat16)
draft.rotary_emb = draft.rotary_emb.to(device)
draft.eval()
print(f"Draft model loaded.", flush=True)

block_mask = make_block_causal_mask(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bfloat16, device=device)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def greedy(logits):
    return logits.argmax(dim=-1)

@torch.no_grad()
def baseline_generate(prompt):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    t0 = time.perf_counter()
    for _ in range(MAX_TOKENS):
        out = target(ids)
        if device.type == "mps": torch.mps.synchronize()
        next_tok = greedy(out.logits[:, -1, :]).unsqueeze(1)
        ids = torch.cat([ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    t1 = time.perf_counter()
    gen = ids.shape[1] - tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    return tokenizer.decode(ids[0, tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]:],
                            skip_special_tokens=True), gen, t1 - t0

@torch.no_grad()
def dflash_generate(prompt):
    """
    Block-level speculative decoding.

    Draft was trained as a causal LM with block-diagonal mask: position i predicts
    token at position i+1, attending only to positions block_start..i. So at inference
    we must feed the block's prefix — we cannot predict all 16 tokens from a zero-padded
    input. We generate the block iteratively (15 small draft forwards), then the target
    verifies the whole 16-token block in ONE large forward. The parallelism win comes
    from the target side, not the draft.
    """
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = ids.shape[1]
    proposed, accepted = 0, 0
    t0 = time.perf_counter()
    tokens_gen = 0

    while tokens_gen < MAX_TOKENS:
        # === 1. Target forward on context → injection features + anchor token ===
        out = target.model(input_ids=ids, output_hidden_states=True,
                           use_cache=False, return_dict=True)
        selected = [out.hidden_states[i] for i in KV_INJECTION_HIDDEN_STATE_INDICES]
        inj = torch.cat(selected, dim=-1)  # [1, ctx, 20480] — full context
        anchor = greedy(target.lm_head(out.last_hidden_state[:, -1:, :])).squeeze(1)  # [1]

        if device.type == "mps": torch.mps.synchronize()
        if anchor.item() == tokenizer.eos_token_id:
            ids = torch.cat([ids, anchor.unsqueeze(1)], dim=1)
            tokens_gen += 1
            break

        # === 2. Draft generates next BLOCK_SIZE-1 tokens iteratively ===
        # Start with [anchor, 0, ..., 0]; after each forward, fill the next slot
        # with draft's prediction. Only the last-real-position's logit is read.
        draft_in = torch.zeros(1, BLOCK_SIZE, dtype=torch.long, device=device)
        draft_in[0, 0] = anchor.item()
        for i in range(BLOCK_SIZE - 1):
            draft_logits = draft(input_ids=draft_in, injection_features=inj, block_mask=block_mask)
            # logit at position i predicts token at position i+1
            next_tok = draft_logits[0, i, :].argmax().item()
            draft_in[0, i + 1] = next_tok
        candidates = draft_in  # [1, BLOCK_SIZE] = [anchor, d1, d2, ..., d15]
        proposed += BLOCK_SIZE - 1

        # === 3. Target verifies all 16 candidates in ONE forward ===
        # Append candidates to ids, run target, get logits at positions ctx-1..ctx+BLOCK_SIZE-2
        # which predict tokens at positions ctx..ctx+BLOCK_SIZE-1 (the full block).
        ctx = ids.shape[1]
        verify_out = target(input_ids=torch.cat([ids, candidates], dim=1),
                            use_cache=False, return_dict=True)
        verify_logits = verify_out.logits[:, ctx - 1: ctx + BLOCK_SIZE - 1, :]  # [1, BLOCK_SIZE, V]
        target_preds = greedy(verify_logits)  # [1, BLOCK_SIZE] — target's pick at each block position

        # === 4. Accept longest matching prefix ===
        # target_preds[0] is target's pick for block pos 0 (should equal anchor — trivial)
        # target_preds[i] is target's pick for block pos i (compare vs candidates[0, i])
        n_acc = 0
        for i in range(BLOCK_SIZE):
            if candidates[0, i].item() == target_preds[0, i].item():
                n_acc += 1
            else:
                break

        if n_acc == 0:
            # Degenerate (shouldn't happen — anchor always matches). Take target's correction.
            acc_toks = target_preds[:, :1]
            n_acc = 1
        else:
            acc_toks = candidates[:, :n_acc]

        # Stats: draft proposed BLOCK_SIZE-1 real tokens (pos 1..15), of which (n_acc-1) accepted
        accepted += max(n_acc - 1, 0)

        ids = torch.cat([ids, acc_toks], dim=1)
        tokens_gen += acc_toks.shape[1]

        if device.type == "mps": torch.mps.synchronize()
        if tokenizer.eos_token_id in acc_toks[0].tolist():
            break

    t1 = time.perf_counter()
    gen = ids.shape[1] - prompt_len
    return (tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True),
            gen, t1 - t0, accepted / max(proposed, 1))

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("QUICK TEST — DFlash vs Baseline")
print("=" * 70)

for i, prompt in enumerate(PROMPTS):
    print(f"\n── Prompt {i+1}: \"{prompt}\"")

    print("  [Baseline]", flush=True)
    b_text, b_toks, b_time = baseline_generate(prompt)
    b_tps = b_toks / b_time
    print(f"  {b_toks} tokens in {b_time:.1f}s ({b_tps:.2f} tok/s)")
    print(f"  Output: {b_text[:120]!r}")

    print("  [DFlash]", flush=True)
    d_text, d_toks, d_time, d_accept = dflash_generate(prompt)
    d_tps = d_toks / d_time
    speedup = d_tps / b_tps if b_tps > 0 else 0
    print(f"  {d_toks} tokens in {d_time:.1f}s ({d_tps:.2f} tok/s)")
    print(f"  Accept rate: {d_accept:.1%}  |  Speedup: {speedup:.2f}x")
    print(f"  Output: {d_text[:120]!r}")

print("\n" + "=" * 70)
print("Done.")
