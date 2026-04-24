#!/usr/bin/env python3
"""
V3 DFlash inference acceptance test — the moment of truth.

Runs baseline Sarvam-30B generation vs. DFlash speculative decoding on a
small prompt set. Writes progress to inference_test_state.json every few
seconds so the web dashboard (/inference) can render live updates.

DFlash V3 inference (matches training):
  - Draft input: [anchor_embed, mask_emb, mask_emb, ..., mask_emb] length L
  - Bidirectional within-block self-attention
  - Cross-attn to target features of context up to anchor
  - Single draft forward → argmax at each position → 16 candidate tokens
  - Target verifies the full block in ONE forward pass
  - Accept longest matching prefix
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from dflash_draft import make_block_bidirectional_mask

TARGET_PATH = "./sarvam-30b"
DRAFT_CKPT  = "./checkpoints/dflash_draft_best.pt"
STATE_FILE  = Path("inference_test_state.json")
MAX_TOKENS  = 64
BLOCK_SIZE  = 16

PROMPTS = [
    "The key insight behind speculative decoding is that",
    "In a mixture-of-experts model, each token is routed to",
    "Block diffusion language models are trained by",
]


class State:
    """Manages the JSON state file the dashboard reads."""

    def __init__(self):
        self.data = {
            "status":       "starting",
            "started_at":   time.time(),
            "completed_at": None,
            "stage":        "init",
            "stage_label":  "Initializing",
            "error":        None,
            "total_prompts": len(PROMPTS),
            "current_prompt_idx": 0,
            "prompts":      [{"text": p, "baseline": None, "dflash": None} for p in PROMPTS],
            "summary":      None,
            "updated_at":   time.time(),
        }
        self.flush()

    def set_stage(self, stage, label):
        self.data["stage"] = stage
        self.data["stage_label"] = label
        self.flush()

    def update_prompt(self, idx, mode, **kwargs):
        entry = self.data["prompts"][idx].get(mode) or {}
        entry.update(kwargs)
        self.data["prompts"][idx][mode] = entry
        self.data["current_prompt_idx"] = idx
        self.flush()

    def finalize(self, ok=True, error=None):
        # Compute summary
        baselines = [p["baseline"] for p in self.data["prompts"] if p.get("baseline")]
        dflashes  = [p["dflash"]   for p in self.data["prompts"] if p.get("dflash")]
        summary = {
            "baseline_avg_tok_per_sec": sum(b.get("tok_per_sec", 0) for b in baselines) / max(len(baselines), 1),
            "dflash_avg_tok_per_sec":   sum(d.get("tok_per_sec", 0) for d in dflashes)   / max(len(dflashes), 1),
            "dflash_avg_acceptance":    sum(d.get("acceptance_rate", 0) for d in dflashes) / max(len(dflashes), 1),
            "dflash_avg_speedup":       sum(d.get("speedup", 0) for d in dflashes) / max(len(dflashes), 1),
        }
        self.data["summary"] = summary
        self.data["status"] = "completed" if ok else "failed"
        self.data["error"] = error
        self.data["completed_at"] = time.time()
        self.flush()

    def flush(self):
        self.data["updated_at"] = time.time()
        tmp = STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(STATE_FILE)


def main():
    state = State()
    try:
        # --- Load target ---
        state.set_stage("loading_target", "Loading Sarvam-30B (~2 min)")
        from transformers import AutoTokenizer
        from modeling_sarvam_moe_dflash import (
            SarvamMoEForCausalLMWithKVInjection, SarvamMoEConfig,
            KV_INJECTION_HIDDEN_STATE_INDICES,
        )

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
        state.set_stage("loading_draft", "Loading V3 draft model")

        # --- Load draft ---
        from train_dflash_sarvam import load_draft_from_checkpoint

        draft = load_draft_from_checkpoint(DRAFT_CKPT, target)
        draft.kv_fusion    = draft.kv_fusion.to(device, dtype=torch.bfloat16)
        draft.layers       = draft.layers.to(device, dtype=torch.bfloat16)
        draft.norm         = draft.norm.to(device, dtype=torch.bfloat16)
        draft.rotary_emb   = draft.rotary_emb.to(device)
        draft.mask_embedding.data = draft.mask_embedding.data.to(device, dtype=torch.bfloat16)
        draft.eval()

        block_mask = make_block_bidirectional_mask(BLOCK_SIZE, BLOCK_SIZE,
                                                    dtype=torch.bfloat16, device=device)

        # --- Helpers ---
        @torch.no_grad()
        def baseline_generate(prompt, idx):
            state.set_stage("running_baseline", f"Baseline generation ({idx+1}/{len(PROMPTS)})")
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            prompt_len = ids.shape[1]
            state.update_prompt(idx, "baseline",
                status="running", tokens_generated=0, max_tokens=MAX_TOKENS,
                elapsed_s=0.0, tok_per_sec=0.0, output="",
            )
            t0 = time.perf_counter()
            for i in range(MAX_TOKENS):
                out = target(ids)
                if device.type == "mps": torch.mps.synchronize()
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)
                if i % 5 == 0 or i == MAX_TOKENS - 1:
                    elapsed = time.perf_counter() - t0
                    gen = ids.shape[1] - prompt_len
                    state.update_prompt(idx, "baseline",
                        tokens_generated=gen, elapsed_s=elapsed,
                        tok_per_sec=gen / max(elapsed, 1e-6),
                    )
                if next_tok.item() == tokenizer.eos_token_id:
                    break
            elapsed = time.perf_counter() - t0
            gen = ids.shape[1] - prompt_len
            output = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
            state.update_prompt(idx, "baseline",
                status="done", tokens_generated=gen, elapsed_s=elapsed,
                tok_per_sec=gen / max(elapsed, 1e-6), output=output[:200],
            )
            return gen / elapsed

        @torch.no_grad()
        def dflash_generate(prompt, idx, baseline_tps):
            state.set_stage("running_dflash", f"DFlash generation ({idx+1}/{len(PROMPTS)})")
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            prompt_len = ids.shape[1]
            proposed = 0; accepted = 0; tokens_gen = 0; blocks = 0
            state.update_prompt(idx, "dflash",
                status="running", tokens_generated=0, max_tokens=MAX_TOKENS,
                blocks_generated=0, acceptance_rate=0.0, elapsed_s=0.0,
                tok_per_sec=0.0, speedup=0.0, output="",
            )
            t0 = time.perf_counter()

            word_embeddings = target.model.word_embeddings

            while tokens_gen < MAX_TOKENS:
                # Step 1: target forward → injection features + anchor token
                tgt_out = target.model(
                    input_ids=ids, output_hidden_states=True,
                    use_cache=False, return_dict=True,
                )
                selected = [tgt_out.hidden_states[i] for i in KV_INJECTION_HIDDEN_STATE_INDICES]
                inj = torch.cat(selected, dim=-1)
                anchor_logits = target.lm_head(tgt_out.last_hidden_state[:, -1:, :])
                anchor = anchor_logits.argmax(dim=-1).squeeze(1)  # [1]

                if device.type == "mps": torch.mps.synchronize()
                if anchor.item() == tokenizer.eos_token_id:
                    ids = torch.cat([ids, anchor.unsqueeze(1)], dim=1)
                    tokens_gen += 1
                    break

                # Step 2: Single draft forward with mask-filled block
                ctx = ids.shape[1]
                # Build input_ids with anchor at pos 0 (rest can be any valid token id —
                # mask_positions tells the model to replace those with mask_embedding)
                draft_in_ids = torch.zeros(1, BLOCK_SIZE, dtype=torch.long, device=device)
                draft_in_ids[0, 0] = anchor.item()
                mask_pos = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool, device=device)
                mask_pos[:, 1:] = True
                pos_ids = torch.arange(ctx - 1, ctx - 1 + BLOCK_SIZE, device=device).unsqueeze(0)

                draft_logits = draft(
                    input_ids=draft_in_ids,
                    injection_features=inj,  # full context features
                    block_mask=block_mask,
                    position_ids=pos_ids,
                    mask_positions=mask_pos,
                )  # [1, L, V]
                cand = draft_logits.argmax(dim=-1)  # [1, L]
                cand[0, 0] = anchor.item()  # enforce anchor at position 0
                proposed += BLOCK_SIZE - 1
                blocks += 1

                # Step 3: Target verifies the block in one forward
                verify_out = target(
                    input_ids=torch.cat([ids, cand], dim=1),
                    use_cache=False, return_dict=True,
                )
                # verify_logits at absolute positions ctx-1 .. ctx+L-2 predict tokens at
                # absolute positions ctx .. ctx+L-1  (= block positions 0 .. L-1)
                verify_logits = verify_out.logits[:, ctx - 1: ctx + BLOCK_SIZE - 1, :]
                tgt_preds = verify_logits.argmax(dim=-1)  # [1, L]

                # Step 4: Accept longest matching prefix
                n_acc = 0
                for i in range(BLOCK_SIZE):
                    if cand[0, i].item() == tgt_preds[0, i].item():
                        n_acc += 1
                    else:
                        break
                if n_acc == 0:
                    acc_toks = tgt_preds[:, :1]
                    n_acc = 1
                else:
                    acc_toks = cand[:, :n_acc]
                accepted += max(n_acc - 1, 0)
                ids = torch.cat([ids, acc_toks], dim=1)
                tokens_gen += acc_toks.shape[1]

                if device.type == "mps": torch.mps.synchronize()

                # Update state
                elapsed = time.perf_counter() - t0
                current_tps = tokens_gen / max(elapsed, 1e-6)
                state.update_prompt(idx, "dflash",
                    tokens_generated=tokens_gen, blocks_generated=blocks,
                    acceptance_rate=accepted / max(proposed, 1),
                    elapsed_s=elapsed, tok_per_sec=current_tps,
                    speedup=current_tps / max(baseline_tps, 1e-6),
                )

                if tokenizer.eos_token_id in acc_toks[0].tolist():
                    break

            elapsed = time.perf_counter() - t0
            output = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
            state.update_prompt(idx, "dflash",
                status="done", tokens_generated=tokens_gen, blocks_generated=blocks,
                acceptance_rate=accepted / max(proposed, 1),
                elapsed_s=elapsed, tok_per_sec=tokens_gen / max(elapsed, 1e-6),
                speedup=(tokens_gen / max(elapsed, 1e-6)) / max(baseline_tps, 1e-6),
                output=output[:200],
            )

        # --- Run each prompt ---
        for idx, prompt in enumerate(PROMPTS):
            baseline_tps = baseline_generate(prompt, idx)
            dflash_generate(prompt, idx, baseline_tps)

        state.set_stage("done", "All prompts complete — summary generated")
        state.finalize(ok=True)
        print("\n✓ Inference test complete.", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Inference test failed:\n{tb}", flush=True)
        state.finalize(ok=False, error=f"{type(e).__name__}: {e}\n\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
