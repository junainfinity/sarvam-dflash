#!/usr/bin/env python3
"""
V3 DFlash smoke test — gates the 4-day retrain.

Runs ~100 training steps on a tiny subset and checks:
  1. Model forward pass works (no crash, shapes correct)
  2. Loss is finite (not NaN/Inf)
  3. Loss decreases (model is learning)
  4. mask_embedding receives gradient (new parameter is in the graph)

If any check fails → abort. Do NOT launch the full training.
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from dflash_draft import (
    DFlashConfig, DFlashDraftModel,
    make_block_bidirectional_mask, make_position_weights,
)
from dflash_data import DFlashDataset
from train_dflash_sarvam import chunked_cross_entropy, sparse_kl_divergence

TARGET_PATH = "./sarvam-30b"
DATA_DIR    = "./dflash_training_data"
NUM_STEPS   = 100
NUM_SHARDS  = 10
BLOCK_SIZE  = 16
MAX_SEQ_LEN = 2048
LR          = 6e-4


def load_target():
    """Load frozen Sarvam-30B target model."""
    from transformers import AutoTokenizer
    from modeling_sarvam_moe_dflash import SarvamMoEForCausalLMWithKVInjection, SarvamMoEConfig

    print("Loading Sarvam-30B (for shared embed/head only)...", flush=True)
    cfg = SarvamMoEConfig.from_pretrained(TARGET_PATH)
    target = SarvamMoEForCausalLMWithKVInjection.from_pretrained(
        TARGET_PATH, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    target.eval()
    return target


def build_draft(target, device):
    cfg = DFlashConfig(
        num_draft_layers=6, ffn_intermediate=2048, block_size=BLOCK_SIZE,
        rope_theta=8_000_000.0, rms_norm_eps=1e-6, initializer_range=0.02,
    )
    draft = DFlashDraftModel(cfg)
    draft.share_target_embeddings(target)
    draft.kv_fusion  = draft.kv_fusion.to(device, dtype=torch.bfloat16)
    draft.layers     = draft.layers.to(device, dtype=torch.bfloat16)
    draft.norm       = draft.norm.to(device, dtype=torch.bfloat16)
    draft.rotary_emb = draft.rotary_emb.to(device)
    draft.mask_embedding.data = draft.mask_embedding.data.to(device, dtype=torch.bfloat16)
    return draft, cfg


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # --- Load models ---
    target = load_target()
    draft, cfg = build_draft(target, device)
    print(f"Draft built. Trainable params: {draft.count_trainable_parameters():,}", flush=True)

    # --- Tiny dataset ---
    dataset = DFlashDataset(DATA_DIR, max_seq_len=MAX_SEQ_LEN)
    subset = Subset(dataset, list(range(NUM_SHARDS)))
    loader = DataLoader(subset, batch_size=2, num_workers=0, shuffle=True)

    # --- Training setup ---
    trainable = draft.get_trainable_parameters()
    opt = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    block_mask  = make_block_bidirectional_mask(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bfloat16, device=device)
    pw          = make_position_weights(BLOCK_SIZE, device=device)

    losses = []
    mask_grad_seen = False

    # --- Training loop ---
    print(f"\nRunning {NUM_STEPS} training steps (batch_size=2 on {NUM_SHARDS} shards)", flush=True)
    print("-" * 80, flush=True)

    draft.train()
    draft.word_embeddings.eval()
    draft.lm_head.eval()

    step = 0
    t_start = time.perf_counter()
    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break

            input_ids   = batch["input_ids"].to(device)
            inj_feats   = batch["injection_features"].to(device)
            teach_logits = batch["teacher_top_logits"].to(device)
            teach_idxs   = batch["teacher_top_indices"].to(device)

            B, S = input_ids.shape
            L = BLOCK_SIZE
            max_P = S - L
            P = int(torch.randint(0, max_P + 1, (1,)).item())

            block_input_ids = input_ids[:, P:P + L]
            context_feats   = inj_feats[:, :P + 1, :]

            block_teach_l = F.pad(teach_logits[:, P:P + L - 1, :], (0, 0, 1, 0))
            block_teach_i = F.pad(teach_idxs[:, P:P + L - 1, :],   (0, 0, 1, 0))

            mask_pos = torch.zeros(B, L, dtype=torch.bool, device=device)
            mask_pos[:, 1:] = True

            pos_ids = torch.arange(P, P + L, device=device).unsqueeze(0).expand(B, -1)

            logits = draft(
                input_ids=block_input_ids,
                injection_features=context_feats,
                block_mask=block_mask,
                position_ids=pos_ids,
                mask_positions=mask_pos,
            )

            ce_per_pos = chunked_cross_entropy(logits, block_input_ids)
            ce = (ce_per_pos * pw.unsqueeze(0)).sum() / (pw.sum() * B)
            kl_per_pos = sparse_kl_divergence(logits, block_teach_l, block_teach_i)
            kl = (kl_per_pos * pw.unsqueeze(0)).sum() / (pw.sum() * B)
            loss = 0.5 * ce + 0.5 * kl

            if not torch.isfinite(loss):
                print(f"❌ FAIL step={step}: loss is NaN/Inf — CE={ce.item()} KL={kl.item()}", flush=True)
                sys.exit(1)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Verify mask_embedding receives gradient
            if draft.mask_embedding.grad is not None:
                g = draft.mask_embedding.grad.abs().mean().item()
                if g > 1e-8:
                    mask_grad_seen = True

            opt.step()

            losses.append(loss.item())
            step += 1

            if step == 1 or step % 10 == 0 or step == NUM_STEPS:
                elapsed = time.perf_counter() - t_start
                rate = step / elapsed if elapsed > 0 else 0
                print(f"  step={step:3d}  loss={loss.item():.4f}  "
                      f"(ce={ce.item():.4f}, kl={kl.item():.4f})  P={P}  rate={rate:.2f}/s",
                      flush=True)

    # --- Verdict ---
    print("-" * 80, flush=True)

    loss_0  = losses[0]
    loss_10 = sum(losses[:10]) / 10
    loss_last = sum(losses[-10:]) / 10

    print(f"\nResults:")
    print(f"  loss  (step 1):  {loss_0:.4f}")
    print(f"  loss (avg 1-10): {loss_10:.4f}")
    print(f"  loss (avg last 10): {loss_last:.4f}")
    print(f"  loss delta: {loss_10 - loss_last:+.4f}")
    print(f"  mask_embedding received gradient: {mask_grad_seen}")

    checks = {
        "loss is finite throughout":              all(torch.tensor(l).isfinite() for l in losses),
        "loss at step 1 < 15 (sane init)":         loss_0 < 15,
        "loss decreased from start to end":        loss_last < loss_10,
        "mask_embedding received gradient > 1e-8": mask_grad_seen,
    }

    print("\nChecks:")
    all_pass = True
    for name, ok in checks.items():
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {name}")
        if not ok:
            all_pass = False

    print("")
    if all_pass:
        print("🎉 SMOKE TEST PASSED — safe to launch full training.")
        sys.exit(0)
    else:
        print("🛑 SMOKE TEST FAILED — do NOT launch full training. Investigate above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
