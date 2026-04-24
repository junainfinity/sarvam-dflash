# PLAN_V3.md — The Corrected DFlash Retrain

**Author:** this plan was written before launching the V3 training run.
**Date:** 2026-04-24
**Preconditions:** V2 training completed with 0% acceptance. Diagnosis in `DEBUGGING_LOG.md`. Architecture spec in `ARCHITECTURE.md`.

---

## 1. The Goal

**Train a draft model for Sarvam-30B that achieves a measurable speedup via DFlash block-speculative decoding on M3 Max.**

Success criteria (in priority order):
1. **Acceptance length ≥ 3 tokens per 16-wide block** on a held-out prompt set. Published DFlash on Qwen3-8B hits 5.8–6.5; even half of that is a valid result on Sarvam-30B.
2. **End-to-end speedup ≥ 1.5×** vs baseline Sarvam-30B on short generations (64–128 tokens). Published: 4–6×.
3. **Zero quality loss** — speculative decoding provably preserves the target distribution; this is architecturally guaranteed, not a tunable.

Out of scope for V3:
- MLX conversion (only after V3 passes inference validation).
- Full MMLU benchmark (only after inference validation).
- Paper-level speedups (4–6×) — a stretch on M3 Max with our cross-attn topology.

---

## 2. The Plan (5 phases, gated)

### Phase 1 — Architecture fix (≈1 hour of code changes)

**`dflash_draft.py`:**
- Add `mask_embedding: nn.Parameter` of shape `[hidden_size]`, init `N(0, 0.02²)`.
- Add `make_block_bidirectional_mask(seq_len, block_size)` — identical to the current block-diagonal mask but with `torch.ones(block_size, block_size)` instead of `torch.tril(...)`.
- Extend `DFlashDraftModel.forward()` to accept `mask_positions: Optional[BoolTensor[B, S]]`. Where True, replace `word_embeddings(input_ids)` with the broadcast `mask_embedding`.
- Expose the mask embedding in `get_trainable_parameters()` so the optimizer picks it up.

**`train_dflash_sarvam.py`:**
- Per batch: sample one random anchor `P ∈ [0, seq_len - block_size]` per sample.
- Slice the sample to just the block: `input_ids[:, P:P+block_size]` (length 16).
- Slice injection features to context-up-to-anchor: `injection_features[:, :P+1, :]`. Pad to the batch's max anchor+1 for batched tensor shape.
- Build `mask_positions` — True at positions `1..block_size-1` of the block, False at position 0 (the anchor).
- Self-attention: use the bidirectional mask.
- Loss: **same-position** CE at block positions `1..block_size-1` (no shift). Sparse KL at the same positions against teacher top-64 logits (sliced to `[:, P+1:P+block_size, :]` per sample). Exponentially-decaying weights, anchor (position 0) weight 0. KL weight 0.5 (unchanged).

All other infrastructure (optimizer, scheduler, resume, SIGTERM handling, checkpoint saving) is unchanged.

### Phase 2 — Smoke test (≈15 minutes, gates Phase 3)

Run `smoke_test.py`:
- Build the draft model with the new architecture.
- Pull 10 shards of training data.
- Run 100 training steps.

Abort-if conditions (any one → STOP, don't launch 4-day training):
- Any loss is NaN / Inf.
- `mask_embedding.grad` is `None` after backward (means it's not actually in the graph).
- Loss after 100 steps is not lower than loss at step 10 (means the model isn't learning).
- Any shape / device error.

Pass conditions (all must hold):
- Loss at step 0: ~12.5 (random over 262144-vocab).
- Loss at step 100: < 10 (at least a small improvement).
- `mask_embedding.grad.abs().mean() > 1e-6` (mask embedding is receiving gradient).
- No crashes.

### Phase 3 — Full training launch (≈4 days wall-clock)

```bash
./launch_v3.sh   # nohup, writes PID file, redirects to dflash_training_v3.log
```

- Config unchanged from V2: `batch_size 2, grad_accum 8, lr 6e-4, warmup 5%, 4 epochs, kl_weight 0.5`.
- Per-step compute is much lower (16 tokens/sample vs 2048), so wall-clock may be faster than V2. Monitor.
- Auto-save every 50 steps. Graceful SIGTERM. All V2 resilience intact.
- **Mid-training safety valve:** every 500 steps, log an acceptance-rate probe on a single held-out prompt. If the probe is 0% at step 2000, abort — something is wrong.

### Phase 4 — Inference validation (≈30 minutes, after training)

Run updated `quick_test.py` with the corrected inference path:
- Input: `[anchor, m, m, ..., m]` (length 16, one learned mask embedding repeated 15 times)
- Target features: full context up to current position
- Bidirectional within-block mask
- Single draft forward per block
- Target verifies, accept longest matching prefix

Measured on 2–3 prompts:
- **Accept**: acceptance rate > 20% — claim success, proceed to Phase 5.
- **Accept with caveats**: acceptance rate 5–20% — partial success, document and move on.
- **Reject**: acceptance rate < 5% — open forensic issue, do not benchmark.

### Phase 5 — Full benchmark (≈2 hours)

Run `benchmark/run_benchmark.py` — the existing script, with inference already fixed during debugging. MMLU prompts, both baseline and DFlash modes, 10 questions.

---

## 3. Scrutiny against the papers

**Primary references:**
- DFlash (`arXiv 2602.06036`) — Chen, Liang, Liu, Z-Lab, Feb 2026
- BD3-LMs / Block Diffusion (`arXiv 2503.09573`) — Arriola et al., ICLR 2025 Oral
- `vllm-project/speculators` RFC #248 — most precise algorithmic spec for DFlash
- `github.com/z-lab/dflash` — reference implementation

| DFlash paper says | Our V3 plan does | Match? | If different — why? |
|---|---|---|---|
| Learned mask embedding at non-anchor positions | Single `nn.Parameter([hidden_size])`, broadcast at non-anchor positions | ✅ | — |
| Bidirectional within-block self-attention | `torch.ones(block_size, block_size)` mask | ✅ | — |
| Same-position mask-fill loss: `logit[i]` predicts `token[i]` | CE(draft_logits[P+i], input_ids[P+i]), no shift | ✅ | — |
| Anchor has weight 0 in loss | Unchanged from V1/V2 (already correct) | ✅ | — |
| Exponentially-decaying position weights | Unchanged from V1/V2 (already correct) | ✅ | — |
| KV injection from 5 uniformly-spaced target layers | Unchanged from V2: `[0, 4, 9, 13, 18]` | ✅ | — |
| Block size 16 | Unchanged | ✅ | — |
| Sparse KL distillation (top-k teacher logits) | Unchanged from V1/V2 (top-64) | ✅ | — |
| Single draft forward per block at inference | Correct inference in `quick_test.py` | ✅ | Inference code already matches paper |
| K/V = `[target_features ‖ mask_tokens]`, `[L, 2L]` bidirectional self-attn | Separate cross-attention layer per block, queries attend to projected target features via separate op | ⚠️ minor divergence | Structurally different, functionally equivalent. Target features reach the draft's residual stream either way. More parameters in our version (extra `q_proj`, `dense` per layer), but no correctness impact. Keeping our current cross-attn topology saves a substantial refactor. |
| Prefill KV cache for target features (written once, persists) | Recompute target features each drafting iteration | ⚠️ efficiency only | Our inference re-runs target per block to extend features to the growing context. Correct, just not the most efficient. Optimize post-validation if speedup is tight. |
| Training: feature leakage prevented via data generation | We slice stored features `[:, :P+1, :]` per random anchor — same effect | ✅ | Reuses the 3.9 TB already on disk. Causal computation of features means the slice is valid. |

**No scrutinized bullet is unresolved.** The two `⚠️` items are intentional simplifications; they preserve correctness and trade compute/params for engineering simplicity.

### Additional paper checks (things not listed above)

- **Tokenizer / vocab:** DFlash paper uses Qwen3-8B tokenizer (150k vocab). We use Sarvam-30B (262k vocab). The sparse-KL (top-64) setup was designed exactly for this — no issue.
- **Draft depth:** paper uses a shallow draft (≤6 layers typically). Our 6-layer draft is in range.
- **Position weighting parameter γ:** paper picks a value around 4; our current code uses `γ ≈ 5` per our training config — fine, in range.
- **Teacher alignment:** paper trains draft against target of the same family. Sarvam is the teacher; we use its top-64 logits directly. Matches.

---

## 4. Why this time is different

| Last time | This time |
|---|---|
| Implemented architecture from my (incorrect) mental model of the paper. | Re-implemented from the paper PDF, the reference repo, and the speculators RFC #248. Cross-referenced all three. |
| Trusted README citation (`arXiv 2503.07807`) without reading the paper. | Verified the real DFlash paper is `arXiv 2602.06036`. README now corrected. |
| No inference smoke test during training. Loss went down → assumed success. | `smoke_test.py` runs before launch (checks learning signal + grad path for mask embedding). Mid-training acceptance probe every 500 steps. |
| 0% acceptance not caught until 4 days of wall-clock had passed. | Acceptance probe fires within hours. Stop signal if still 0% at step 2000. |
| Attention mask was causal within block. | Explicitly bidirectional via `make_block_bidirectional_mask`, named for its behavior. |
| Loss was shifted next-token. | Same-position mask-fill, no shift. Documented inline. |
| No mask embedding — non-anchor positions held either real tokens (training) or zeros (inference). Enormous distribution shift. | Learned `mask_embedding` parameter at non-anchor positions in **both** training and inference. Matches paper. |
| Full-sequence target features → unmasked cross-attn → leaked target labels into conditioning. | Per-anchor feature slicing `features[:, :P+1, :]` — no leakage, no data regeneration needed. |
| Documentation was aspirational ("here's how it should work"). | Documentation is now forensic + forward-looking (`ARCHITECTURE.md`, `DEBUGGING_LOG.md`, this file, updated `HANDOFF.md`). |

---

## 5. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Smoke test passes but full training still gives 0% acceptance | low | Mid-training acceptance probe; abort at step 2000 if still 0%. |
| Per-anchor sampling gives weak gradient signal (1 block per sample vs 128 in V2) | medium | Per-step compute is ~50× lighter; even 4 epochs provide ~200k × 15 = 3M token-predictions. Paper's DFlash trained on comparable volumes. |
| Mask embedding doesn't receive gradient (graph bug) | medium | Smoke test explicitly checks `mask_embedding.grad.abs().mean() > 1e-6` after 10 steps. |
| Device-map shuffle on MPS for the new mask_embedding param | low | Pattern-match existing `kv_fusion.kv_proj.weight.device` usage; add mask_embedding the same way. |
| Data loading becomes the bottleneck at 50× higher step throughput | medium | Monitor `step/sec`. If it's CPU-bound, increase `num_workers`. |
| Laptop sleeps during 4-day run | low | `caffeinate -i` wrapper in launch script; disable sleep in System Settings. |

---

## 6. Files touched / created

**Modified:**
- `dflash_draft.py` — mask embedding, bidirectional mask, input_embeds path
- `train_dflash_sarvam.py` — per-anchor sampling, feature slicing, mask-fill loss, mid-training probe

**New:**
- `smoke_test.py` — pre-launch validation
- `status.sh` — terminal dashboard for user to check progress after closing Claude
- `launch_v3.sh` — nohup launcher with caffeinate, PID tracking, log redirect
- `probe_acceptance.py` — called by the mid-training probe; single-prompt inference acceptance measurement
- `PLAN_V3.md` — this file

**Unchanged:**
- `dflash_data.py` (data already generated and valid)
- `modeling_sarvam_moe_dflash.py` (target-side hooks already correct)
- `benchmark/run_benchmark.py` (already has the inference fix; just waiting for a working checkpoint)

---

## 7. Decision record

- Launch V3 retrain **if and only if** smoke test passes all 4 check conditions.
- If smoke test fails: log the failure mode, do not start the 4-day run.
- If mid-training probe at step 2000 shows 0% acceptance: save state, kill process, investigate.
