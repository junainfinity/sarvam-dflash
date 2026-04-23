# HANDOFF.md — DFlash / Sarvam-30B Operational Guide

**Last updated:** 2026-04-24
**Maintained by:** arjun / junainfinity
**GitHub:** https://github.com/junainfinity/sarvam-dflash
**Hardware:** Apple M3 Max MacBook Pro, 128 GB unified memory

> ⚠️ **CRITICAL — READ BEFORE TOUCHING CODE:** After the 4-day training completed successfully (best loss 0.8600), end-to-end testing revealed **0% acceptance rate** during speculative decoding. Root cause: our README cited the wrong paper, and our training has **three architectural mismatches** vs. the real DFlash algorithm. The current trained checkpoint **cannot be used for speculative decoding as intended** — the model must be retrained with corrected objective/mask/mask-embedding. See [`DEBUGGING_LOG.md`](DEBUGGING_LOG.md) for the full post-mortem, the three bugs, and the exact fix specification. **Do NOT proceed to MLX conversion or benchmark with the current checkpoint — it will not work.**

---

## 1. PROJECT GOAL

Build a **DFlash speculative decoding** system for **Sarvam-30B**, a 30B-parameter Mixture-of-Experts language model. The goal is to generate text significantly faster without any quality loss.

**How it works:**
1. A tiny ~275M-parameter *draft model* proposes a 16-token block in parallel, using a diffusion-style masked prediction approach
2. The frozen Sarvam-30B *target model* verifies the entire block in one forward pass
3. The speculative decoding algorithm accepts the longest correct prefix — guaranteeing the output distribution is identical to running Sarvam-30B alone
4. Net result: 2–4× throughput improvement on Apple Silicon

The draft model is trained via **offline knowledge distillation** — Sarvam-30B runs once over the training corpus to extract hidden states and teacher logits, which are stored as shards. The draft model then trains against these stored features without needing Sarvam-30B loaded during training.

---

## 2. ARCHITECTURE

### Sarvam-30B (frozen teacher)
- 30B parameter Mixture-of-Experts transformer
- 19 decoder layers, hidden_dim=4096, vocab_size=262,144
- Weights: `./sarvam-30b/` (~60 GB, not in git)
- Modified by `modeling_sarvam_moe_dflash.py` to expose KV injection hooks

### Draft Model (`dflash_draft.py`)
- **6 transformer layers**, ~275M trainable parameters
- Shares frozen word embeddings (1.07B params) and LM head (1.07B params) from Sarvam-30B
- Per-layer structure: `self-attn → cross-attn → SwiGLU FFN`
- **Self-attention:** 16 Q heads, 4 KV heads (GQA 4:1), head_dim=64, RoPE θ=8M, QK-norm
- **Cross-attention:** 16 Q heads, 4 KV heads — attends to injection features from Sarvam-30B
- **FFN:** SwiGLU, intermediate_dim=2048
- **Block-diagonal causal mask:** each 16-token block attends only within itself
- **KV Fusion:** single linear projection `20480 → 512` shared across all layers

### KV Injection
Sarvam-30B hidden states from 5 uniformly-spaced layers are extracted and concatenated:
```
KV_INJECTION_LAYER_INDICES = [0, 4, 9, 13, 18]
injection_features: [B, seq_len, 20480]   # 5 layers × 4096 hidden_dim
```
These are projected to cross-attention K/V for the draft model.

### Training Loss (50/50 weighted)
- **Position-weighted cross-entropy:** exponentially decaying weights within each 16-token block; the first (anchor) token in each block has weight 0
- **Sparse KL divergence:** computed only over the teacher's top-64 token vocabulary indices (memory-efficient for vocab_size=262,144)

### Training Config
| Parameter | Value |
|-----------|-------|
| Epochs | 4 |
| Learning rate | 6e-4 |
| LR schedule | Cosine with 5% linear warmup |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| Effective batch size | 16 (batch_size=2 × grad_accum=8) |
| Gradient clip | 1.0 |
| KL weight | 0.5 |
| Block size | 16 |
| Sequence length | 2048 |
| Save interval | Every 50 optimizer steps |

---

## 3. KEY FILES

| File | What it does |
|------|-------------|
| `dflash_data.py` | Phase 1: runs frozen Sarvam-30B over fineweb-edu-score-2, extracts KV features + teacher logits, saves safetensors shards |
| `dflash_draft.py` | Draft model architecture: `DFlashDraftModel`, `DFlashDecoderLayer`, `DFlashKVFusion`, `make_block_causal_mask()` |
| `train_dflash_sarvam.py` | Phase 2: full training loop with pause/resume, warm-start via `--pretrain_checkpoint`, auto-save every 50 steps, graceful SIGTERM handler |
| `modeling_sarvam_moe_dflash.py` | Patches Sarvam-30B with `get_kv_injection_features()`, `get_teacher_logits()`, `get_injection_and_logits()` |
| `run_datagen.sh` | nohup launcher for data generation — PID-tracked, resume-safe, logs to `dflash_datagen.log` |
| `run_train.sh` | Runs Phase 1 then Phase 2 in sequence; skips Phase 1 if shards already exist |
| `dflash_mlx/model.py` | Full MLX reimplementation of draft model for Apple Silicon inference |
| `dflash_mlx/convert.py` | Converts PyTorch checkpoint + Sarvam-30B shared weights → `weights.npz` + `config.json` |
| `dflash_mlx/load.py` | Loads converted MLX model and runs a smoke-test forward pass |
| `dflash_mlx/config.json` | Model config (already populated from prior conversion run) |
| `dflash_mlx/weights.npz` | Converted MLX weights (large, not in git) |
| `benchmark/run_benchmark.py` | Phase 4: measures TTFT, tok/s, acceptance rate, speedup ratio (baseline vs speculative) |
| `benchmark/mmlu_questions.py` | MMLU question loader for the benchmark |
| `benchmark/results/` | Benchmark JSON output and markdown report |
| `requirements.txt` | Python dependencies: torch, transformers, datasets, accelerate, safetensors, wandb, tqdm |
| `MLX_CONVERSION.md` | Step-by-step guide for MLX conversion and quantization |

### Directories not in git
| Directory | Contents | Size |
|-----------|----------|------|
| `dflash_training_data/` | 50,000 safetensors shards | ~3.9 TB |
| `sarvam-30b/` | Sarvam-30B model weights | ~60 GB |
| `checkpoints/` | Draft model training checkpoints | ~6 GB |
| `checkpoints_v1/` | Archived V1 training run (6 epochs on 5,993 shards) | ~11 GB |

---

## 4. COMPLETE RUN HISTORY

### Phase 1: Data Generation

The goal was to generate 50,000 safetensors shards from Sarvam-30B over the fineweb-edu-score-2 dataset. Each shard (~84.9 MB) contains:
```
input_ids:            [1, 2048]        int32
injection_features:   [1, 2048, 20480] bfloat16
teacher_top_logits:   [1, 2048, 64]    float32
teacher_top_indices:  [1, 2048, 64]    int32
```

The script is deterministic and resume-safe: it re-iterates the HuggingFace streaming dataset from the beginning on restart but skips writing any shards that already exist. Because `streaming=True` with no shuffle reads Parquet files in fixed lexicographic order, shard N always corresponds to the same sample N across all runs.

| Run | PID | Date | Outcome |
|-----|-----|------|---------|
| Run 0 | (pre-log) | pre-Apr 16 | Generated shards 0–1,308 |
| Run A | (pre-log) | pre-Apr 16 | Found 1,309 existing, generated 1,309–5,991 |
| Attempt 1 | 2054 | Apr 16 12:32:37 | ❌ Crashed: `ModuleNotFoundError: No module named 'datasets'`. Fix: `pip install datasets` |
| Attempt 2 | 2185 | Apr 16 12:32:59 | ❌ Crashed: `ValueError: accelerate not installed`. Fix: `pip install accelerate` |
| **Final run** | **2316** | **Apr 16 12:33:35** | ✅ Found 5,992 existing shards, ran 36h 24m to completion at shard 50,000 |

**Verification performed:** all 50,000 shards present, zero gaps, zero duplicates, uniform 84.9 MB file sizes, distinct `input_ids` per shard. Dataset is clean and self-consistent.

**Total dataset size:** 3.9 TB in `dflash_training_data/`

---

### Phase 2: Training

#### V1 Training (on partial ~5,993 shards)
- Ran 6 epochs, 2,247 total optimizer steps
- Best loss: **0.9364** (CE: 1.774, KL: 0.099)
- All checkpoints archived to `checkpoints_v1/`:
  - `dflash_draft_best.pt` — 524 MB, model weights only (74 tensors)
  - `dflash_draft_epoch_1.pt` through `dflash_draft_epoch_6.pt` — 1.5 GB each, full state

#### V2 Cold Start (PID 50777) — abandoned
- Started training from random init on full 50k dataset, 6 epochs (18,750 total steps)
- Ran ~11 hours to step 1,160 (6.2%)
- Step 10 loss: **6.46**
- Decision: abort and warm-start from V1 instead. The V1 architecture is identical (same 74 tensors, same shapes), so weights transfer directly.
- **Code change made:** Added `--pretrain_checkpoint` argument to `train_dflash_sarvam.py`. It loads model weights only (no optimizer/scheduler state), allowing the LR schedule to reset fresh for the new dataset size.

#### V2 Warm Start Attempt 1 (PID 65609) — epoch reduction
- Loaded V1 `checkpoints_v1/dflash_draft_best.pt` via `--pretrain_checkpoint` (74/74 tensors)
- Fresh optimizer and scheduler; 6 epochs, 18,750 total steps
- Step 10 loss: **0.42** — 15× lower than cold start at the same step
- Ran to step ~178, then user requested reducing to **4 epochs** to save time
- Killed gracefully via SIGTERM; training script's signal handler saved a full checkpoint at step 178 before exit
- **LR schedule note:** the `lr_lambda` closure is not stored in checkpoints — only `last_epoch`. Changing `--epochs 4` creates a new cosine schedule with `total_steps=12,500` and `warmup_steps=625`. The restored `last_epoch=178` correctly positions within the new schedule (still in warmup), causing only a small LR discontinuity (~1.14e-4 → ~1.72e-4).

#### V2 Warm Start Attempt 2 — CURRENT RUN
- **PID: 68181**
- **Started:** ~Apr 18 12:02 (resumed from step 178 checkpoint)
- **Command:**
  ```bash
  nohup python3 train_dflash_sarvam.py \
      --target_model_path ./sarvam-30b \
      --data_dir ./dflash_training_data \
      --output_dir ./checkpoints \
      --num_draft_layers 6 --ffn_intermediate 2048 --block_size 16 \
      --max_seq_len 2048 --batch_size 2 --grad_accum 8 --lr 6e-4 \
      --epochs 4 --warmup_ratio 0.05 --kl_weight 0.5 \
      --save_interval 50 --log_interval 10 --num_workers 4 \
      >> dflash_training.log 2>&1 &
  ```
- **Note:** No `--pretrain_checkpoint` in this command — the V1 warm-start weights are already baked into `checkpoints/resume_checkpoint.pt` from attempt 1. The script auto-resumes from that checkpoint.
- **Progress as of Apr 22:** Step ~9,170 / 12,500 (73.4%), Epoch 3/4, LR ~1.09e-04
- **Rate:** ~37 s/step
- **ETA:** ~34h → ~Apr 23

**Current checkpoints:**
| File | Size | Contents |
|------|------|---------|
| `checkpoints/dflash_draft_best.pt` | 524 MB | Best model so far (weights only) |
| `checkpoints/dflash_draft_epoch_1.pt` | 1.5 GB | Full state at end of epoch 1 |
| `checkpoints/dflash_draft_epoch_2.pt` | 1.5 GB | Full state at end of epoch 2 |
| `checkpoints/dflash_draft_interrupted_e0_b1425.pt` | 1.5 GB | Full state from interrupted cold-start run |
| `checkpoints/resume_checkpoint.pt` | 1.5 GB | Latest full state (updated every 50 steps) |

---

## 5. HOW TO CHECK CURRENT STATE

```bash
# Is training running?
ps aux | grep train_dflash | grep -v grep

# Check current step and loss
tail -30 dflash_training.log

# Precise status
ps -p 68181 -o pid,etime,command && tail -15 dflash_training.log

# Count training shards (should be 50000)
find dflash_training_data -name 'shard_*.safetensors' | wc -l

# List checkpoints by recency
ls -lt checkpoints/ | head -10

# Confirm best checkpoint exists
ls -lh checkpoints/dflash_draft_best.pt
```

### If PID 68181 is gone

Check `tail -100 dflash_training.log` for what happened:

**If training completed normally** — log ends with something like `Training complete. Best loss: X.XXXX` and both `checkpoints/dflash_draft_best.pt` and `checkpoints/dflash_draft_final.pt` will exist. Proceed to Phase 3.

**If it crashed** — `checkpoints/resume_checkpoint.pt` will exist at the last 50-step save. Restart with the exact same command above (no `--pretrain_checkpoint` — the resume checkpoint already has warm-started weights):
```bash
nohup python3 train_dflash_sarvam.py \
    --target_model_path ./sarvam-30b \
    --data_dir ./dflash_training_data \
    --output_dir ./checkpoints \
    --num_draft_layers 6 --ffn_intermediate 2048 --block_size 16 \
    --max_seq_len 2048 --batch_size 2 --grad_accum 8 --lr 6e-4 \
    --epochs 4 --warmup_ratio 0.05 --kl_weight 0.5 \
    --save_interval 50 --log_interval 10 --num_workers 4 \
    >> dflash_training.log 2>&1 &
echo "PID: $!"
```
The script auto-detects `checkpoints/resume_checkpoint.pt` and resumes from the last saved step.

---

## 6. HOW TO FINISH THE PROJECT

Once training completes (step 12,500 / epoch 4 done):

### Step 1: Verify training output
```bash
ls -lh checkpoints/dflash_draft_best.pt   # must exist (~524 MB)
ls -lh checkpoints/dflash_draft_final.pt  # must exist (~524 MB)
tail -20 dflash_training.log              # should show completion + final best loss
```

### Step 2: MLX Conversion
Converts the PyTorch checkpoint to MLX format for native Apple Silicon inference.
```bash
cd dflash_mlx
python3 convert.py \
    --checkpoint ../checkpoints/dflash_draft_best.pt \
    --sarvam ../sarvam-30b \
    --output .
# Produces: weights.npz (~4.8 GB fp16) and updates config.json
```
For optional INT4 quantization (reduces to ~1.2 GB), see `MLX_CONVERSION.md`.

### Step 3: Smoke Test
```bash
cd dflash_mlx
python3 load.py
# Should print logit shape and value range with no errors
```

### Step 4: Benchmark
```bash
cd /Users/arjun/Projects/sarvam-dflash
python3 benchmark/run_benchmark.py \
    --mode both \
    --target_model_path ./sarvam-30b \
    --draft_checkpoint ./checkpoints/dflash_draft_best.pt \
    --num_questions 10
```
Results saved to `benchmark/results/benchmark_results.json`.

**Important:** Sarvam-30B is a pretrained base model (not instruction-tuned). MMLU prompts without few-shot formatting will likely produce poor accuracy for both baseline and speculative runs. The meaningful benchmark metric is **tok/s speedup** and **acceptance rate**, not accuracy.

### Step 5: Update README and push to GitHub
```bash
# Update README.md training results table with final loss values
# Then:
git add README.md HANDOFF.md benchmark/results/
git commit -m "Training complete: final results and benchmark"
git push origin main
```

---

## 7. KNOWN ISSUES & GOTCHAS

- **`pin_memory` warning on MPS:** `UserWarning: 'pin_memory' argument is set as true but not supported on MPS` — non-critical, MPS doesn't support pinned memory. Training proceeds normally.
- **`ls shard_*.safetensors | wc -l` fails** with 50k files ("argument list too long"). Use `find dflash_training_data -name 'shard_*.safetensors' | wc -l` instead.
- **Resume checkpoint overrides warm-start:** The `--pretrain_checkpoint` code runs *before* the resume check in `train_dflash_sarvam.py`. If a `resume_checkpoint.pt` already exists, the resume will overwrite the warm-start weights. Solution when warm-starting fresh: delete `resume_checkpoint.pt` first (as was done for PID 65609).
- **V1 optimizer state is not portable:** V1 trained on 2,247 steps (5,993 shards × 6 epochs). V2 has 12,500 total steps (50,000 shards × 4 epochs). Only model weights transfer — optimizer/scheduler must reset.
- **Sarvam-30B dynamic import:** The model code in `sarvam-30b/` uses relative imports that break when loaded from outside the package. `modeling_sarvam_moe_dflash.py` handles this with a custom dynamic loader that patches relative imports.
- **MMLU benchmark caveat:** Sarvam-30B is a pretrained base model. Accuracy numbers are not meaningful without proper few-shot formatting. Focus on latency/throughput metrics.

---

## 8. FULL DIRECTORY STRUCTURE

```
sarvam-dflash/
├── dflash_data.py                   # Phase 1: offline extraction from Sarvam-30B → safetensors shards
├── dflash_draft.py                  # Draft model architecture (DFlashDraftModel, DFlashDecoderLayer, etc.)
├── train_dflash_sarvam.py           # Phase 2: training loop, pause/resume, --pretrain_checkpoint warm-start
├── modeling_sarvam_moe_dflash.py    # Sarvam-30B patched with KV injection + top-k logit extraction hooks
├── run_datagen.sh                   # nohup launcher for dflash_data.py (PID-tracked, resume-safe)
├── run_train.sh                     # Runs Phase 1 then Phase 2 in sequence; skips Phase 1 if data exists
├── requirements.txt                 # torch, transformers, datasets, accelerate, safetensors, wandb, tqdm
├── README.md                        # Architecture docs, quick start, training results
├── HANDOFF.md                       # This file — operational log and takeover guide
├── MLX_CONVERSION.md                # Guide for converting to MLX format + optional INT4 quantization
│
├── dflash_mlx/                      # MLX reimplementation for Apple Silicon inference
│   ├── model.py                     # Full MLX draft model (RMSNorm, RoPE, KVFusion, attn, FFN, etc.)
│   ├── convert.py                   # PyTorch checkpoint + Sarvam-30B shared weights → weights.npz + config.json
│   ├── load.py                      # Loads converted MLX model; runs smoke-test forward pass
│   ├── config.json                  # Model hyperparameters (populated after first conversion run)
│   └── weights.npz                  # Converted MLX weights — large file, not in git
│
├── benchmark/
│   ├── run_benchmark.py             # Measures TTFT, tok/s, acceptance rate, speedup (baseline vs speculative)
│   ├── mmlu_questions.py            # MMLU question loader for benchmark prompts
│   └── results/
│       ├── benchmark_results.json   # Raw benchmark output (JSON)
│       └── BENCHMARK_REPORT.md      # Human-readable benchmark summary
│
├── dflash_training_data/            # NOT IN GIT — 50,000 safetensors shards, ~3.9 TB total
│   └── shard_000000.safetensors     # ... through shard_049999.safetensors (~84.9 MB each)
│
├── sarvam-30b/                      # NOT IN GIT — Sarvam-30B model weights (~60 GB)
│   └── ...                          # HuggingFace model files (config.json, *.safetensors, tokenizer, etc.)
│
├── checkpoints/                     # NOT IN GIT — V2 training checkpoints (current run)
│   ├── dflash_draft_best.pt         # Best checkpoint so far — model weights only (524 MB)
│   ├── dflash_draft_epoch_1.pt      # Full state at end of epoch 1 (1.5 GB)
│   ├── dflash_draft_epoch_2.pt      # Full state at end of epoch 2 (1.5 GB)
│   ├── dflash_draft_interrupted_e0_b1425.pt  # Interrupted cold-start checkpoint (1.5 GB)
│   └── resume_checkpoint.pt         # Latest full state, updated every 50 steps (1.5 GB)
│
└── checkpoints_v1/                  # NOT IN GIT — V1 training run archive (DO NOT DELETE)
    ├── dflash_draft_best.pt         # V1 best — model weights only, loss=0.9364 (524 MB)
    ├── dflash_draft_epoch_1.pt      # through epoch_6.pt — V1 full states (1.5 GB each)
    └── resume_checkpoint.pt         # V1 final resume state (epoch 5, step 2244)
```

---

## 9. PIPELINE STATUS SUMMARY

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Data Generation | ✅ Complete | 50,000 shards, 3.9 TB, verified clean. Reusable for retrain (causal features can be sliced per-anchor). |
| Phase 2: Training (V2, 4 epochs) | ✅ Complete but **invalid** | Best loss 0.8600, but trained with the wrong objective. See `DEBUGGING_LOG.md`. |
| Phase 2.5: Architecture fixes | ⏳ **Required next** | Add `mask_embedding`, bidirectional within-block mask, same-position mask-fill loss, per-anchor feature slicing. Fully specified in `DEBUGGING_LOG.md §5`. |
| Phase 2.6: Retrain V3 | ⏳ **Blocked on Phase 2.5** | Cannot warm-start from V2 (new architecture). ~4 days wall-clock. |
| Phase 3: MLX Conversion | ⏳ Blocked on V3 retrain | Current `weights.npz` was deleted during cleanup — wrong architecture. |
| Phase 4: Benchmark | ⏳ Blocked on V3 retrain | Will show 0% acceptance with current weights. |

### Disk state after 2026-04-24 cleanup

| Directory | Contents | Size |
|-----------|----------|------|
| `checkpoints/` | `dflash_draft_best.pt` only (V2, loss 0.8600) — historical record | 524 MB |
| `checkpoints_v1/` | `dflash_draft_best.pt` only (V1, loss 0.9364) — historical record | 524 MB |
| `dflash_mlx/` | Source code only; `weights.npz` deleted | 28 KB |
| `dflash_training_data/` | Untouched — 50,000 shards, reusable | ~3.9 TB |
| `sarvam-30b/` | Untouched — target model weights | ~60 GB |

~23 GB freed by deleting V1/V2 full-state checkpoints, interrupted/final duplicates, and the invalid MLX weights.
