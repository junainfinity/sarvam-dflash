# sarvam-dflash

A DFlash (block diffusion speculative decoding) pipeline for [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-m), a 30B parameter Mixture-of-Experts language model. This project trains a lightweight ~275M parameter draft model that generates 16-token blocks in parallel, conditioned on KV-injected hidden states from the frozen Sarvam-30B target — enabling speculative decoding without modifying the target model.

Based on the offline distillation approach from ["Training Domain Draft Models for Speculative Decoding"](https://arxiv.org/abs/2503.07807) (arXiv 2503.07807), which is 11–25% better than online distillation.

**Hardware:** M3 Max MacBook Pro (128GB unified memory). Sarvam-30B runs fully in bfloat16 on MPS/CPU.

> **Operational details:** For a full run history, current training status, crash recovery instructions, and step-by-step completion guide, see [`HANDOFF.md`](HANDOFF.md).

---

## Architecture

### Phase 1 — Offline Data Generation (`dflash_data.py`)

The frozen Sarvam-30B target model runs a single forward pass per sample from the `HuggingFaceFW/fineweb-edu-score-2` dataset, extracting two things:

**KV injection features** — hidden states from 5 uniformly-spaced layers (layers 0, 4, 9, 13, 18) concatenated along the feature axis:
```
injection_features: [B, seq_len, 20480]   # 5 layers × 4096 hidden dim
```

**Teacher top-k logits** — top-64 logits and their vocabulary indices for white-box distillation:
```
teacher_top_logits:  [B, seq_len, 64]    # float32
teacher_top_indices: [B, seq_len, 64]    # int32
```

Each sample is saved as a `shard_NNNNNN.safetensors` file containing all four tensors (`input_ids`, `injection_features`, `teacher_top_logits`, `teacher_top_indices`). Shards are ~85 MB each. The full 50,000-sample dataset is ~4.25 TB.

The extraction is resume-safe: on restart, the script counts existing shards and skips ahead, so no work is lost on crash or interruption.

### Phase 2 — Draft Model (`dflash_draft.py`, `train_dflash_sarvam.py`)

The draft model is a 6-layer dense transformer with ~275M trainable parameters. It shares the frozen word embeddings (1.07B params) and LM head (1.07B params) from Sarvam-30B.

**Per-layer architecture:**
```
x = x + self_attn(norm(x))        # GQA self-attention with block-diagonal causal mask
x = x + cross_attn(norm(x), kv)   # Cross-attention to target KV features
x = x + ffn(norm(x))              # SwiGLU FFN
```

Key design choices:
- **Self-attention:** 16 Q heads, 4 KV heads (4:1 GQA), head_dim=64, RoPE (θ=8M), QK-norm
- **Cross-attention:** 16 Q heads, 4 KV heads, attends to full target context (no mask)
- **KV fusion:** Single linear projection `20480 → 512` (2 × 4 heads × 64 dim) shared across all 6 layers
- **Block-diagonal causal mask:** Each block of 16 tokens can only attend within its block
- **FFN:** SwiGLU with intermediate dim 2048

**Training loss** (50/50 weighted sum):
- Position-weighted cross-entropy: exponentially decaying weights within each 16-token block (anchor token has zero weight)
- Sparse KL distillation: KL divergence computed only on the teacher's top-64 tokens (memory efficient for vocab_size=262144)

**Training hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Epochs | 6 |
| Learning rate | 6e-4 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| LR schedule | Cosine with 5% warmup |
| Effective batch size | 16 (batch=2 × grad_accum=8) |
| Grad clip | 1.0 |
| KL weight | 0.5 |
| Block size | 16 |
| Sequence length | 2048 |

**Training results:**
| Epoch | Avg Loss | CE Loss | KL Loss |
|-------|----------|---------|---------|
| 1 | — | — | — |
| 6 | 0.9364 | 1.7740 | 0.0989 |
| **Best** | **0.9364** | — | — |

### Model modifications (`modeling_sarvam_moe_dflash.py`)

`SarvamMoEForCausalLMWithKVInjection` subclasses `SarvamMoEForCausalLM` and adds three methods:
- `get_kv_injection_features()` — extract 5-layer hidden state concatenation
- `get_teacher_logits()` — get top-k logits only (memory efficient)
- `get_injection_and_logits()` — **single forward pass** returning both (used during data generation)

The Sarvam-30B model code lives in `sarvam-30b/` and is imported via a custom dynamic loader that patches relative imports, since the HuggingFace model code uses relative imports that break when loaded from outside the package.

The 5 injection layers are selected as uniformly-spaced indices into the 19 decoder layers:
```python
KV_INJECTION_LAYER_INDICES = [0, 4, 9, 13, 18]
# → hidden_states indices [1, 5, 10, 14, 19]  (index 0 = embedding output)
```

---

## File Structure

```
sarvam-dflash/
├── dflash_data.py               # Phase 1: offline data generation + DFlashDataset
├── dflash_draft.py              # Draft model architecture (DFlashDraftModel)
├── train_dflash_sarvam.py       # Phase 2: training loop with full pause/resume
├── modeling_sarvam_moe_dflash.py # KV injection hooks into Sarvam MoE
├── run_datagen.sh               # nohup launcher for data generation (PID-tracked, resume-safe)
├── run_train.sh                 # Training launcher (Phases 1 & 2 in sequence)
├── requirements.txt             # Python dependencies
├── MLX_CONVERSION.md            # Guide for converting to MLX for Mac inference
├── benchmark/
│   ├── run_benchmark.py         # MMLU benchmark runner (baseline vs speculative)
│   ├── mmlu_questions.py        # MMLU question loader
│   └── results/                 # Benchmark JSON + markdown reports
├── dflash_mlx/                  # MLX conversion artifacts (WIP)
│   ├── model.py                 # Draft model in MLX
│   ├── convert.py               # Weight conversion scripts
│   └── load.py                  # MLX model loader
├── sarvam-30b/                  # Sarvam-30B model weights (not in git, ~60GB)
├── dflash_training_data/        # Extracted shard files (not in git, ~4TB when complete)
└── checkpoints/                 # Trained draft model checkpoints (not in git, ~10GB)
    ├── dflash_draft_best.pt     # Best checkpoint (model weights only, 524MB)
    ├── dflash_draft_epoch_N.pt  # Per-epoch full state checkpoints (1.5GB each)
    ├── dflash_draft_final.pt    # Final checkpoint (model weights only, 524MB)
    └── resume_checkpoint.pt     # Latest full state for training resume (1.5GB)
```

---

## Scripts

### `run_datagen.sh` — Data generation launcher

Designed for unattended, restart-safe operation:
- Checks for a stale PID file and cleans up before starting
- Launches `dflash_data.py` via `nohup` so it survives terminal disconnection
- Writes PID to `dflash_datagen.pid` for process management
- Logs all output to `dflash_datagen.log`
- The underlying Python script auto-resumes by counting existing shards

```bash
./run_datagen.sh
# Monitor:
tail -f dflash_datagen.log
```

### `run_train.sh` — Full pipeline launcher

Runs both phases in sequence. If `dflash_training_data/metadata.pt` exists, skips Phase 1.

```bash
./run_train.sh
# To resume after interruption, just re-run. Auto-detects checkpoints/resume_checkpoint.pt
```

---

## Quick Start

### Prerequisites

```bash
pip install torch transformers datasets accelerate safetensors wandb tqdm
```

You need the Sarvam-30B model weights in `./sarvam-30b/`. Download from HuggingFace:
```bash
huggingface-cli download sarvamai/sarvam-m --local-dir ./sarvam-30b
```

### Phase 1: Generate training data

```bash
# Background mode (recommended — takes days on a single Mac):
./run_datagen.sh

# Or foreground:
python3 dflash_data.py \
    --target_model_path ./sarvam-30b \
    --dataset_name HuggingFaceFW/fineweb-edu-score-2 \
    --output_dir ./dflash_training_data \
    --max_samples 50000 \
    --max_seq_len 2048 \
    --batch_size 1 \
    --teacher_top_k 64
```

### Phase 2: Train the draft model

```bash
python3 train_dflash_sarvam.py \
    --target_model_path ./sarvam-30b \
    --data_dir ./dflash_training_data \
    --output_dir ./checkpoints \
    --num_draft_layers 6 \
    --ffn_intermediate 2048 \
    --block_size 16 \
    --max_seq_len 2048 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 6e-4 \
    --epochs 6 \
    --kl_weight 0.5
```

Interrupt with Ctrl+C for a graceful shutdown — it saves a full checkpoint before exiting. Re-running the same command resumes from that checkpoint.

---

## Resilience Design

Both phases are designed to survive crashes and interruptions with zero lost work:

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| Resume on restart | Counts existing shards, skips | Loads `resume_checkpoint.pt` |
| Checkpoint granularity | Per sample (1 shard) | Every 50 optimizer steps |
| Graceful SIGINT/SIGTERM | — | Saves checkpoint before exit |
| Atomic writes | — | `tmp → rename` prevents corruption |
| PID tracking | `dflash_datagen.pid` | — |
| Session persistence | `nohup` in `run_datagen.sh` | — |

---

## Run History

### Attempt 1 (failed)
`ModuleNotFoundError: No module named 'datasets'` — missing `datasets` package.

### Attempt 2 (failed)
`ValueError: ... requires accelerate` — `accelerate` not installed with `device_map="auto"`.

### Attempt 3 (current — running)
Successfully started after installing dependencies. Resumed from 5,992 existing shards on this run.

**Current status (as of 2026-04-17):**
- Process: PID 2316, running since ~12:33 PM IST on 2026-04-16
- Shards: ~29,607 / 50,000 (59.1%)
- Data written: ~2.5 TB
- Rate: ~3.0–3.1 s/sample
- ETA: ~17 hours remaining

**Training:** Completed on a prior subset of the data. All 6 epochs finished. Best loss: 0.9364.

---

## MLX Conversion

See [`MLX_CONVERSION.md`](MLX_CONVERSION.md) for a step-by-step guide to converting the trained draft model to MLX format for efficient inference on Apple Silicon.

Memory footprint of the draft model in MLX:

| Component | FP16 | INT4 |
|-----------|------|------|
| Draft trainable (275M) | 0.55 GB | 0.14 GB |
| Shared embedding (1.07B) | 2.14 GB | 0.54 GB |
| Shared LM head (1.07B) | 2.14 GB | 0.54 GB |
| **Draft total** | **4.83 GB** | **1.22 GB** |

Full speculative decoding requires Sarvam-30B (INT4: ~16 GB) + the draft model — fits on any M-series Mac with 32+ GB.
