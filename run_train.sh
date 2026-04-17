#!/bin/bash
set -euo pipefail

# =============================================================================
# DFlash Draft Model Training Pipeline for Sarvam-30B
#
# Phase 1: Generate training data (one-time, runs frozen 32B model)
# Phase 2: Train the ~275M draft model
#
# Supports pause/resume: Ctrl+C saves checkpoint, re-running resumes.
# =============================================================================

TARGET_MODEL="./sarvam-30b"
DATA_DIR="./dflash_training_data"
CHECKPOINT_DIR="./checkpoints"
DATASET="HuggingFaceFW/fineweb-edu-score-2"

# Adjust based on your hardware
DATA_GEN_BATCH_SIZE=1    # Batch size for 32B model forward (1 for 128GB Mac)
TRAIN_BATCH_SIZE=2       # Batch size for draft training
GRAD_ACCUM=8             # Effective batch = TRAIN_BATCH_SIZE * GRAD_ACCUM = 16
SAVE_INTERVAL=50         # Save resume checkpoint every N optimizer steps

echo "=============================================="
echo "DFlash Training Pipeline for Sarvam-30B"
echo "=============================================="

# -----------------------------------------------------------------------------
# Phase 1: Offline Data Generation
# -----------------------------------------------------------------------------
if [ ! -f "${DATA_DIR}/metadata.pt" ]; then
    echo ""
    echo "[Phase 1] Generating training data from frozen Sarvam-30B..."
    echo "  Dataset: ${DATASET}"
    echo "  Output:  ${DATA_DIR}"
    echo ""

    python3 dflash_data.py \
        --target_model_path "${TARGET_MODEL}" \
        --dataset_name "${DATASET}" \
        --max_samples 50000 \
        --max_seq_len 2048 \
        --output_dir "${DATA_DIR}" \
        --batch_size "${DATA_GEN_BATCH_SIZE}" \
        --teacher_top_k 64

    echo "[Phase 1] Data generation complete."
else
    echo "[Phase 1] Training data already exists at ${DATA_DIR}, skipping."
fi

# -----------------------------------------------------------------------------
# Phase 2: Train Draft Model
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 2] Training DFlash draft model..."
echo "  Layers: 6, FFN: 2048, Block size: 16"
echo "  LR: 6e-4, Epochs: 6, Effective batch: $((TRAIN_BATCH_SIZE * GRAD_ACCUM))"
echo "  Save interval: every ${SAVE_INTERVAL} optimizer steps"
echo ""

python3 train_dflash_sarvam.py \
    --target_model_path "${TARGET_MODEL}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${CHECKPOINT_DIR}" \
    --num_draft_layers 6 \
    --ffn_intermediate 2048 \
    --block_size 16 \
    --max_seq_len 2048 \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr 6e-4 \
    --epochs 6 \
    --warmup_ratio 0.05 \
    --kl_weight 0.5 \
    --save_interval "${SAVE_INTERVAL}" \
    --log_interval 10

echo ""
echo "[Phase 2] Training complete. Checkpoints saved to ${CHECKPOINT_DIR}"
echo ""
echo "Best checkpoint: ${CHECKPOINT_DIR}/dflash_draft_best.pt"
echo "Final checkpoint: ${CHECKPOINT_DIR}/dflash_draft_final.pt"
echo ""
echo "To resume after interruption, just re-run this script."
echo "It auto-detects ${CHECKPOINT_DIR}/resume_checkpoint.pt and continues."
