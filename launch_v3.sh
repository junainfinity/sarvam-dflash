#!/bin/bash
# launch_v3.sh — launch the V3 DFlash training in the background.
#
# This uses `caffeinate -i` to prevent the Mac from sleeping during the 4-day
# training, wraps `nohup` so the process survives terminal close, writes a
# PID file, and redirects all output to dflash_training_v3.log.
#
# After running this script:
#   - quit your terminal / close the Claude Desktop app → training keeps going
#   - check progress any time from any terminal: ./status.sh
#   - kill gracefully:  kill $(cat dflash_training_v3.pid)   (will save checkpoint)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PID_FILE="$PROJECT_DIR/dflash_training_v3.pid"
LOG_FILE="$PROJECT_DIR/dflash_training_v3.log"

# --- Check for running training ---
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Training already running (PID $OLD_PID). Refusing to start another."
        echo "   To stop it first:  kill $OLD_PID"
        exit 1
    else
        echo "Stale PID file found (PID $OLD_PID no longer alive). Removing."
        rm -f "$PID_FILE"
    fi
fi

# --- Environment sanity ---
if [[ ! -d ./sarvam-30b ]]; then
    echo "❌ ./sarvam-30b not found. Cannot train without the target model."
    exit 1
fi
if [[ ! -d ./dflash_training_data ]]; then
    echo "❌ ./dflash_training_data not found. Cannot train without shards."
    exit 1
fi

SHARD_COUNT=$(find ./dflash_training_data -name 'shard_*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$SHARD_COUNT" -lt 1000 ]]; then
    echo "❌ Only $SHARD_COUNT shards found in dflash_training_data/. Expected 50,000."
    exit 1
fi
echo "✓ Found $SHARD_COUNT training shards"

# Remove any V2 resume_checkpoint so V3 starts fresh
if [[ -f "$PROJECT_DIR/checkpoints/resume_checkpoint.pt" ]]; then
    echo "Removing stale V2 resume_checkpoint.pt so V3 starts from scratch..."
    rm -f "$PROJECT_DIR/checkpoints/resume_checkpoint.pt"
fi

# --- Print the plan ---
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  DFlash V3 Training Launch"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Target model:  ./sarvam-30b"
echo "  Data:          ./dflash_training_data ($SHARD_COUNT shards)"
echo "  Output:        ./checkpoints"
echo "  Config:        batch=2, grad_accum=8, lr=6e-4, warmup 5%, 4 epochs"
echo "  Log file:      $LOG_FILE"
echo "  PID file:      $PID_FILE"
echo "  Sleep:         prevented via 'caffeinate -i' for duration of run"
echo ""
echo "  After launch, run ./status.sh to monitor. Training survives terminal"
echo "  close, ssh disconnect, and Claude Desktop quitting."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# --- Launch (nohup + caffeinate) ---
nohup caffeinate -i python3 train_dflash_sarvam.py \
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
    --epochs 4 \
    --warmup_ratio 0.05 \
    --kl_weight 0.5 \
    --save_interval 50 \
    --log_interval 10 \
    --num_workers 4 \
    > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "✓ Launched. PID=$NEW_PID"
echo ""
echo "Give it ~2 minutes to load Sarvam-30B, then run:"
echo "    ./status.sh"
echo ""
echo "To stop gracefully (will save checkpoint):"
echo "    kill \$(cat $PID_FILE)"
