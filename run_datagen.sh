#!/bin/bash
# DFlash data generation launcher - designed for unattended/restart-safe operation
# Logs to dflash_datagen.log, supports resume via existing shards

PROJ_DIR="/Users/arjun/Projects/sarvam-dflash"
LOG_FILE="$PROJ_DIR/dflash_datagen.log"
PID_FILE="$PROJ_DIR/dflash_datagen.pid"

cd "$PROJ_DIR" || exit 1

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Data generation already running (PID $OLD_PID). Exiting."
        exit 0
    else
        echo "Stale PID file found. Cleaning up."
        rm -f "$PID_FILE"
    fi
fi

echo "$(date): Starting data generation..." >> "$LOG_FILE"

nohup python3 dflash_data.py \
    --target_model_path ./sarvam-30b \
    --dataset_name HuggingFaceFW/fineweb-edu-score-2 \
    --output_dir ./dflash_training_data \
    --max_samples 50000 \
    --max_seq_len 2048 \
    --batch_size 1 \
    --teacher_top_k 64 \
    >> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "$(date): Data generation started with PID $!" >> "$LOG_FILE"
echo "Data generation started (PID $!). Logs: $LOG_FILE"
