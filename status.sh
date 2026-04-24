#!/bin/bash
# status.sh — live status dashboard for DFlash V3 training.
#
# Usage:
#   ./status.sh          # runs forever, refreshes every 30s, Ctrl-C to exit
#   ./status.sh once     # prints once and exits (useful for `watch` or one-shot checks)
#
# Safe to run in ANY terminal, anywhere. Reads from files written by the
# training process. The training process itself does NOT depend on this
# dashboard — you can close this, close Claude, close your laptop lid, and
# the training keeps going as long as the nohup'd PID is alive.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

PID_FILE="$PROJECT_DIR/dflash_training_v3.pid"
LOG_FILE="$PROJECT_DIR/dflash_training_v3.log"
CKPT_DIR="$PROJECT_DIR/checkpoints"

render() {
    clear
    printf "\033[1;36m"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║           DFlash V3 Training Status — Sarvam-30B (M3 Max)          ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    printf "\033[0m"
    echo ""
    printf "Time now: \033[1m%s\033[0m\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo ""

    # --- Process status ---
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            ELAPSED=$(ps -p "$PID" -o etime= | tr -d ' ')
            printf "Process: \033[1;32m● RUNNING\033[0m  PID=%s  elapsed=%s\n" "$PID" "$ELAPSED"
        else
            printf "Process: \033[1;31m● STOPPED\033[0m  (PID %s no longer alive — check log for reason)\n" "$PID"
        fi
    else
        printf "Process: \033[1;33m● NOT STARTED\033[0m  (no PID file at %s)\n" "$PID_FILE"
    fi
    echo ""

    # --- Progress ---
    if [[ -f "$LOG_FILE" ]]; then
        # Latest step line
        LATEST=$(grep -E "^\[Epoch" "$LOG_FILE" | tail -1)
        if [[ -n "$LATEST" ]]; then
            # Parse out numbers
            STEP=$(echo "$LATEST" | sed -n 's/.*Step \([0-9]*\)\/.*/\1/p')
            TOTAL=$(echo "$LATEST" | sed -n 's/.*Step [0-9]*\/\([0-9]*\).*/\1/p')
            EPOCH=$(echo "$LATEST" | sed -n 's/.*Epoch \([0-9]*\)\/\([0-9]*\).*/\1\/\2/p')
            LOSS=$(echo "$LATEST" | sed -n 's/.*Loss: \([0-9.]*\).*/\1/p')
            CE=$(echo "$LATEST" | sed -n 's/.*CE: \([0-9.]*\).*/\1/p')
            KL=$(echo "$LATEST" | sed -n 's/.*KL: \([0-9.]*\).*/\1/p')
            LR=$(echo "$LATEST" | sed -n 's/.*LR: \([^ |]*\).*/\1/p')

            printf "Epoch:   \033[1m%s\033[0m\n" "$EPOCH"
            printf "Step:    \033[1m%s / %s\033[0m" "$STEP" "$TOTAL"
            if [[ -n "$STEP" && -n "$TOTAL" && "$TOTAL" != "0" ]]; then
                PCT=$(awk -v s="$STEP" -v t="$TOTAL" 'BEGIN { printf "%.1f", s/t*100 }')
                printf "  (\033[1m%s%%\033[0m)" "$PCT"
                # Render a 40-char progress bar
                BAR_LEN=40
                FILLED=$(awk -v s="$STEP" -v t="$TOTAL" -v b="$BAR_LEN" 'BEGIN { printf "%d", s/t*b }')
                EMPTY=$((BAR_LEN - FILLED))
                printf "  ["
                printf "█%.0s" $(seq 1 "$FILLED" 2>/dev/null)
                printf "░%.0s" $(seq 1 "$EMPTY" 2>/dev/null)
                printf "]"
            fi
            echo ""
            printf "Loss:    \033[1m%s\033[0m   (CE=%s, KL=%s)\n" "$LOSS" "$CE" "$KL"
            printf "LR:      %s\n" "$LR"

            # ETA: based on step throughput since start of log
            FIRST_STEP_LINE=$(grep -E "^\[Epoch" "$LOG_FILE" | head -1)
            if [[ -n "$FIRST_STEP_LINE" && -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" > /dev/null 2>&1; then
                # Rough ETA: use elapsed time / step count
                PID=$(cat "$PID_FILE")
                START_TIME=$(ps -p "$PID" -o lstart= 2>/dev/null)
                if [[ -n "$START_TIME" && -n "$STEP" && -n "$TOTAL" ]]; then
                    START_EPOCH=$(date -j -f "%a %b %e %H:%M:%S %Y" "$START_TIME" "+%s" 2>/dev/null)
                    NOW_EPOCH=$(date "+%s")
                    if [[ -n "$START_EPOCH" ]]; then
                        ELAPSED_S=$((NOW_EPOCH - START_EPOCH))
                        if [[ "$STEP" -gt 0 ]]; then
                            S_PER_STEP=$(awk -v e="$ELAPSED_S" -v s="$STEP" 'BEGIN { printf "%.2f", e/s }')
                            REMAINING=$((TOTAL - STEP))
                            ETA_S=$(awk -v r="$REMAINING" -v sps="$S_PER_STEP" 'BEGIN { printf "%d", r*sps }')
                            ETA_H=$(awk -v s="$ETA_S" 'BEGIN { printf "%.1f", s/3600 }')
                            ETA_D=$(awk -v s="$ETA_S" 'BEGIN { printf "%.2f", s/86400 }')
                            FINISH=$(date -v "+${ETA_S}S" "+%Y-%m-%d %H:%M" 2>/dev/null)
                            printf "Rate:    %s sec/step\n" "$S_PER_STEP"
                            printf "ETA:     %s hours (%s days) → finishes around \033[1m%s\033[0m\n" "$ETA_H" "$ETA_D" "$FINISH"
                        fi
                    fi
                fi
            fi
        else
            echo "(no training step lines in log yet — may still be loading Sarvam-30B)"
        fi

        # --- Recent log tail ---
        echo ""
        printf "\033[1mLast 6 log lines:\033[0m\n"
        tail -6 "$LOG_FILE" | sed 's/^/  /'
    else
        printf "\033[1;33m(no log file yet at %s)\033[0m\n" "$LOG_FILE"
    fi

    echo ""

    # --- Checkpoints ---
    if [[ -d "$CKPT_DIR" ]]; then
        CKPT_COUNT=$(find "$CKPT_DIR" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
        if [[ -n "$CKPT_COUNT" && "$CKPT_COUNT" -gt 0 ]]; then
            LATEST_CKPT=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null | head -1)
            LATEST_NAME=$(basename "$LATEST_CKPT" 2>/dev/null)
            LATEST_SIZE=$(ls -lh "$LATEST_CKPT" 2>/dev/null | awk '{print $5}')
            LATEST_TIME=$(ls -l "$LATEST_CKPT" 2>/dev/null | awk '{print $6, $7, $8}')
            printf "Checkpoints: \033[1m%s files\033[0m  |  latest: %s (%s, %s)\n" \
                   "$CKPT_COUNT" "$LATEST_NAME" "$LATEST_SIZE" "$LATEST_TIME"
        fi
    fi

    echo ""
    printf "\033[90mTraining is backgrounded via nohup. This dashboard reads files;\n"
    printf "closing it or closing Claude does NOT affect training.\033[0m\n"
    if [[ "$1" != "once" ]]; then
        printf "\033[90m(refreshing every 30s — Ctrl-C to exit)\033[0m\n"
    fi
}

if [[ "$1" == "once" ]]; then
    render once
    exit 0
fi

while true; do
    render
    sleep 30
done
