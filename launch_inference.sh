#!/bin/bash
# launch_inference.sh — run the V3 inference acceptance test as a background job.
#
# Writes progress to inference_test_state.json — the web dashboard at
# http://127.0.0.1:8765/inference reads that file and auto-refreshes every 2s.
# The test survives Claude closing, terminal closing, and laptop lid close
# (via caffeinate).

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PID_FILE="$PROJECT_DIR/dflash_inference.pid"
LOG_FILE="$PROJECT_DIR/dflash_inference.log"
STATE_FILE="$PROJECT_DIR/inference_test_state.json"

# Already running?
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Inference test already running (PID $OLD_PID)."
        echo "   View at http://127.0.0.1:8765/inference"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Sanity
if [[ ! -f ./checkpoints/dflash_draft_best.pt ]]; then
    echo "❌ ./checkpoints/dflash_draft_best.pt not found. Did training finish?"
    exit 1
fi
if [[ ! -d ./sarvam-30b ]]; then
    echo "❌ ./sarvam-30b not found."
    exit 1
fi

# Clear stale state file so the dashboard shows a fresh run
rm -f "$STATE_FILE"

# Dashboard running?
if ! curl -sf http://127.0.0.1:8765/api/ping > /dev/null 2>&1; then
    echo "⚠️  Dashboard server not running. Start it first:"
    echo "      ./launch_dashboard.sh"
    echo "   Proceeding anyway; you'll need to start the dashboard separately to watch progress."
fi

# Launch
echo "═══════════════════════════════════════════════════════════════════════"
echo "  V3 Inference Acceptance Test"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Checkpoint:  ./checkpoints/dflash_draft_best.pt"
echo "  Prompts:     3  |  max tokens: 64  |  block size: 16"
echo "  Log:         $LOG_FILE"
echo "  State JSON:  $STATE_FILE  (dashboard reads this)"
echo "  PID file:    $PID_FILE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

nohup caffeinate -i python3 inference_test.py > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "✓ Launched. PID=$NEW_PID"
echo ""
echo "Watch progress in the browser:"
echo "    http://127.0.0.1:8765/inference"
echo ""
echo "Or tail the log:"
echo "    tail -f $LOG_FILE"

# Open the inference page
open "http://127.0.0.1:8765/inference" 2>/dev/null || true
