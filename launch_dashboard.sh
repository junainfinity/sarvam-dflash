#!/bin/bash
# launch_dashboard.sh — start the DFlash training dashboard as a background service.
#
# Opens http://127.0.0.1:8765 in your default browser. Dashboard keeps running
# even if you close Claude Desktop, close the browser, or close your terminal.
# To stop the dashboard:    kill $(cat dflash_dashboard.pid)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PID_FILE="$PROJECT_DIR/dflash_dashboard.pid"
LOG_FILE="$PROJECT_DIR/dflash_dashboard.log"
PORT="${DFLASH_DASHBOARD_PORT:-8765}"

# Already running?
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "✓ Dashboard already running (PID $OLD_PID)."
        echo "  Open: http://127.0.0.1:$PORT"
        open "http://127.0.0.1:$PORT" 2>/dev/null || true
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Port clash check
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    echo "⚠️  Port $PORT already in use by another process."
    echo "   Set DFLASH_DASHBOARD_PORT to a different port and retry."
    exit 1
fi

# Launch
nohup python3 "$PROJECT_DIR/dashboard_server.py" > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

# Wait briefly for the server to be ready
for i in {1..20}; do
    if curl -sf "http://127.0.0.1:$PORT/api/ping" > /dev/null 2>&1; then
        break
    fi
    sleep 0.25
done

if curl -sf "http://127.0.0.1:$PORT/api/ping" > /dev/null 2>&1; then
    echo "✓ Dashboard running at http://127.0.0.1:$PORT  (PID $NEW_PID)"
    echo ""
    echo "  Open the URL in any browser. Close this terminal, close Claude —"
    echo "  the dashboard and the training both keep running."
    echo ""
    echo "  To stop the dashboard:  kill $NEW_PID"
    open "http://127.0.0.1:$PORT" 2>/dev/null || true
else
    echo "⚠️  Dashboard did not respond within 5s. Check $LOG_FILE for errors."
    exit 1
fi
