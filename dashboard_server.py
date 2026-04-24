#!/usr/bin/env python3
"""
DFlash V3 Training Dashboard — local web server.

Runs independently of Claude. Serves dashboard.html on http://127.0.0.1:8765
and a JSON /api/status endpoint the dashboard polls every 2 seconds.

Pause  = SIGSTOP to the training PID (OS-level freeze — resumable)
Resume = SIGCONT to the training PID
Cancel = SIGTERM (the training script catches this and saves a checkpoint)

Safe to run via nohup; it's a tiny stdlib HTTP server with no external deps.
"""

import http.server
import json
import os
import re
import signal
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).parent.resolve()
PID_FILE    = PROJECT_DIR / "dflash_training_v3.pid"
LOG_FILE    = PROJECT_DIR / "dflash_training_v3.log"
HTML_FILE   = PROJECT_DIR / "dashboard.html"
CKPT_DIR    = PROJECT_DIR / "checkpoints"

PORT = int(os.environ.get("DFLASH_DASHBOARD_PORT", "8765"))

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def read_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None

def process_state(pid):
    """'running' | 'paused' | 'stopped' | 'not_started'"""
    if pid is None:
        return "not_started"
    try:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                           capture_output=True, text=True, timeout=3)
        if r.returncode != 0:
            return "stopped"
        stat = r.stdout.strip()
        if not stat:
            return "stopped"
        # macOS / BSD ps stat flags: T = stopped (paused), R/S = running/sleep
        if "T" in stat:
            return "paused"
        return "running"
    except Exception:
        return "stopped"

def parse_elapsed(s):
    """ps etime format: DD-HH:MM:SS / HH:MM:SS / MM:SS"""
    if not s:
        return None
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        h, m, sec = parts
        return days * 86400 + h * 3600 + m * 60 + sec
    except Exception:
        return None

def process_elapsed_seconds(pid):
    if pid is None:
        return None
    try:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "etime="],
                           capture_output=True, text=True, timeout=3)
        return parse_elapsed(r.stdout.strip())
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

STEP_RE = re.compile(
    r"\[Epoch\s+(\d+)\s*/\s*(\d+)\]\s+"
    r"Step\s+(\d+)\s*/\s*(\d+).*?"
    r"Loss:\s*([\d.]+)\s*\(CE:\s*([\d.]+),\s*KL:\s*([\d.]+)\).*?"
    r"LR:\s*([0-9.eE+\-]+)"
)

def parse_log():
    if not LOG_FILE.exists():
        return {"phase": "not_started", "steps": [], "recent_lines": []}

    try:
        content = LOG_FILE.read_text(errors="ignore")
    except Exception:
        return {"phase": "not_started", "steps": [], "recent_lines": []}

    # Phase detection (latest wins)
    phase = "starting_training"
    if "Loading Sarvam-30B" in content or "Loading weights" in content:
        phase = "loading_target"
    if "Starting training:" in content:
        phase = "starting_training"
    if STEP_RE.search(content):
        phase = "training"
    if "Saving checkpoint before shutdown" in content or "Checkpoint saved: " in content:
        # still technically training unless terminated
        pass
    if re.search(r"Training complete\.", content):
        phase = "completed"
    if re.search(r"Traceback|\bError\b", content):
        # Don't change phase — let user see via recent lines
        pass

    steps = []
    # Strip ANSI tqdm noise; keep every [Epoch ...] line
    for m in STEP_RE.finditer(content):
        steps.append({
            "epoch": int(m.group(1)),
            "total_epochs": int(m.group(2)),
            "step": int(m.group(3)),
            "total_steps": int(m.group(4)),
            "loss": float(m.group(5)),
            "ce":   float(m.group(6)),
            "kl":   float(m.group(7)),
            "lr":   float(m.group(8)),
        })

    # Recent lines — strip the long tqdm progress lines
    lines = [ln for ln in content.rstrip().split("\n")
             if "Loading weights" not in ln]
    recent = lines[-12:]

    return {"phase": phase, "steps": steps, "recent_lines": recent}

def list_checkpoints():
    if not CKPT_DIR.exists():
        return []
    items = []
    for p in CKPT_DIR.glob("*.pt"):
        try:
            st = p.stat()
            items.append({
                "name": p.name,
                "size_mb": st.st_size / (1024 * 1024),
                "modified": st.st_mtime,
            })
        except Exception:
            continue
    items.sort(key=lambda x: x["modified"], reverse=True)
    return items

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs):
        pass  # silence default per-request logging

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                body = HTML_FILE.read_bytes()
            except Exception as e:
                self._send_json({"error": f"dashboard.html missing: {e}"}, 500)
                return
            self._send_html(body)
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/ping":
            self._send_json({"ok": True})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        pid = read_pid()
        if pid is None or process_state(pid) in ("stopped", "not_started"):
            self._send_json({"error": "no live training process"}, 400)
            return
        try:
            if path == "/api/pause":
                os.kill(pid, signal.SIGSTOP)
                self._send_json({"ok": True, "action": "paused"})
            elif path == "/api/resume":
                os.kill(pid, signal.SIGCONT)
                self._send_json({"ok": True, "action": "resumed"})
            elif path == "/api/cancel":
                os.kill(pid, signal.SIGTERM)
                self._send_json({"ok": True, "action": "cancel sent (saving checkpoint…)"})
            else:
                self.send_error(404)
        except ProcessLookupError:
            self._send_json({"error": "process vanished"}, 400)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _api_status(self):
        pid = read_pid()
        state = process_state(pid)
        log = parse_log()
        ckpts = list_checkpoints()

        rate = None
        eta = None
        if pid and state in ("running", "paused") and log["steps"]:
            elapsed = process_elapsed_seconds(pid)
            latest = log["steps"][-1]
            if elapsed and latest["step"] > 0:
                rate = elapsed / latest["step"]
                remaining = latest["total_steps"] - latest["step"]
                eta = int(remaining * rate)

        self._send_json({
            "pid": pid,
            "state": state,
            "phase": log["phase"],
            "steps": log["steps"][-200:],  # last 200 for chart
            "latest": log["steps"][-1] if log["steps"] else None,
            "recent_lines": log["recent_lines"],
            "checkpoints": ckpts,
            "rate_sec_per_step": rate,
            "eta_seconds": eta,
        })

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def main():
    srv = ThreadedServer(("127.0.0.1", PORT), DashboardHandler)
    print(f"[dashboard] serving http://127.0.0.1:{PORT}")
    print(f"[dashboard] project dir: {PROJECT_DIR}")
    print(f"[dashboard] watching PID file: {PID_FILE}")
    print(f"[dashboard] watching log file: {LOG_FILE}")
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()

if __name__ == "__main__":
    main()
