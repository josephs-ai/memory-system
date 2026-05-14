"""
Search and retrieve from service status.

Key functions: read_json, pid_alive, proc_matches_service, proc_start_time
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import psutil

WORKSPACE = Path.home() / ".openclaw" / "workspace"
RUNTIME = WORKSPACE / ".memory-index" / "runtime"
LOG_FILE = WORKSPACE / ".memory-index" / "search_service.log"
PID_FILE = RUNTIME / "search_service.pid"
META_FILE = RUNTIME / "search_service.json"
PORT = int(os.environ.get("OPENCLAW_SEARCH_SERVICE_PORT", "8791"))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pid_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def proc_matches_service(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline())
        return "uvicorn" in cmdline and "search_memory_service:app" in cmdline
    except Exception:
        return False


def proc_start_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def proc_stats(pid: int) -> dict:
    try:
        proc = psutil.Process(pid)
        return {
            "name": proc.name(),
            "status": proc.status(),
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "memory_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
            "create_time": proc.create_time(),
        }
    except Exception:
        return {}


def tail_log(path: Path, n: int = 10) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-n:]
    except Exception:
        return []


def classify_health(port: int) -> dict:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3)
        body = r.json()
        return {
            "ok": r.status_code == 200 and bool(body.get("ok")),
            "kind": "ok" if r.status_code == 200 and bool(body.get("ok")) else "http_error",
            "status_code": r.status_code,
            "body": body,
        }
    except httpx.ConnectTimeout as e:
        return {"ok": False, "kind": "timeout", "error": repr(e)}
    except httpx.ConnectError as e:
        return {"ok": False, "kind": "connect_error", "error": repr(e)}
    except Exception as e:
        return {"ok": False, "kind": "other_error", "error": repr(e)}


def summarize_status(
    *,
    pid_exists: bool,
    running: bool,
    proc_ok: bool,
    health_ok: bool,
    stale_pid: bool,
) -> str:
    if stale_pid:
        return "STALE"
    if not pid_exists and not running:
        return "OFFLINE"
    if running and proc_ok and health_ok:
        return "HEALTHY"
    if running:
        return "DEGRADED"
    return "OFFLINE"


def main() -> None:
    pid_exists = PID_FILE.exists()
    pid = None
    if pid_exists:
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except Exception:
            pid = None

    meta = read_json(META_FILE)

    meta_port = meta.get("port")
    port_warning = None
    if meta_port is not None and int(meta_port) != PORT:
        port_warning = {
            "env_port": PORT,
            "meta_port": meta_port,
            "warning": "checking_different_port_than_service_started_with",
        }

    running = bool(pid and pid_alive(pid))
    proc_ok = bool(pid and proc_matches_service(pid))
    health = classify_health(PORT)

    meta_started_at = meta.get("started_at")
    live_started_at = proc_start_time(pid) if pid else None

    start_time_warning = None
    stale_pid = False
    if pid and running and meta_started_at and live_started_at:
        # PID recycle protection: if process creation time differs meaningfully,
        # treat the pid file/meta as stale.
        if abs(float(meta_started_at) - float(live_started_at)) > 5:
            stale_pid = True
            start_time_warning = {
                "meta_started_at": meta_started_at,
                "live_create_time": live_started_at,
                "warning": "pid_reuse_or_stale_meta_detected",
            }

    status = summarize_status(
        pid_exists=pid_exists,
        running=running,
        proc_ok=proc_ok,
        health_ok=bool(health.get("ok")),
        stale_pid=stale_pid,
    )

    out = {
        "status": status,
        "running": running,
        "pid": pid,
        "pid_file_exists": pid_exists,
        "pid_matches_service": proc_ok,
        "stale_pid_detected": stale_pid,
        "port": PORT,
        "port_warning": port_warning,
        "start_time_warning": start_time_warning,
        "meta": meta,
        "process": proc_stats(pid) if pid else {},
        "health": health,
        "log_file": str(LOG_FILE),
    }

    if status == "DEGRADED" or status == "STALE":
        out["log_tail"] = tail_log(LOG_FILE, 10)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
