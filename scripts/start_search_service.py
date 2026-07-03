"""
Start the search service service/daemon.

Key functions: atomic_write, pid_alive, proc_matches_service, port_in_use
"""
from __future__ import annotations

import psutil
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

WORKSPACE = Path.home() / ".openclaw" / "workspace"
RUNTIME = WORKSPACE / ".memory-index" / "runtime"
LOG_FILE = WORKSPACE / ".memory-index" / "search_service.log"
PID_FILE = RUNTIME / "search_service.pid"
META_FILE = RUNTIME / "search_service.json"
SCRIPT_DIR = WORKSPACE / ".memory-index" / "scripts"
PY = Path.home() / ".openclaw" / "venvs" / "memory-db" / "bin" / "python"
PORT = int(os.environ.get("OPENCLAW_SEARCH_SERVICE_PORT", "8791"))
START_TIMEOUT_SECONDS = int(os.environ.get("OPENCLAW_SEARCH_START_TIMEOUT", "30"))


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def proc_matches_service(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8", errors="ignore")
        return "uvicorn" in cmdline and "search_memory_service:app" in cmdline
    except Exception:
        return False


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def health_ok(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def find_listening_service_pid(port: int) -> int | None:
    """Return the PID of a uvicorn search_memory_service process that is
    actually LISTENING on `port`, or None. Used to adopt / re-sync a service
    that is running but whose PID file is missing or stale (e.g. started
    manually or out-of-band). This is what keeps PID tracking aligned with
    reality instead of trusting a recorded PID that may be dead or recycled.
    """
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status != psutil.CONN_LISTEN:
                continue
            laddr = conn.laddr
            if not laddr or laddr.port != port:
                continue
            pid = conn.pid
            if pid and proc_matches_service(pid):
                return pid
    except Exception:
        # net_connections can raise AccessDenied on some platforms; fall back
        # to leaving adoption to the health probe path.
        return None
    return None


def write_runtime_files(pid: int, env: dict) -> None:
    """Persist the authoritative PID + meta for the running service. Centralized
    so every code path that confirms a healthy service writes consistent state.
    """
    atomic_write(PID_FILE, str(pid))
    try:
        started_at = psutil.Process(pid).create_time()
    except Exception:
        started_at = time.time()
    atomic_write(
        META_FILE,
        json.dumps(
            {
                "pid": pid,
                "started_at": started_at,
                "port": PORT,
                "embed_model": env.get("OPENCLAW_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                "rerank_model": env.get("OPENCLAW_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                "python": sys.version,
            },
            indent=2,
        ),
    )


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            # A recorded PID is only "already running" if it is alive, is
            # genuinely our service, AND the service is actually reachable on
            # its port. The previous check trusted pid_alive+cmdline alone,
            # which reported a hung/non-listening uvicorn (the original outage
            # mode) as healthy and never recovered it.
            if pid_alive(pid) and proc_matches_service(pid) and health_ok(PORT):
                # Re-sync meta every time we confirm health. This repairs stale
                # metadata after PID reuse/restart/adoption and keeps
                # search_service_status.py from reporting STALE for a healthy
                # live daemon.
                write_runtime_files(pid, os.environ.copy())
                print(json.dumps({"status": "already_running", "pid": pid, "port": PORT}))
                return
        except Exception:
            pass
        # PID file is stale, mismatched, or points at a non-listening process.
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)

    if port_in_use(PORT):
        if health_ok(PORT):
            # A healthy service is already serving this port but our PID file
            # is missing/stale (e.g. started out-of-band). Adopt it: re-sync
            # the PID/meta to the real listening process so status/stop work.
            adopted = find_listening_service_pid(PORT)
            if adopted is not None:
                write_runtime_files(adopted, os.environ.copy())
                print(json.dumps({"status": "adopted_existing_service", "pid": adopted, "port": PORT}))
            else:
                print(json.dumps({"status": "port_in_use_existing_healthy_service", "port": PORT}))
            return
        # Port held by something that is not answering /health. Do not spawn a
        # second instance onto the same port; surface for operator action.
        print(json.dumps({"status": "port_in_use_unhealthy", "port": PORT}))
        return

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== starting search service port={PORT} at {time.time()} ===\n")
        log.flush()
        proc = subprocess.Popen(
            [
                str(PY),
                "-m",
                "uvicorn",
                "search_memory_service:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ],
            cwd=str(SCRIPT_DIR),
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=env,
        )

    deadline = time.time() + START_TIMEOUT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            print(json.dumps({"status": "failed_to_start", "returncode": proc.returncode, "log_file": str(LOG_FILE)}))
            return
        if health_ok(PORT):
            write_runtime_files(proc.pid, env)
            print(json.dumps({"status": "started", "pid": proc.pid, "port": PORT, "log_file": str(LOG_FILE)}))
            return
        time.sleep(1.0)

    try:
        proc.terminate()
    except Exception:
        pass

    print(json.dumps({"status": "startup_timeout", "port": PORT, "log_file": str(LOG_FILE)}))


if __name__ == "__main__":
    main()
