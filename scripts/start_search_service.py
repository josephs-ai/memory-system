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


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            if pid_alive(pid) and proc_matches_service(pid):
                print(json.dumps({"status": "already_running", "pid": pid, "port": PORT}))
                return
        except Exception:
            pass
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)

    if port_in_use(PORT):
        if health_ok(PORT):
            print(json.dumps({"status": "port_in_use_existing_healthy_service", "port": PORT}))
            return
        print(json.dumps({"status": "port_in_use_unhealthy", "port": PORT}))
        return

    env = os.environ.copy()

    with open(LOG_FILE, "a", encoding="utf-8") as log:
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
            atomic_write(PID_FILE, str(proc.pid))
            atomic_write(
                META_FILE,
                json.dumps(
                    {
                        "pid": proc.pid,
                        "started_at": psutil.Process(proc.pid).create_time(),
                        "port": PORT,
                        "embed_model": env.get("OPENCLAW_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                        "rerank_model": env.get("OPENCLAW_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                        "python": sys.version,
                    },
                    indent=2,
                ),
            )
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
