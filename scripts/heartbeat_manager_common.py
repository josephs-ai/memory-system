"""
Heartbeat subsystem: manager common management.

Key functions: ensure_runtime_dir, load_env_file, build_runtime_env, atomic_write_text
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_INDEX_DIR = WORKSPACE / ".memory-index"
RUNTIME_DIR = MEMORY_INDEX_DIR / "runtime"
ENV_FILE = MEMORY_INDEX_DIR / ".env"

PID_FILE = RUNTIME_DIR / "heartbeat_worker.pid"
META_FILE = RUNTIME_DIR / "heartbeat_worker.json"
LOG_FILE = MEMORY_INDEX_DIR / "heartbeat_worker.log"
WORKER_SCRIPT = MEMORY_INDEX_DIR / "scripts" / "heartbeat_memory_worker.py"


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    file_env = load_env_file(ENV_FILE)
    for key, value in file_env.items():
        env.setdefault(key, value)

    env.setdefault("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory")
    env.setdefault("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")
    return env


def atomic_write_text(path: Path, text: str) -> None:
    ensure_runtime_dir()
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def process_group_alive(pid: int | None) -> bool:
    return bool(pid and pid_alive(pid))


def remove_runtime_files() -> None:
    PID_FILE.unlink(missing_ok=True)
    META_FILE.unlink(missing_ok=True)


def terminate_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
        return
    except Exception:
        pass

    try:
        os.kill(pid, sig)
    except Exception:
        pass


def log_file_stats() -> dict[str, Any]:
    try:
        st = LOG_FILE.stat()
        return {
            "exists": True,
            "size_bytes": st.st_size,
            "mtime": st.st_mtime,
        }
    except Exception:
        return {
            "exists": False,
            "size_bytes": 0,
            "mtime": None,
        }
