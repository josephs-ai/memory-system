import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
LOCKS_DIR = WORKSPACE / ".memory-index" / "locks"
LOCKS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def lock_path(project_id: str) -> Path:
    return LOCKS_DIR / f"{project_id}.lock"


def acquire_project_lock(project_id: str, owner: str, ttl_seconds: int = 1800) -> tuple[bool, str]:
    path = lock_path(project_id)

    # stale lock cleanup
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            created_ts = float(data.get("created_ts", 0))
        except Exception:
            created_ts = 0

        now_ts = time.time()
        if created_ts and (now_ts - created_ts) > ttl_seconds:
            try:
                path.unlink()
            except Exception:
                pass

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return False, data.get("owner", "unknown")
        except Exception:
            return False, "unknown"

    payload = {
        "project_id": project_id,
        "owner": owner,
        "created_at": now_iso(),
        "created_ts": time.time(),
        "pid": os.getpid(),
    }

    # atomic enough for this level: create only if absent
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return True, owner


def release_project_lock(project_id: str, owner: str):
    path = lock_path(project_id)
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("owner") != owner:
            return
    except Exception:
        return

    try:
        path.unlink()
    except Exception:
        pass
