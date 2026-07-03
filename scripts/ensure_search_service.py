"""
Ensure the memory search FastAPI service is running.

The search service is intentionally started out-of-process so retrieval can keep
sentence-transformer / reranker models warm. Historically it was fire-and-forget:
if the uvicorn process died, runtime/search_service.pid and .json stayed stale
and nothing restarted it. This watchdog mirrors the tracked-path watcher ensure
pattern and is safe to call periodically from heartbeat/systemd/cron.

Key functions: run_json, status_healthy, main
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from heartbeat_manager_common import MEMORY_INDEX_DIR

STARTER = MEMORY_INDEX_DIR / "scripts" / "start_search_service.py"
STATUS = MEMORY_INDEX_DIR / "scripts" / "search_service_status.py"


def run_json(cmd: list[str], *, timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        return {"status": "exec_not_found", "error": str(e), "cmd": cmd}
    except subprocess.TimeoutExpired as e:
        return {"status": "timeout", "error": str(e), "cmd": cmd}
    except OSError as e:
        return {"status": "exec_os_error", "error": str(e), "cmd": cmd}
    except Exception as e:
        return {"status": "exec_exception", "error": str(e), "cmd": cmd}

    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "cmd": cmd,
        }

    try:
        return json.loads(proc.stdout)
    except Exception:
        return {
            "status": "invalid_json",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "cmd": cmd,
        }


def status_healthy(status: dict) -> bool:
    return (
        isinstance(status, dict)
        and status.get("status") == "HEALTHY"
        and bool(status.get("running"))
        and bool(status.get("pid_matches_service"))
        and bool((status.get("health") or {}).get("ok"))
    )


def start_result_ok(result: dict) -> bool:
    return isinstance(result, dict) and result.get("status") in {
        "already_running",
        "started",
        "adopted_existing_service",
        "port_in_use_existing_healthy_service",
    }


def main() -> None:
    status = run_json([sys.executable, str(STATUS)], timeout=20)

    if status_healthy(status):
        print(json.dumps({
            "status": "already_healthy",
            "pid": status.get("pid"),
            "service_status": status,
        }, ensure_ascii=False))
        return

    result = run_json([sys.executable, str(STARTER)], timeout=90)
    after = run_json([sys.executable, str(STATUS)], timeout=20)

    out = {
        "status": "start_attempted",
        "ok": start_result_ok(result) and status_healthy(after),
        "prior_status": status,
        "start_result": result,
        "after_status": after,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if not out["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
