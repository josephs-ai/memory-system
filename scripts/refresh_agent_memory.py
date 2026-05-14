"""
Memory system utility: refresh agent memory.

Key functions: now_iso, load_state, file_mtime, tracked_event_files
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from project_memory_paths import (
    get_project_current_file,
    get_latest_project_daily_file,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_refresh_config import (
    get_agent_memory_paths,
    get_agent_refresh_interval,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"
STATE_FILE = LOGS_DIR / "agent-memory-refresh-state.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def tracked_event_files(project_id: str | None):
    files = []
    if not project_id:
        return files

    from project_memory_paths import (
        get_project_current_file,
        get_latest_project_daily_file,
        get_project_summary_file,
    )

    current_file = get_project_current_file(project_id)
    latest_daily = get_latest_project_daily_file(project_id)
    summary_file = get_project_summary_file(project_id)

    files.append(current_file)
    if latest_daily is not None:
        files.append(latest_daily)
    files.append(summary_file)

    return files


def build_event_state(project_id: str | None):
    state = {}
    for path in tracked_event_files(project_id):
        state[str(path)] = file_mtime(path)
    return state


def event_triggered(old_state: dict, new_state: dict) -> bool:
    for path, new_mtime in new_state.items():
        old_mtime = float(old_state.get(path, 0))
        if new_mtime > old_mtime:
            return True
    return False

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def latest_daily_file():
    daily_dir = WORKSPACE / ".memory-index" / "timeline" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(daily_dir.glob("*.md"))
    if not files:
        return None
    return files[-1]


def resolve_memory_path_token(token):
    if token == "__LATEST_DAILY__":
        return latest_daily_file()
    return Path(token)

def read_memory_bundle(agent_name: str, project_id: str | None = None):
    bundle = []

    for raw_path in get_agent_memory_paths(agent_name):
        raw_name = str(raw_path)

        if raw_name == "__LATEST_DAILY__":
            resolved = latest_daily_file()
            if resolved is None:
                bundle.append({
                    "file": "__LATEST_DAILY__",
                    "exists": False,
                    "content": "",
                })
                continue
            path = resolved

        elif raw_name == "__PROJECT_CURRENT__":
            if not project_id:
                bundle.append({
                    "file": "__PROJECT_CURRENT__",
                    "exists": False,
                    "content": "",
                })
                continue
            path = get_project_current_file(project_id)

        elif raw_name == "__PROJECT_LATEST_DAILY__":
            if not project_id:
                bundle.append({
                    "file": "__PROJECT_LATEST_DAILY__",
                    "exists": False,
                    "content": "",
                })
                continue
            resolved = get_latest_project_daily_file(project_id)
            if resolved is None:
                bundle.append({
                    "file": "__PROJECT_LATEST_DAILY__",
                    "exists": False,
                    "content": "",
                })
                continue
            path = resolved

        else:
            path = raw_path

        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            text = ""

        bundle.append({
            "file": str(path),
            "exists": path.exists(),
            "content": text,
        })

    return bundle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    state = load_state()
    agent_state = state.get(args.agent, {})
    interval = get_agent_refresh_interval(args.agent)

    now = datetime.now(timezone.utc).timestamp()
    last_ts = float(agent_state.get("last_refresh_ts", 0))

    old_event_state = agent_state.get("event_files", {})
    new_event_state = build_event_state(args.project_id)

    interval_elapsed = (now - last_ts) >= interval
    event_changed = event_triggered(old_event_state, new_event_state)

    if not args.force and not interval_elapsed and not event_changed:
        remaining = int(interval - (now - last_ts))
        print("SKIPPED")
        print("reason=interval_not_elapsed_and_no_event_change")
        print(f"agent={args.agent}")
        print(f"seconds_remaining={remaining}")
        return

    bundle = read_memory_bundle(args.agent, args.project_id)

    agent_state = {
        "last_refresh_ts": now,
        "last_refresh_at": now_iso(),
        "interval_seconds": interval,
        "project_id": args.project_id,
        "event_files": new_event_state,
        "refresh_trigger": "force" if args.force else ("event" if event_changed and not interval_elapsed else "interval"),
        "files": [x["file"] for x in bundle],
    }
    state[args.agent] = agent_state
    save_state(state)

    print("REFRESHED")
    print(f"agent={args.agent}")
    print(f"interval_seconds={interval}")
    print(f"files={len(bundle)}")
 
    print(f"trigger={agent_state.get('refresh_trigger')}")


if __name__ == "__main__":
    main()
