"""
Checkpoint all agents state for crash recovery.

Key functions: find_latest_session_for_agent, list_agents_with_sessions, main
"""
import json
import subprocess
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"
CHECKPOINT_SCRIPT = SCRIPTS_DIR / "checkpoint_agent.py"
STATE_FILE = LOGS_DIR / "checkpoint_all_agents_state.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

def find_latest_session_for_agent(agent_name: str):
    sessions_dir = AGENTS_DIR / agent_name / "sessions"
    sessions_json = sessions_dir / "sessions.json"

    candidates = []

    if sessions_json.exists():
        try:
            data = json.loads(sessions_json.read_text(encoding="utf-8"))
            for _, meta in data.items():
                session_file = meta.get("sessionFile")
                updated_at = meta.get("updatedAt", 0)
                if session_file:
                    p = Path(session_file)
                    if p.exists():
                        candidates.append((updated_at, p))
        except Exception:
            pass

    for p in sessions_dir.glob("*.jsonl"):
        try:
            mtime = int(p.stat().st_mtime * 1000)
        except Exception:
            mtime = 0
        candidates.append((mtime, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def list_agents_with_sessions():
    rows = []
    if not AGENTS_DIR.exists():
        return rows

    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name
        latest = find_latest_session_for_agent(agent_name)
        if latest:
            rows.append((agent_name, latest))
    return rows


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def session_signature(path: Path) -> dict:
    st = path.stat()
    return {
        "session_file": str(path),
        "mtime_ms": int(st.st_mtime * 1000),
        "size": int(st.st_size),
    }


def should_checkpoint(agent_name: str, latest: Path, state: dict) -> bool:
    sig = session_signature(latest)
    prev = state.get(agent_name) or {}
    return not (
        prev.get("session_file") == sig["session_file"]
        and prev.get("mtime_ms") == sig["mtime_ms"]
        and prev.get("size") == sig["size"]
    )


def main():
    rows = list_agents_with_sessions()
    if not rows:
        print("No agent sessions found.")
        return

    state = load_state()

    print("Agents with latest sessions:")
    for agent_name, path in rows:
        marker = "checkpoint" if should_checkpoint(agent_name, path, state) else "skip_unchanged"
        print(f"- {agent_name}: {path} [{marker}]")
    print()

    changed = 0
    skipped = 0

    for agent_name, latest in rows:
        if not should_checkpoint(agent_name, latest, state):
            print(f"===== SKIP {agent_name} =====")
            print("No session changes since last checkpoint.")
            print()
            skipped += 1
            continue

        print(f"===== CHECKPOINT {agent_name} =====")
        result = subprocess.run(
            [
                "python3",
                str(CHECKPOINT_SCRIPT),
                "--agent", agent_name,
                "--reason", f"Step 25 cross-agent checkpoint ({agent_name})"
            ],
            text=True,
            capture_output=True
        )

        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print("STDERR:")
            print(result.stderr.strip())

        if result.returncode == 0:
            state[agent_name] = session_signature(latest)
            changed += 1
        else:
            print(f"Checkpoint failed for {agent_name}; state not advanced.")

        print()

    save_state(state)
    print(f"Summary: checkpointed={changed} skipped_unchanged={skipped} total={len(rows)}")

if __name__ == "__main__":
    main()
