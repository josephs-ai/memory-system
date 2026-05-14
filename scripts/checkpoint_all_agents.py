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
CHECKPOINT_SCRIPT = SCRIPTS_DIR / "checkpoint_agent.py"

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

def main():
    rows = list_agents_with_sessions()
    if not rows:
        print("No agent sessions found.")
        return

    print("Agents with latest sessions:")
    for agent_name, path in rows:
        print(f"- {agent_name}: {path}")
    print()

    for agent_name, _ in rows:
        print(f"===== CHECKPOINT {agent_name} =====")
        result = subprocess.run(
            [
                "python",
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

        print()

if __name__ == "__main__":
    main()
