"""
Read Transcript — utility for the OpenClaw memory system.

Key functions: find_latest_session_for_agent, list_latest_sessions_all_agents, find_latest_session_for_agent, read_events
"""
import json
import argparse
from pathlib import Path
from collections import Counter

OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"

def find_latest_session_for_agent(agent_name: str):
    sessions_dir = AGENTS_DIR / agent_name / "sessions"
    sessions_json = sessions_dir / "sessions.json"

    if not sessions_json.exists():
        return None

    try:
        data = json.loads(sessions_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    latest = None
    latest_ts = -1

    for _, meta in data.items():
        session_file = meta.get("sessionFile")
        updated_at = meta.get("updatedAt", 0)
        if session_file and updated_at > latest_ts:
            latest = Path(session_file)
            latest_ts = updated_at

    return latest

def list_latest_sessions_all_agents():
    results = []
    if not AGENTS_DIR.exists():
        return results

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

def read_events(path: Path):
    events = []
    bad = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                bad += 1

    return events, bad

def summarize(path: Path):
    events, bad = read_events(path)

    print(f"file: {path}")
    print(f"parsed_events: {len(events)}")
    print(f"bad_lines: {bad}")
    print()

    type_counter = Counter()
    role_counter = Counter()

    for e in events:
        t = e.get("type", "<no-type>")
        type_counter[t] += 1

        if t == "message":
            role = e.get("message", {}).get("role", "<no-role>")
            role_counter[role] += 1

    print("event types:")
    for k, v in type_counter.most_common():
        print(f"  {k}: {v}")
    print()

    if role_counter:
        print("message roles:")
        for k, v in role_counter.most_common():
            print(f"  {k}: {v}")
        print()

    print("first 8 events:")
    for e in events[:8]:
        print(json.dumps(e, ensure_ascii=False)[:1200])
        print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--agent")
    parser.add_argument("--all-agents", action="store_true")
    args = parser.parse_args()

    if args.all_agents:
        rows = list_latest_sessions_all_agents()
        if not rows:
            print("No agent sessions found.")
            return
        for agent_name, path in rows:
            print(f"{agent_name}: {path}")
        return

    if args.agent:
        path = find_latest_session_for_agent(args.agent)
        if not path:
            print(f"No latest session found for agent: {args.agent}")
            return
        summarize(path)
        return

    if args.file:
        summarize(Path(args.file))
        return

    print("Use one of: --file PATH | --agent NAME | --all-agents")

if __name__ == "__main__":
    main()
