import json
import argparse
import re
from pathlib import Path

OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"

LOW_VALUE_PATTERNS = [
    r"let me know if you need anything else",
    r"i have opened google",
    r"i've opened your default browser",
    r"i have opened your browser",
    r"the browser window should pop up",
    r"i can also open x directly",
]

VOLATILE_PATTERNS = [
    r"top ten links",
    r"top 10 links",
    r"here are some of the top search results",
    r"breaking reports",
    r"highly discussed news",
    r"unfiltered real-time reports",
]

MEMORY_SIGNAL_PATTERNS = [
    r"functionality to control browser",
    r"browser automation capabilities",
    r"native `browser` tool",
    r"chrome devtools protocol",
    r"current access",
    r"restricted in my specific tool policy",
    r"execute the `openclaw browser` cli commands",
    r"supports managing different profiles",
]

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

def list_latest_sessions_all_agents():
    rows = []
    if not AGENTS_DIR.exists():
        return rows
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue
        latest = find_latest_session_for_agent(agent_dir.name)
        if latest:
            rows.append((agent_dir.name, latest))
    return rows

def load_events(path: Path):
    events = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events

def strip_sender_wrapper(text: str) -> str:
    text = re.sub(
        r"Sender \(untrusted metadata\):\s*```json.*?```\s*",
        "",
        text,
        flags=re.DOTALL
    )
    return text.strip()

def strip_system_spam(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for line in lines:
        s = line.strip()

        if s.startswith("System: [") and (
            "WhatsApp gateway connected" in s
            or "WhatsApp gateway disconnected" in s
            or "Exec completed" in s
        ):
            continue

        if "Read HEARTBEAT.md if it exists" in s:
            continue
        if "When reading HEARTBEAT.md" in s:
            continue
        if s.startswith("Current time:"):
            continue

        kept.append(line)

    return "\n".join(kept).strip()

def clean_user_text(text: str) -> str:
    text = strip_sender_wrapper(text)
    text = strip_system_spam(text)
    text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
    return text

def extract_assistant_text(content_list):
    texts = []
    for block in content_list:
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text", "").strip()
            if not txt:
                continue
            if txt.startswith("think\n"):
                continue
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
            txt = re.sub(r"</?final>", "", txt, flags=re.IGNORECASE)
            txt = txt.strip()
            if txt:
                texts.append(txt)
    return "\n".join(texts).strip()

def is_low_value_assistant(text: str) -> bool:
    t = text.lower()
    if any(re.search(p, t) for p in MEMORY_SIGNAL_PATTERNS):
        return False
    if any(re.search(p, t) for p in LOW_VALUE_PATTERNS):
        return True
    if any(re.search(p, t) for p in VOLATILE_PATTERNS):
        return True
    return False

def dehydrate_events(events):
    out = []

    for e in events:
        if e.get("type") != "message":
            continue

        msg = e.get("message", {})
        role = msg.get("role")

        if role == "toolResult":
            continue

        content = msg.get("content", [])

        if role == "user":
            parts = []
            for block in content:
                if block.get("type") == "text":
                    txt = clean_user_text(block.get("text", ""))
                    if txt:
                        parts.append(txt)
            text = "\n".join(parts).strip()
            if text:
                out.append(("USER", text))

        elif role == "assistant":
            text = extract_assistant_text(content)
            if not text or text == "HEARTBEAT_OK":
                continue
            if is_low_value_assistant(text):
                continue
            out.append(("ASSISTANT", text))

    return out

def print_dehydrated(label, pairs):
    print(f"===== {label} =====")
    for role, text in pairs:
        print(f"{role}: {text}")
        print()
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--agent")
    parser.add_argument("--all-agents", action="store_true")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        pairs = dehydrate_events(load_events(path))
        print_dehydrated(str(path), pairs)
        return

    if args.agent:
        path = find_latest_session_for_agent(args.agent)
        if not path:
            print(f"No latest session found for agent: {args.agent}")
            return
        pairs = dehydrate_events(load_events(path))
        print_dehydrated(f"{args.agent}: {path}", pairs)
        return

    if args.all_agents:
        rows = list_latest_sessions_all_agents()
        if not rows:
            print("No sessions found.")
            return
        for agent_name, path in rows:
            pairs = dehydrate_events(load_events(path))
            print_dehydrated(f"{agent_name}: {path}", pairs)
        return

    print("Use one of: --file PATH | --agent NAME | --all-agents")

if __name__ == "__main__":
    main()
