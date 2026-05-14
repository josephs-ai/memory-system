"""
Event ledger for tracking memory system timeline events —
appends structured events for later rendering into timeline views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"
WORKSPACE = Path.home() / ".openclaw" / "workspace"
DEFAULT_LEDGER = WORKSPACE / ".memory-index" / "timeline" / "events.jsonl"


def find_latest_session_for_agent(agent_name: str) -> Path | None:
    sessions_dir = AGENTS_DIR / agent_name / "sessions"
    sessions_json = sessions_dir / "sessions.json"
    candidates: list[tuple[int, Path]] = []

    if sessions_json.exists():
        try:
            data = json.loads(sessions_json.read_text(encoding="utf-8"))
            for _, meta in data.items():
                session_file = meta.get("sessionFile")
                updated_at = int(meta.get("updatedAt") or 0)
                if session_file:
                    p = Path(session_file)
                    if p.exists():
                        candidates.append((updated_at, p))
        except Exception:
            pass

    for p in sessions_dir.glob("*.jsonl"):
        try:
            candidates.append((int(p.stat().st_mtime * 1000), p))
        except Exception:
            candidates.append((0, p))

    if not candidates:
        return None
    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[0][1]



def load_session_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows



def _extract_text_blocks(content: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for block in content or []:
        if block.get("type") != "text":
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
    return "\n".join(parts).strip()



def visible_event_records(events: list[dict[str, Any]], *, agent_name: str, session_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("type") != "message":
            continue
        msg = event.get("message") or {}
        role = str(msg.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        text = _extract_text_blocks(msg.get("content"))
        if not text or text == "HEARTBEAT_OK":
            continue
        timestamp = str(event.get("timestamp") or "")
        event_basis = str(event.get("id") or f"{timestamp}:{index}")
        digest = hashlib.sha1(f"{session_path}|{event_basis}|{role}|{text}".encode("utf-8")).hexdigest()[:16]
        records.append(
            {
                "event_id": f"timeline-{digest}",
                "agent_name": agent_name,
                "session_file": str(session_path),
                "source_session": session_path.name,
                "timestamp": timestamp,
                "day_key": timestamp[:10] if timestamp else None,
                "role": role,
                "text": text,
                "source": "transcript_event",
            }
        )
    return records



def append_timeline_events(records: list[dict[str, Any]], ledger_path: Path = DEFAULT_LEDGER) -> dict[str, int]:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                event_id = obj.get("event_id")
                if event_id:
                    existing_ids.add(str(event_id))

    appended = 0
    skipped = 0
    with ledger_path.open("a", encoding="utf-8") as f:
        for record in records:
            event_id = str(record.get("event_id") or "")
            if not event_id or event_id in existing_ids:
                skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_ids.add(event_id)
            appended += 1
    return {"appended": appended, "skipped": skipped}



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent")
    parser.add_argument("--file")
    parser.add_argument("--out", default=str(DEFAULT_LEDGER))
    args = parser.parse_args()

    if args.file:
        session_path = Path(args.file)
        agent_name = args.agent or session_path.parent.parent.parent.name if len(session_path.parts) >= 4 else "unknown"
    elif args.agent:
        session_path = find_latest_session_for_agent(args.agent)
        agent_name = args.agent
    else:
        raise SystemExit("Use --agent NAME or --file PATH")

    if not session_path or not session_path.exists():
        raise SystemExit("Session file not found")

    events = load_session_events(session_path)
    records = visible_event_records(events, agent_name=agent_name, session_path=session_path)
    stats = append_timeline_events(records, Path(args.out))
    print(json.dumps({"ok": True, "session": str(session_path), **stats}))


if __name__ == "__main__":
    main()
