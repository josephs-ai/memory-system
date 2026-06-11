"""
checkpoint_agent.py — incremental transcript checkpoint pipeline.

Layers:
  S1  Strong cursor: inode/device/hash identity + newline-safe offset
  S2  Replay-window delta: only process bytes since committed cursor
  S3  Delta-aware dehydration: append-only dehydration, atomic + UUID-scoped
"""
import hashlib
import json
import os
import shutil
import argparse
import subprocess
import uuid
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
DEHYDRATED_DIR = WORKSPACE / ".memory-index" / "dehydrated"
CHUNKS_DIR = DEHYDRATED_DIR / "chunks_runtime"
STATE_DIR = WORKSPACE / ".memory-index" / "state"

for _d in (DEHYDRATED_DIR, CHUNKS_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEHYDRATE_SCRIPT = SCRIPTS_DIR / "dehydrate_transcript.py"
CHUNK_SCRIPT = SCRIPTS_DIR / "chunk_by_topic.py"
EXTRACT_CHUNK_UPDATES_SCRIPT = SCRIPTS_DIR / "extract_chunk_updates.py"
ROUTE_SCRIPT = SCRIPTS_DIR / "route_memory_items_batch.py"
PROCESS_AUTO_SCRIPT = SCRIPTS_DIR / "process_auto_memory_items.py"
MAINTENANCE_CYCLE_SCRIPT = SCRIPTS_DIR / "run_memory_maintenance_cycle.py"

CURSOR_VERSION = "v2-hash-window-4096"
HASH_WINDOW_BYTES = 4096
REPLAY_WINDOW_BYTES = 4096


# ---------------------------------------------------------------------------
# S1: Strong cursor helpers
# ---------------------------------------------------------------------------

def _hash_window(path: Path, offset: int) -> str:
    """SHA-256 of up to HASH_WINDOW_BYTES starting at *offset*."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(HASH_WINDOW_BYTES)
    return hashlib.sha256(data).hexdigest()


def build_cursor(transcript: Path, processed_offset: int = 0) -> dict:
    """Build a v2 cursor for *transcript* at *processed_offset*."""
    st = transcript.stat()
    size = st.st_size
    head_hash = _hash_window(transcript, 0) if size > 0 else ""
    tail_start = max(0, size - HASH_WINDOW_BYTES)
    tail_hash = _hash_window(transcript, tail_start)

    # Find last complete newline offset (safe commit boundary)
    safe_end = processed_offset
    if processed_offset == 0 and size > 0:
        # Scan to end, find last newline-terminated byte position
        safe_end = _last_newline_offset(transcript, 0, size)

    return {
        "cursor_version": CURSOR_VERSION,
        "path": str(transcript),
        "inode": st.st_ino,
        "device": st.st_dev,
        "size": size,
        "head_hash": head_hash,
        "tail_hash": tail_hash,
        "processed_offset": processed_offset,
        "committed_offset": safe_end,
    }


def _last_newline_offset(path: Path, start: int, end: int) -> int:
    """Return byte offset just after the last '\\n' in [start, end)."""
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    idx = data.rfind(b"\n")
    if idx == -1:
        return start
    return start + idx + 1


def load_cursor(agent: str) -> dict | None:
    p = STATE_DIR / f"{agent}.cursor.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cursor(agent: str, cursor: dict) -> None:
    p = STATE_DIR / f"{agent}.cursor.json"
    tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(cursor, indent=2), encoding="utf-8")
    shutil.move(str(tmp), str(p))


# ---------------------------------------------------------------------------
# S2: Transcript classification + replay-window delta
# ---------------------------------------------------------------------------

def classify_transcript(transcript: Path, cursor: dict | None) -> dict:
    """
    Classify the current transcript relative to the last cursor.

    Returns dict with keys:
      classification  — fresh_start | same_file_append | same_file_truncated |
                        rotated_or_replaced | unknown_identity_change
      committed_offset
      current_stat    — {size, inode, device}
    """
    st = transcript.stat()
    current_stat = {"size": st.st_size, "inode": st.st_ino, "device": st.st_dev}

    if cursor is None:
        return {"classification": "fresh_start", "committed_offset": 0, "current_stat": current_stat}

    if cursor.get("cursor_version") != CURSOR_VERSION:
        return {"classification": "unknown_identity_change", "committed_offset": 0, "current_stat": current_stat}

    same_inode = (cursor.get("inode") == current_stat["inode"] and
                  cursor.get("device") == current_stat["device"])
    committed = cursor.get("committed_offset", 0)

    if not same_inode:
        # Different inode — rotated or replaced
        return {"classification": "rotated_or_replaced", "committed_offset": 0, "current_stat": current_stat}

    if current_stat["size"] < committed:
        return {"classification": "same_file_truncated", "committed_offset": 0, "current_stat": current_stat}

    if current_stat["size"] == cursor.get("size") and current_stat["size"] == committed:
        # No new bytes
        return {"classification": "same_file_append", "committed_offset": committed, "current_stat": current_stat}

    # Verify head hash still matches (protects against in-place rewrites).
    # Only check when committed_offset >= HASH_WINDOW_BYTES; otherwise the
    # hash window extends into not-yet-committed bytes and any append would
    # change it, producing false unknown_identity_change results.
    if current_stat["size"] > 0 and committed >= HASH_WINDOW_BYTES:
        head_now = _hash_window(transcript, 0)
        if head_now != cursor.get("head_hash", ""):
            return {"classification": "unknown_identity_change", "committed_offset": 0, "current_stat": current_stat}

    return {"classification": "same_file_append", "committed_offset": committed, "current_stat": current_stat}


def read_safe_replay_delta(transcript: Path, committed_offset: int) -> tuple[str, int, int]:
    """
    Read new content from *committed_offset* with a replay overlap window.

    Returns (delta_text, replay_start, safe_end) where:
      - replay_start  = max(0, committed_offset - REPLAY_WINDOW_BYTES)
      - safe_end      = byte offset just past last complete newline in file
      - delta_text    = decoded text from replay_start..safe_end
    """
    st = transcript.stat()
    size = st.st_size
    replay_start = max(0, committed_offset - REPLAY_WINDOW_BYTES)
    safe_end = _last_newline_offset(transcript, 0, size)
    if safe_end <= committed_offset:
        return ("", committed_offset, committed_offset)
    with open(transcript, "rb") as f:
        f.seek(replay_start)
        raw = f.read(safe_end - replay_start)
    return (raw.decode("utf-8", errors="replace"), replay_start, safe_end)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def find_latest_session_for_agent(agent_name: str) -> Path | None:
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


def run_capture(cmd: list) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def clear_old_chunks(chunk_dir: Path) -> None:
    for p in chunk_dir.glob("*.txt"):
        p.unlink()


# ---------------------------------------------------------------------------
# S3: Delta-aware dehydration (atomic + UUID-scoped temp)
# ---------------------------------------------------------------------------

def can_use_delta(classification: str, committed_offset: int, current_stat: dict) -> bool:
    return (
        classification == "same_file_append"
        and committed_offset > 0
        and current_stat["size"] > committed_offset
    )


def run_delta_dehydration(agent: str, dehydrated_file: Path,
                          transcript: Path, committed_offset: int) -> None:
    """
    Append-only dehydration using --offset + --no-header.
    Uses a UUID-scoped temp file and atomic rename for crash safety.
    """
    run_id = uuid.uuid4().hex
    tmp_file = dehydrated_file.with_suffix(f".{run_id}.tmp")

    # Copy existing content into temp file (so rename is atomic swap)
    if dehydrated_file.exists():
        shutil.copy2(str(dehydrated_file), str(tmp_file))
    else:
        tmp_file.touch()

    # Append new delta content
    delta_text = run_capture([
        "python3", str(DEHYDRATE_SCRIPT),
        "--agent", agent,
        "--offset", str(committed_offset),
        "--no-header",
    ])
    with open(tmp_file, "a", encoding="utf-8") as f:
        f.write(delta_text)

    # Atomic rename — only visible after success
    shutil.move(str(tmp_file), str(dehydrated_file))
    print(f"[S3] Delta-appended dehydration to: {dehydrated_file}")


def run_full_dehydration(agent: str, dehydrated_file: Path) -> None:
    dehydrated = run_capture([
        "python3", str(DEHYDRATE_SCRIPT),
        "--agent", agent,
    ])
    run_id = uuid.uuid4().hex
    tmp_file = dehydrated_file.with_suffix(f".{run_id}.tmp")
    tmp_file.write_text(dehydrated, encoding="utf-8")
    shutil.move(str(tmp_file), str(dehydrated_file))
    print(f"[S3] Full dehydration written to: {dehydrated_file}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--reason", default="Structured checkpoint run")
    parser.add_argument("--pair-window", type=int, default=3)
    args = parser.parse_args()

    transcript = find_latest_session_for_agent(args.agent)
    if not transcript:
        print(f"No latest session found for agent: {args.agent}")
        return

    # S1+S2: classify transcript
    cursor = load_cursor(args.agent)
    result = classify_transcript(transcript, cursor)
    classification = result["classification"]
    committed_offset = result["committed_offset"]
    current_stat = result["current_stat"]
    print(f"[S2] classification={classification} committed_offset={committed_offset} size={current_stat['size']}")

    dehydrated_file = DEHYDRATED_DIR / f"{args.agent}.latest.txt"
    chunk_dir = CHUNKS_DIR / args.agent
    chunk_dir.mkdir(parents=True, exist_ok=True)
    clear_old_chunks(chunk_dir)

    # S3: delta or full dehydration
    if can_use_delta(classification, committed_offset, current_stat):
        run_delta_dehydration(args.agent, dehydrated_file, transcript, committed_offset)
    else:
        run_full_dehydration(args.agent, dehydrated_file)

    # S4: delta-aware chunking — only chunk new bytes when possible
    chunk_cmd = [
        "python3", str(CHUNK_SCRIPT),
        "--input", str(transcript),
        "--outdir", str(chunk_dir),
    ]
    if can_use_delta(classification, committed_offset, current_stat):
        chunk_cmd += ["--offset", str(committed_offset)]
        print(f"[S4] Delta chunking from offset={committed_offset}")
    else:
        print("[S4] Full chunking (no eligible delta)")
    run_capture(chunk_cmd)

    extract_cmd = [
        "python3", str(EXTRACT_CHUNK_UPDATES_SCRIPT),
        "--chunk-dir", str(chunk_dir),
        "--source-agent", args.agent,
        "--source-session", str(transcript.name),
    ]
    cursor_file = STATE_DIR / f"{args.agent}.cursor.json"
    if cursor_file.exists():
        extract_cmd += ["--cursor-file", str(cursor_file)]

    extracted = run_capture(extract_cmd).strip()

    extracted_path = DEHYDRATED_DIR / f"{args.agent}.extracted.txt"
    extracted_path.write_text(
        extracted + ("\n" if extracted and not extracted.endswith("\n") else ""),
        encoding="utf-8",
    )

    routed = run_capture([
        "python3", str(ROUTE_SCRIPT),
        "--input", str(extracted_path),
    ]).strip()

    print("Structured routing output:")
    print(routed)
    print()

    if "No structured items to route." in routed or routed.strip() == "NONE":
        print("Checkpoint policy decision: allow=False reason=no-durable-updates")
        print("Skipping checkpoint due to policy.")
        return

    if "AUTO " in routed or "INBOX " in routed or "DISCARDED " in routed or "SKIP_DUPLICATE " in routed:
        print("Checkpoint policy decision: allow=True reason=structured-items-processed")
        print()

        processed = run_capture(["python3", str(PROCESS_AUTO_SCRIPT)]).strip()
        print("AUTO processing output:")
        print(processed)
        print()

        maintenance = run_capture(["python3", str(MAINTENANCE_CYCLE_SCRIPT)]).strip()
        print("Maintenance cycle output:")
        print(maintenance)
        print()

        # S1: save cursor only after successful downstream processing
        new_cursor = build_cursor(transcript, processed_offset=current_stat["size"])
        save_cursor(args.agent, new_cursor)
        print(f"[S1] Cursor saved: committed_offset={new_cursor['committed_offset']}")
        return

    print("Checkpoint policy decision: allow=False reason=no-actionable-structured-output")
    print("Skipping checkpoint due to policy.")


if __name__ == "__main__":
    main()
