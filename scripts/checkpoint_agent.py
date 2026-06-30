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
import sys
import time
import uuid
import re
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

# Raw archive directory — dehydrated transcripts are copied here immediately
# after dehydration, BEFORE the slow extraction/routing steps. This ensures
# transcript content survives even if extraction crashes, times out, or is
# skipped due to oversize limits.
ARCHIVE_DIR = DEHYDRATED_DIR / "archives"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Transcripts larger than this (in bytes) will be dehydrated and archived
# but skip the slow LLM extraction in the normal cycle. They can be
# reprocessed during quiet-hours maintenance. Default: 2 MB.
OVERSIZE_THRESHOLD_BYTES = int(os.environ.get("OPENCLAW_OVERSIZE_THRESHOLD_BYTES", str(2 * 1024 * 1024)))

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
    """Build a v2 cursor for *transcript* at *processed_offset*.

    The committed_offset is always clamped to the last complete-newline
    boundary at or before *processed_offset*. Committing past the last
    newline would mark a partial (still-being-written) trailing line as
    processed; once that line is later completed it would be skipped by
    read_safe_replay_delta (safe_end <= committed_offset), silently
    dropping transcript records. Anchoring on the newline boundary keeps
    the dehydration and chunking deltas line-aligned and replay-safe.
    """
    st = transcript.stat()
    size = st.st_size
    head_hash = _hash_window(transcript, 0) if size > 0 else ""
    tail_start = max(0, size - HASH_WINDOW_BYTES)
    tail_hash = _hash_window(transcript, tail_start)

    # Always commit to the last complete-newline boundary at/below the
    # requested processed_offset. Never commit raw file size, which may
    # include a partial trailing line.
    if size == 0:
        safe_end = 0
    else:
        scan_end = size if processed_offset <= 0 else min(processed_offset, size)
        safe_end = _last_newline_offset(transcript, 0, scan_end)

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


def transcript_cursor_key(transcript: Path) -> str:
    """Stable, path-scoped cursor key for one transcript file.

    The old pipeline used one cursor per agent, which is unsafe when an
    agent has multiple session transcripts (active + rotated/reset files).
    This key intentionally includes the resolved path so each transcript gets
    independent checkpoint state.
    """
    resolved = str(transcript.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", transcript.name)[:96]
    return f"{safe_name}.{digest}"


def cursor_path(agent: str, transcript: Path | None = None) -> Path:
    if transcript is None:
        # Backward-compatible legacy location used by existing tests and any
        # manual tooling that has not yet been migrated.
        return STATE_DIR / f"{agent}.cursor.json"
    return STATE_DIR / "cursors" / agent / f"{transcript_cursor_key(transcript)}.cursor.json"


def load_cursor(agent: str, transcript: Path | None = None) -> dict | None:
    p = cursor_path(agent, transcript)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cursor(agent: str, cursor: dict, transcript: Path | None = None) -> None:
    p = cursor_path(agent, transcript)
    p.parent.mkdir(parents=True, exist_ok=True)
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
    _head = _hash_window(transcript, 0) if st.st_size > 0 else ""
    _tail_start = max(0, st.st_size - HASH_WINDOW_BYTES)
    _tail = _hash_window(transcript, _tail_start)
    current_stat = {"size": st.st_size, "inode": st.st_ino, "device": st.st_dev,
                    "head_hash": _head, "tail_hash": _tail}

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


# Recently-rotated transcripts: OpenClaw does not hard-delete sessions, it
# renames them to "<uuid>.jsonl.deleted.<ISO8601>". The old discovery globbed
# only "*.jsonl", so any bytes appended after the last checkpoint but before
# rotation were never ingested into memory — silent data loss. We treat a
# rotated file as a pending transcript whenever it still has uncommitted bytes
# past its saved cursor (or has no cursor at all).
ROTATED_GLOB = "*.jsonl.deleted.*"
# Only sweep rotations newer than this to bound work; older ones are assumed
# already handled (or intentionally abandoned). Override via env.
ROTATED_LOOKBACK_DAYS = int(os.environ.get("OPENCLAW_ROTATED_LOOKBACK_DAYS", "14"))


def find_pending_transcripts_for_agent(agent_name: str) -> list[Path]:
    """Return transcripts with uncommitted bytes: the active session plus any
    recently-rotated "*.jsonl.deleted.*" files whose tail was never checkpointed.

    A rotated file is included only if it has bytes past its committed cursor
    (or no cursor yet). This makes a rotation while OpenClaw is running
    recoverable instead of dropping the un-checkpointed tail.
    """
    sessions_dir = AGENTS_DIR / agent_name / "sessions"
    pending: list[Path] = []
    seen: set[str] = set()

    latest = find_latest_session_for_agent(agent_name)
    if latest is not None:
        pending.append(latest)
        seen.add(str(latest.resolve()))

    if not sessions_dir.exists():
        return pending

    cutoff = time.time() - ROTATED_LOOKBACK_DAYS * 86400
    rotated = []
    for p in sessions_dir.glob(ROTATED_GLOB):
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime < cutoff or st.st_size == 0:
            continue
        rotated.append((st.st_mtime, p))

    # Oldest-first so ingestion order roughly matches chronology.
    rotated.sort(key=lambda x: x[0])
    for _, p in rotated:
        rp = str(p.resolve())
        if rp in seen:
            continue
        cur = load_cursor(agent_name, p)
        committed = (cur or {}).get("committed_offset", 0)
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > committed:  # uncommitted tail remains
            pending.append(p)
            seen.add(rp)

    return pending


def run_capture(cmd: list) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("SUBPROCESS_FAILED", file=sys.stderr)
        print("command=" + json.dumps(cmd), file=sys.stderr)
        if result.stdout:
            print("--- child stdout ---", file=sys.stderr)
            print(result.stdout[-8000:], file=sys.stderr)
        if result.stderr:
            print("--- child stderr ---", file=sys.stderr)
            print(result.stderr[-8000:], file=sys.stderr)
        result.check_returncode()
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
        "--file", str(transcript),
        "--offset", str(committed_offset),
        "--no-header",
    ])
    with open(tmp_file, "a", encoding="utf-8") as f:
        f.write(delta_text)

    # Atomic rename — only visible after success
    shutil.move(str(tmp_file), str(dehydrated_file))
    print(f"[S3] Delta-appended dehydration to: {dehydrated_file}")


def run_full_dehydration(agent: str, dehydrated_file: Path, transcript: Path) -> None:
    dehydrated = run_capture([
        "python3", str(DEHYDRATE_SCRIPT),
        "--file", str(transcript),
    ])
    run_id = uuid.uuid4().hex
    tmp_file = dehydrated_file.with_suffix(f".{run_id}.tmp")
    tmp_file.write_text(dehydrated, encoding="utf-8")
    shutil.move(str(tmp_file), str(dehydrated_file))
    print(f"[S3] Full dehydration written to: {dehydrated_file}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def checkpoint_transcript(args, transcript: Path) -> None:
    """Run the full delta checkpoint pipeline for a single transcript file."""
    if not transcript.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript}")
    print(f"=== checkpoint transcript: {transcript.name} ===")
    # S1+S2: classify transcript
    cursor = load_cursor(args.agent, transcript)
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
        run_full_dehydration(args.agent, dehydrated_file, transcript)

    # ── Raw archival safety net ───────────────────────────────────────
    # Copy the dehydrated content to a timestamped archive BEFORE the slow
    # extraction/routing steps. This guarantees the transcript content is
    # preserved even if downstream processing crashes, times out, or is
    # skipped due to oversize limits.
    is_rotated = ".jsonl.reset." in transcript.name or ".jsonl.deleted." in transcript.name
    if is_rotated and dehydrated_file.exists():
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        archive_name = f"{args.agent}.{transcript.stem}.{ts}.txt"
        archive_path = ARCHIVE_DIR / archive_name
        try:
            shutil.copy2(str(dehydrated_file), str(archive_path))
            print(f"[ARCHIVE] Raw dehydrated content archived to: {archive_path.name}")
        except Exception as e:
            print(f"[ARCHIVE] WARNING: failed to archive dehydrated content: {e}")

    # ── Oversize guard ────────────────────────────────────────────────
    # Transcripts larger than OVERSIZE_THRESHOLD_BYTES are dehydrated and
    # archived (above) but skip the slow LLM extraction. The cursor is
    # still advanced so the file isn't retried every cycle. These can be
    # reprocessed during quiet-hours maintenance with --reprocess-archives.
    transcript_bytes = current_stat["size"]
    if transcript_bytes > OVERSIZE_THRESHOLD_BYTES:
        print(f"[OVERSIZE] Transcript is {transcript_bytes / (1024*1024):.1f} MB"
              f" (threshold={OVERSIZE_THRESHOLD_BYTES / (1024*1024):.1f} MB)"
              f" — skipping LLM extraction, advancing cursor.")
        try:
            new_cursor = build_cursor(transcript, processed_offset=current_stat["size"])
        except FileNotFoundError:
            new_cursor = {
                "cursor_version": CURSOR_VERSION,
                "path": str(transcript),
                "inode": current_stat.get("inode", 0),
                "device": current_stat.get("device", 0),
                "size": current_stat["size"],
                "head_hash": current_stat.get("head_hash", ""),
                "tail_hash": current_stat.get("tail_hash", ""),
                "committed_offset": current_stat["size"],
                "processed_offset": current_stat["size"],
                "note": "oversize-archive-only",
            }
        save_cursor(args.agent, new_cursor, transcript)
        print(f"[S1] Cursor saved (oversize): committed_offset={new_cursor['committed_offset']}")
        return

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
        "--extraction-mode", args.extraction_mode,
    ]
    if args.ollama_url:
        extract_cmd += ["--ollama-url", args.ollama_url]
    if args.qwen_model:
        extract_cmd += ["--qwen-model", args.qwen_model]
    cursor_file = cursor_path(args.agent, transcript)
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

        # S1: save cursor only after successful downstream processing.
        # Pass raw size as the upper bound; build_cursor clamps the actual
        # committed_offset to the last complete-newline boundary so a partial
        # trailing line is never marked processed (see build_cursor docstring).
        # build_cursor re-stats the transcript. If the file was deleted/renamed
        # between S4 processing and here (e.g. OpenClaw rotated it to
        # *.jsonl.deleted.*), use current_stat from classify_transcript instead.
        try:
            new_cursor = build_cursor(transcript, processed_offset=current_stat["size"])
        except FileNotFoundError:
            print(f"[S1] Transcript moved/deleted after processing; using pre-processing stat for cursor.")
            new_cursor = {
                "cursor_version": CURSOR_VERSION,
                "path": str(transcript),
                "inode": current_stat.get("inode", 0),
                "device": current_stat.get("device", 0),
                "size": current_stat["size"],
                "head_hash": current_stat.get("head_hash", ""),
                "tail_hash": current_stat.get("tail_hash", ""),
                "committed_offset": current_stat["size"],
                "processed_offset": current_stat["size"],
                "note": "built-from-cached-stat-file-deleted",
            }
        save_cursor(args.agent, new_cursor, transcript)
        print(f"[S1] Cursor saved: committed_offset={new_cursor['committed_offset']}")
        return

    print("Checkpoint policy decision: allow=False reason=no-actionable-structured-output")
    print("Skipping checkpoint due to policy.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument(
        "--transcript",
        help="Explicit session transcript path to checkpoint; supports rotated/reset files",
    )
    parser.add_argument("--reason", default="Structured checkpoint run")
    parser.add_argument("--pair-window", type=int, default=3)
    # Optional second-mode extraction. Default remains structural/deterministic.
    parser.add_argument(
        "--extraction-mode", choices=["structural", "qwen", "hybrid"], default="structural"
    )
    parser.add_argument("--ollama-url", default="")
    parser.add_argument("--qwen-model", default="")
    parser.add_argument(
        "--no-rotated",
        action="store_true",
        help="Only checkpoint the active session; skip recently-rotated *.deleted.* tails.",
    )
    args = parser.parse_args()

    # Explicit single-transcript mode (used by tooling / watcher targeting a file).
    if args.transcript:
        transcript = Path(args.transcript).expanduser()
        checkpoint_transcript(args, transcript)
        return

    # Discovery mode: checkpoint the active session PLUS any recently-rotated
    # transcripts whose tail was never committed (issue: logs discarded on
    # rotation while OpenClaw runs were never pushed to memory).
    if args.no_rotated:
        latest = find_latest_session_for_agent(args.agent)
        transcripts = [latest] if latest else []
    else:
        transcripts = find_pending_transcripts_for_agent(args.agent)

    if not transcripts:
        print(f"No latest session found for agent: {args.agent}")
        return

    failures = 0
    for transcript in transcripts:
        try:
            checkpoint_transcript(args, transcript)
        except FileNotFoundError as e:
            # A rotated file can vanish (final hard-delete) between discovery
            # and processing; skip it without aborting the whole run.
            print(f"[skip] {e}")
        except Exception as e:  # isolate one bad transcript from the rest
            failures += 1
            print(f"[error] checkpoint failed for {transcript.name}: {type(e).__name__}: {e}", file=sys.stderr)

    if failures:
        print(f"[done] completed with {failures} transcript failure(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
