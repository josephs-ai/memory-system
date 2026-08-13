"""
Rebuild the candidate queue from every session transcript, not just the latest.

Routine checkpointing follows each agent's *current* session, so history that
scrolled out of "latest" is never revisited. That is why the queue could not be
restored by re-running checkpoint_all_agents: 493 transcripts exist, but only 22
were reachable.

This walks all of them. Extraction is deterministic (hashing and regexes, no
model calls), so replaying a transcript reproduces the same candidate ids and
routing dedups the repeats.

Deliberately does NOT touch state/*.cursor.json. Those drive the live 5-minute
checkpoint timer, and rewinding them would make routine checkpointing reprocess
history on every tick. Progress is tracked separately here.

    python3 scripts/backfill_all_sessions.py --limit 5     # try a few first
    python3 scripts/backfill_all_sessions.py               # everything, resumable
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

WORKSPACE = Path.home() / ".openclaw" / "workspace"
AGENTS_DIR = Path.home() / ".openclaw" / "agents"
INDEX = WORKSPACE / ".memory-index"
SCRATCH = INDEX / "dehydrated" / "chunks_backfill"
LOGS = INDEX / "logs"
STATE_FILE = LOGS / "backfill_all_sessions_state.json"

CHUNK_SCRIPT = SCRIPT_DIR / "chunk_by_topic.py"
EXTRACT_SCRIPT = SCRIPT_DIR / "extract_chunk_updates.py"
ROUTE_SCRIPT = SCRIPT_DIR / "route_memory_items_batch.py"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run(cmd: list[str], timeout: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR), timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        return False, (proc.stderr or "").strip()[:300]
    return True, proc.stdout


def process_session(path: Path, agent: str, timeout: int) -> tuple[bool, str]:
    """Chunk -> extract -> route one transcript, from byte 0."""
    chunk_dir = SCRATCH / agent / path.stem
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    ok, err = run(
        ["python3", str(CHUNK_SCRIPT), "--input", str(path), "--outdir", str(chunk_dir)],
        timeout,
    )
    if not ok:
        return False, f"chunk: {err}"

    # No --cursor-file: the whole transcript is in scope, which is the point.
    ok, extracted = run(
        [
            "python3", str(EXTRACT_SCRIPT),
            "--chunk-dir", str(chunk_dir),
            "--source-agent", agent,
            "--source-session", path.name,
        ],
        timeout,
    )
    if not ok:
        return False, f"extract: {extracted}"

    extracted = extracted.strip()
    if not extracted:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return True, "no candidates"

    extracted_path = chunk_dir / "extracted.txt"
    extracted_path.write_text(extracted + "\n", encoding="utf-8")

    ok, routed = run(
        ["python3", str(ROUTE_SCRIPT), "--input", str(extracted_path)], timeout
    )
    shutil.rmtree(chunk_dir, ignore_errors=True)
    if not ok:
        return False, f"route: {routed}"

    counts = {}
    for line in routed.splitlines():
        head = line.strip().split(" ")[0]
        if head in {"AUTO", "INBOX", "DISCARDED", "SKIP_DUPLICATE", "PENDING_STABLE"}:
            counts[head] = counts.get(head, 0) + 1
    return True, " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "routed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--agent", default=None, help="restrict to one agent")
    parser.add_argument("--timeout", type=int, default=900, help="per-step seconds")
    parser.add_argument("--redo", action="store_true", help="ignore recorded progress")
    args = parser.parse_args()

    state = {} if args.redo else load_state()
    sessions = sorted(AGENTS_DIR.glob("*/sessions/*.jsonl"))
    if args.agent:
        sessions = [p for p in sessions if p.parent.parent.name == args.agent]

    todo = []
    for path in sessions:
        key = str(path)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        # Re-process if the file grew since it was last handled.
        if state.get(key, {}).get("size") == size:
            continue
        todo.append((path, size))

    print(f"{len(sessions)} transcripts, {len(todo)} to process")
    if args.limit:
        todo = todo[: args.limit]

    done = failed = 0
    started = time.time()
    for i, (path, size) in enumerate(todo, 1):
        agent = path.parent.parent.name
        t0 = time.time()
        ok, detail = process_session(path, agent, args.timeout)
        elapsed = time.time() - t0
        status = "ok " if ok else "FAIL"
        print(
            f"[{i}/{len(todo)}] {status} {agent}/{path.name[:22]} "
            f"{size/1e6:5.1f}MB {elapsed:5.1f}s  {detail[:70]}",
            flush=True,
        )
        if ok:
            state[str(path)] = {"size": size, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
            done += 1
        else:
            failed += 1
        # Persist as we go so an interrupted run resumes rather than restarts.
        if i % 5 == 0 or i == len(todo):
            save_state(state)

    save_state(state)
    print(f"PROCESSED={done} FAILED={failed} ELAPSED={time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
