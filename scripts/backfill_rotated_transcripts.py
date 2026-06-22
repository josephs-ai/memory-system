"""
backfill_rotated_transcripts.py — one-time recovery for un-ingested rotated sessions.

OpenClaw rotates a live session by renaming "<uuid>.jsonl" to
"<uuid>.jsonl.deleted.<ISO8601>". Before the checkpoint pipeline learned to
follow rotated files, any bytes appended after the last committed cursor but
before rotation were never dehydrated/extracted/routed — that conversation
never reached memory.

This tool sweeps every agent's sessions directory for "*.jsonl.deleted.*"
files that still have bytes past their committed cursor and runs each through
checkpoint_agent.checkpoint_transcript(), committing a cursor afterward so the
work is idempotent. Unlike the live discovery path it ignores the recency
window by default (this is a historical recovery).

Usage:
    python3 backfill_rotated_transcripts.py --dry-run
    python3 backfill_rotated_transcripts.py            # process all agents
    python3 backfill_rotated_transcripts.py --agent coder4
    python3 backfill_rotated_transcripts.py --min-bytes 200
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import checkpoint_agent as C

AGENTS_DIR = Path.home() / ".openclaw" / "agents"
ROTATED_GLOB = "*.jsonl.deleted.*"


def uncommitted_rotated(agent: str, min_bytes: int) -> list[tuple[Path, int, int]]:
    """Return (path, committed_offset, size) for rotated files with a real tail."""
    sessions_dir = AGENTS_DIR / agent / "sessions"
    out = []
    if not sessions_dir.exists():
        return out
    for p in sorted(sessions_dir.glob(ROTATED_GLOB)):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue
        cur = C.load_cursor(agent, p)
        committed = (cur or {}).get("committed_offset", 0) if cur else 0
        if size - committed >= min_bytes:
            out.append((p, committed, size))
    # Largest uncommitted tail first (most signal recovered soonest).
    out.sort(key=lambda r: -(r[2] - r[1]))
    return out


def list_agents() -> list[str]:
    if not AGENTS_DIR.exists():
        return []
    return sorted(d.name for d in AGENTS_DIR.iterdir() if (d / "sessions").exists())


def make_args(agent: str, extraction_mode: str) -> SimpleNamespace:
    # checkpoint_transcript only reads these attributes off `args`.
    return SimpleNamespace(
        agent=agent,
        extraction_mode=extraction_mode,
        ollama_url="",
        qwen_model="",
        reason="rotated-transcript-backfill",
        pair_window=3,
        transcript=None,
        no_rotated=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", help="Limit to one agent; default = all agents.")
    ap.add_argument("--min-bytes", type=int, default=64,
                    help="Skip rotated files whose uncommitted tail is smaller than this.")
    ap.add_argument("--extraction-mode", choices=["structural", "qwen", "hybrid"], default="structural")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be processed; no writes.")
    args = ap.parse_args()

    agents = [args.agent] if args.agent else list_agents()

    total_files = 0
    total_bytes = 0
    processed = 0
    failures = 0

    for agent in agents:
        targets = uncommitted_rotated(agent, args.min_bytes)
        if not targets:
            continue
        agent_bytes = sum(size - committed for _, committed, size in targets)
        total_files += len(targets)
        total_bytes += agent_bytes
        print(f"[{agent}] {len(targets)} rotated transcript(s), ~{agent_bytes/1024:.0f} KB uncommitted")

        if args.dry_run:
            for p, committed, size in targets:
                print(f"    would process +{size - committed} bytes  {p.name}")
            continue

        ck_args = make_args(agent, args.extraction_mode)
        for p, committed, size in targets:
            print(f"  -> {p.name} (+{size - committed} bytes)")
            try:
                C.checkpoint_transcript(ck_args, p)
                processed += 1
            except FileNotFoundError:
                print(f"     [skip] vanished before processing: {p.name}")
            except Exception as e:
                failures += 1
                print(f"     [error] {type(e).__name__}: {e}", file=sys.stderr)

    print()
    print(f"SUMMARY files={total_files} uncommitted~={total_bytes/1024:.0f}KB "
          f"processed={processed} failures={failures} dry_run={args.dry_run}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
