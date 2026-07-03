"""
Reprocess archived dehydrated transcripts that were skipped by the fast
checkpoint cycle (oversize files, timed-out files).

Runs with longer budgets during quiet hours (nightly).
Reads raw dehydrated .txt files from the archives directory, chunks them,
runs extraction, and routes the results — the same pipeline as
checkpoint_agent.py but without needing the original transcript file.

Usage:
    python3 reprocess_archives.py [--max-files N] [--max-total-seconds N]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
DEHYDRATED_DIR = WORKSPACE / ".memory-index" / "dehydrated"
ARCHIVE_DIR = DEHYDRATED_DIR / "archives"
PROCESSED_DIR = ARCHIVE_DIR / "processed"
CHUNKS_DIR = DEHYDRATED_DIR / "chunks_runtime"

CHUNK_SCRIPT = SCRIPTS_DIR / "chunk_by_topic.py"
EXTRACT_CHUNK_UPDATES_SCRIPT = SCRIPTS_DIR / "extract_chunk_updates.py"
ROUTE_SCRIPT = SCRIPTS_DIR / "route_memory_items_batch.py"
PROCESS_AUTO_SCRIPT = SCRIPTS_DIR / "process_auto_memory_items.py"
PRUNE_SCRIPT = SCRIPTS_DIR / "prune_old_discards.py"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_capture(cmd: list, timeout: int | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"SUBPROCESS_FAILED: {' '.join(str(c) for c in cmd[:3])}", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-4000:], file=sys.stderr)
        result.check_returncode()
    return result.stdout


def discover_archives() -> list[Path]:
    """Return unprocessed archive files, newest first."""
    if not ARCHIVE_DIR.exists():
        return []
    archives = []
    for p in ARCHIVE_DIR.glob("*.txt"):
        if p.stat().st_size == 0:
            continue
        archives.append(p)
    # Newest first — most likely to contain memory the user is missing right now
    archives.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return archives


def reprocess_one(archive: Path, per_file_timeout: int) -> bool:
    """Reprocess a single archived dehydrated transcript. Returns True on success."""
    # Parse agent name from archive filename: {agent}.{transcript_stem}.{ts}.txt
    parts = archive.stem.split(".", 1)
    agent = parts[0] if parts else "unknown"

    chunk_dir = CHUNKS_DIR / f"archive_{agent}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Clear old chunks
    for p in chunk_dir.glob("*.txt"):
        p.unlink()

    print(f"  Chunking {archive.name} ({archive.stat().st_size / 1024:.0f} KB)...")

    # chunk_by_topic expects a transcript (jsonl) but our archive is dehydrated text.
    # Write it as the input directly — chunk_by_topic handles plain text too.
    run_capture([
        "python3", str(CHUNK_SCRIPT),
        "--input", str(archive),
        "--outdir", str(chunk_dir),
    ], timeout=per_file_timeout)

    chunk_count = len(list(chunk_dir.glob("*.txt")))
    print(f"  Produced {chunk_count} chunks.")
    if chunk_count == 0:
        return True  # Nothing to extract

    # Extract
    extracted = run_capture([
        "python3", str(EXTRACT_CHUNK_UPDATES_SCRIPT),
        "--chunk-dir", str(chunk_dir),
        "--source-agent", agent,
        "--source-session", f"archive:{archive.name}",
        "--extraction-mode", "structural",
    ], timeout=per_file_timeout).strip()

    if not extracted or extracted == "NONE":
        print("  No items extracted.")
        return True

    extracted_path = DEHYDRATED_DIR / f"{agent}.archive_extracted.txt"
    extracted_path.write_text(
        extracted + ("\n" if not extracted.endswith("\n") else ""),
        encoding="utf-8",
    )

    # Route
    routed = run_capture([
        "python3", str(ROUTE_SCRIPT),
        "--input", str(extracted_path),
    ], timeout=per_file_timeout).strip()

    print(f"  Routing: {routed[:200]}...")

    # Process AUTO items
    if "AUTO " in routed or "INBOX " in routed:
        run_capture(["python3", str(PROCESS_AUTO_SCRIPT)], timeout=60)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-total-seconds", type=int, default=1800,
                        help="Total budget for quiet-hours reprocessing (default: 30 min)")
    parser.add_argument("--per-file-timeout", type=int, default=300,
                        help="Per-file timeout (default: 5 min — much longer than fast cycle)")
    parser.add_argument("--prune", action="store_true", default=True,
                        help="Also prune old discards (default: True)")
    args = parser.parse_args()

    run_start = time.monotonic()
    archives = discover_archives()

    if not archives:
        print("No archived transcripts to reprocess.")
        if args.prune:
            run_capture(["python3", str(PRUNE_SCRIPT)])
        return

    print(f"Found {len(archives)} archived transcripts. Processing up to {args.max_files}.")

    processed = 0
    failed = 0

    for archive in archives[:args.max_files]:
        elapsed = time.monotonic() - run_start
        if elapsed >= args.max_total_seconds:
            print(f"Budget exhausted after {elapsed:.0f}s.")
            break

        print(f"=== Reprocessing: {archive.name} ===")
        try:
            success = reprocess_one(archive, args.per_file_timeout)
            if success:
                # Move to processed directory
                dest = PROCESSED_DIR / archive.name
                archive.rename(dest)
                processed += 1
                print(f"  Done. Moved to processed/")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {args.per_file_timeout}s — skipping.")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    # Prune old discards while we're here
    if args.prune:
        print("\n=== Pruning old discards ===")
        try:
            run_capture(["python3", str(PRUNE_SCRIPT)])
        except Exception as e:
            print(f"Prune failed: {e}")

    total_elapsed = time.monotonic() - run_start
    print(f"\nSummary: processed={processed} failed={failed} elapsed={total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
