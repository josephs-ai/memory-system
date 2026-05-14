"""
Create memory restore point.

Key functions: now_stamp, copy_if_exists, main
"""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"
REVIEW_DIR = MEMORY_DIR / "review"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"
RESTORE_DIR = LOGS_DIR / "restore-points"

RESTORE_DIR.mkdir(parents=True, exist_ok=True)


def now_stamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z").replace(":", "-")


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="manual")
    args = parser.parse_args()

    stamp = now_stamp()
    out_dir = RESTORE_DIR / f"restore-point-{stamp}-{args.label}"

    targets = [
        REVIEW_DIR / "canonical-items.jsonl",
        REVIEW_DIR / "canonical-superseded.jsonl",
        REVIEW_DIR / "pending-stable.jsonl",
        REVIEW_DIR / "inbox.jsonl",
        REVIEW_DIR / "auto.jsonl",
        REVIEW_DIR / "discarded.jsonl",
        REVIEW_DIR / "retrieval-feedback.jsonl",
        REVIEW_DIR / "decisions.log",
        MEMORY_DIR / "projects",
        MEMORY_DIR / "browser.md",
        MEMORY_DIR / "worker.md",
        MEMORY_DIR / "gateway.md",
        MEMORY_DIR / "preferences.md",
        MEMORY_DIR / "memory-system.md",
        MEMORY_DIR / "learned-fixes.md",
    ]

    for src in targets:
        rel = src.relative_to(WORKSPACE)
        dst = out_dir / rel
        copy_if_exists(src, dst)

    print("RESTORE_POINT_CREATED")
    print(f"path={out_dir}")


if __name__ == "__main__":
    main()
