"""
Extract candidate memory items from dehydrated transcript chunks.
Identifies decisions, lessons, facts, preferences, and rules worth remembering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from structured_memory_extractor import extract_structured_candidates


parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("--source-agent", default="unknown")
parser.add_argument("--source-session", default="unknown")
parser.add_argument("--source-chunk", default=None)
args = parser.parse_args()


def main() -> None:
    path = Path(args.input_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = extract_structured_candidates(
        text,
        source_agent=args.source_agent,
        source_session=args.source_session,
        source_chunk=args.source_chunk or path.name,
    )
    if not rows:
        print("NONE")
        return
    for row in rows:
        print(json.dumps(row.to_legacy_item(), ensure_ascii=False))


if __name__ == "__main__":
    main()
