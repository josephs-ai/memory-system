"""
Memory system utility: filter memory candidates.

Key functions: load_jsonl, keep_item
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
args = parser.parse_args()


def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s == "NONE":
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                pass
    return items


def keep_item(item: dict) -> bool:
    if item.get("contains_reasoning"):
        return False
    if item.get("contains_speculation"):
        return False
    return True


items = load_jsonl(Path(args.input_path))
kept = [item for item in items if keep_item(item)]
if not kept:
    print("NONE")
else:
    for item in kept:
        print(json.dumps(item, ensure_ascii=False))
