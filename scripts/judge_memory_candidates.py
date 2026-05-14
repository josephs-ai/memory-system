"""
Evaluate and score memory candidates for quality, novelty, and relevance.
Gates items before they enter the durable memory store.
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


items = load_jsonl(Path(args.input_path))
if not items:
    print("NONE")
else:
    for item in items:
        print(json.dumps(item, ensure_ascii=False))
