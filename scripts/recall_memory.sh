#!/usr/bin/env bash
set -euo pipefail

QUERY="${1:-current setup}"
TOPK="${2:-5}"

source ~/.openclaw/workspace/.memory-venv/bin/activate
python ~/.openclaw/workspace/.memory-index/scripts/retrieve_memory_hybrid_boosted.py "$QUERY" --topk "$TOPK"
