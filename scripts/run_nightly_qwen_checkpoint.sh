#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/joseph-ding/.openclaw/workspace"
LOG_DIR="$WORKSPACE/.memory-index/logs"
LOCK_FILE="$WORKSPACE/.memory-index/state/nightly_qwen_checkpoint.lock"
mkdir -p "$LOG_DIR" "$WORKSPACE/.memory-index/state"

exec >> "$LOG_DIR/nightly_qwen_checkpoint.log" 2>&1

echo "[$(date --iso-8601=seconds)] nightly_qwen_checkpoint start"

# Avoid overlapping runs if Qwen is slow or the Mac is asleep.
flock -n "$LOCK_FILE" \
  bash -lc '
    set -euo pipefail
    WORKSPACE="/home/joseph-ding/.openclaw/workspace"
    cd "$WORKSPACE"

    # Fast availability check: if the Mac Ollama endpoint is unavailable, skip cleanly.
    if ! curl -fsS --max-time 10 http://192.168.64.1:11435/api/tags >/dev/null; then
      echo "[$(date --iso-8601=seconds)] Mac Ollama unavailable; skipping Qwen hybrid checkpoint"
      exit 0
    fi

    OPENCLAW_WORKSPACE="$WORKSPACE" \
      python3 .memory-index/scripts/checkpoint_agent.py \
        --agent main \
        --reason "nightly Qwen hybrid memory checkpoint" \
        --extraction-mode hybrid \
        --ollama-url http://192.168.64.1:11435 \
        --qwen-model qwen3:32b
  '

status=$?
echo "[$(date --iso-8601=seconds)] nightly_qwen_checkpoint end status=$status"
exit "$status"
