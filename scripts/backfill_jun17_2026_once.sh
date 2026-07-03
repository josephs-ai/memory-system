#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/joseph-ding/.openclaw/workspace"
SCRIPTS="$WORKSPACE/.memory-index/scripts"
OUTROOT="$WORKSPACE/.memory-index/backfill/2026-06-17"
LOGDIR="$WORKSPACE/.memory-index/logs"
LOG="$LOGDIR/backfill-2026-06-17-$(date -u +%Y%m%dT%H%M%SZ).log"
SESSION_LIST="$OUTROOT/jun17_sessions.txt"

mkdir -p "$OUTROOT/dehydrated" "$OUTROOT/chunks" "$OUTROOT/extracted" "$LOGDIR"
exec > >(tee -a "$LOG") 2>&1

cleanup() {
  systemctl --user start openclaw-memory-checkpoint.timer >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[backfill] started $(date -Is)"
echo "[backfill] log=$LOG"

echo "[backfill] stopping checkpoint timer to avoid overlap"
systemctl --user stop openclaw-memory-checkpoint.timer >/dev/null 2>&1 || true
# If a timer-triggered service is running, wait briefly for it to finish rather than killing it.
for i in $(seq 1 24); do
  if ! systemctl --user is-active --quiet openclaw-memory-checkpoint.service; then
    break
  fi
  echo "[backfill] waiting for active checkpoint service... ${i}/24"
  sleep 5
done
if systemctl --user is-active --quiet openclaw-memory-checkpoint.service; then
  echo "[backfill] checkpoint service still active; stopping it to prevent concurrent DB writes"
  systemctl --user stop openclaw-memory-checkpoint.service || true
  sleep 3
fi

find /home/joseph-ding/.openclaw/agents -path '*/sessions/*.jsonl' \
  -newermt '2026-06-17 00:00:00' ! -newermt '2026-06-18 00:00:00' \
  -printf '%p\n' | sort > "$SESSION_LIST"

total=$(wc -l < "$SESSION_LIST" | tr -d ' ')
echo "[backfill] sessions=$total"

processed=0
failed=0
routed_any=0
while IFS= read -r transcript; do
  [ -n "$transcript" ] || continue
  processed=$((processed+1))
  agent=$(awk -F/ '{for(i=1;i<=NF;i++) if($i=="agents") {print $(i+1); exit}}' <<<"$transcript")
  session=$(basename "$transcript")
  safe_session=$(printf '%s' "$session" | tr -c 'A-Za-z0-9_.-' '_')
  stem="${processed}_${agent}_${safe_session}"
  dehydrated="$OUTROOT/dehydrated/$stem.txt"
  chunkdir="$OUTROOT/chunks/$stem"
  extracted="$OUTROOT/extracted/$stem.extracted.txt"
  mkdir -p "$chunkdir"

  echo
  echo "===== [$processed/$total] agent=$agent session=$session ====="
  echo "[backfill] transcript=$transcript"

  if ! python3 "$SCRIPTS/dehydrate_transcript.py" --file "$transcript" > "$dehydrated"; then
    echo "[backfill] ERROR dehydrate failed for $transcript"
    failed=$((failed+1)); continue
  fi

  if ! python3 "$SCRIPTS/chunk_by_topic.py" --input "$transcript" --outdir "$chunkdir"; then
    echo "[backfill] ERROR chunk failed for $transcript"
    failed=$((failed+1)); continue
  fi

  if ! python3 "$SCRIPTS/extract_chunk_updates.py" \
      --chunk-dir "$chunkdir" \
      --source-agent "$agent" \
      --source-session "$session" \
      --extraction-mode structural \
      --ignore-cache > "$extracted"; then
    echo "[backfill] ERROR extract failed for $transcript"
    failed=$((failed+1)); continue
  fi

  if grep -qvE '^\s*$|^NONE$|^No structured items to route\.' "$extracted"; then
    echo "[backfill] routing extracted items"
    if python3 "$SCRIPTS/route_memory_items_batch.py" --input "$extracted"; then
      routed_any=1
    else
      echo "[backfill] ERROR route failed for $transcript"
      failed=$((failed+1)); continue
    fi
  else
    echo "[backfill] no structured items extracted"
  fi
done < "$SESSION_LIST"

echo
echo "[backfill] processing AUTO queue"
python3 "$SCRIPTS/process_auto_memory_items.py" || true

echo
echo "[backfill] running maintenance cycle"
python3 "$SCRIPTS/run_memory_maintenance_cycle.py" || true

echo
echo "[backfill] done $(date -Is) processed=$processed failed=$failed routed_any=$routed_any log=$LOG"
exit $failed
