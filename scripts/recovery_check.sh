#!/usr/bin/env bash
# recovery_check.sh — Detects empty Qdrant/Neo4j after crash and rebuilds from PostgreSQL.
# Safe to run on every boot. Idempotent — skips if data already present.
# Exit codes: 0=ok/recovered, 1=recovery failed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEMORY_INDEX="$(dirname "$SCRIPT_DIR")"
VENV="/home/joseph-ding/.openclaw/workspace/.memory-venv"
LOG="$MEMORY_INDEX/logs/recovery.log"
mkdir -p "$MEMORY_INDEX/logs"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

# ── Wait for services ──
wait_for_service() {
    local name="$1" url="$2" max_wait="${3:-60}"
    local elapsed=0
    while ! curl -sf "$url" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$max_wait" ]; then
            log "FATAL: $name not reachable after ${max_wait}s"
            return 1
        fi
    done
    log "$name is up (waited ${elapsed}s)"
}

wait_for_service "Qdrant" "http://localhost:6333/healthz" 90
wait_for_service "Neo4j" "http://localhost:7474" 90

# ── Check PostgreSQL (source of truth) ──
PG_COUNT=$(psql -d openclaw_memory -tAc "SELECT count(*) FROM memory_items WHERE status='active'" 2>/dev/null || echo "0")
if [ "$PG_COUNT" -eq 0 ]; then
    log "WARNING: PostgreSQL has 0 active memory items — nothing to recover from"
    exit 0
fi
log "PostgreSQL has $PG_COUNT active memory items"

# ── Check Qdrant ──
QDRANT_POINTS=$(curl -s http://localhost:6333/collections/memory_items 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('points_count',0))" 2>/dev/null || echo "MISSING")

NEEDS_QDRANT_REBUILD=false
if [ "$QDRANT_POINTS" = "MISSING" ] || [ "$QDRANT_POINTS" -lt "$((PG_COUNT / 2))" ] 2>/dev/null; then
    NEEDS_QDRANT_REBUILD=true
    log "Qdrant needs rebuild: points=$QDRANT_POINTS vs pg_active=$PG_COUNT"
fi

# ── Check Neo4j ──
NEO4J_NODES=$(curl -s -X POST http://localhost:7474/db/neo4j/query/v2 \
    -H "Content-Type: application/json" \
    -u neo4j:neo4jpassword \
    -d '{"statement":"MATCH (n) RETURN count(n) AS c"}' 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data']['values'][0][0])" 2>/dev/null || echo "0")

NEEDS_NEO4J_REBUILD=false
if [ "$NEO4J_NODES" -lt 10 ]; then
    NEEDS_NEO4J_REBUILD=true
    log "Neo4j needs rebuild: nodes=$NEO4J_NODES"
fi

# ── Rebuild if needed ──
if [ "$NEEDS_QDRANT_REBUILD" = false ] && [ "$NEEDS_NEO4J_REBUILD" = false ]; then
    log "All systems healthy — no recovery needed (qdrant=$QDRANT_POINTS, neo4j=$NEO4J_NODES)"
    exit 0
fi

log "=== RECOVERY STARTING ==="
source "$VENV/bin/activate"

if [ "$NEEDS_QDRANT_REBUILD" = true ]; then
    log "Recreating Qdrant memory_items collection (384-dim cosine)..."
    # Delete if exists (ignore errors)
    curl -s -X DELETE http://localhost:6333/collections/memory_items >/dev/null 2>&1 || true
    curl -s -X DELETE http://localhost:6333/collections/code_chunks >/dev/null 2>&1 || true
    sleep 1

    curl -sf -X PUT http://localhost:6333/collections/memory_items \
        -H "Content-Type: application/json" \
        -d '{"vectors":{"size":384,"distance":"Cosine"}}' >/dev/null
    curl -sf -X PUT http://localhost:6333/collections/code_chunks \
        -H "Content-Type: application/json" \
        -d '{"vectors":{"size":384,"distance":"Cosine"}}' >/dev/null

    log "Embedding memory items into Qdrant..."
    python "$SCRIPT_DIR/embed_memory_items.py" >> "$LOG" 2>&1
    FINAL_POINTS=$(curl -s http://localhost:6333/collections/memory_items \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['points_count'])" 2>/dev/null || echo "?")
    log "Qdrant memory_items rebuild complete: $FINAL_POINTS points"

    log "Embedding code chunks into Qdrant..."
    python "$SCRIPT_DIR/embed_code_chunks.py" >> "$LOG" 2>&1
    log "Qdrant code_chunks rebuild complete"
fi

if [ "$NEEDS_NEO4J_REBUILD" = true ]; then
    log "Rebuilding Neo4j code graph..."
    python "$SCRIPT_DIR/sync_code_graph.py" >> "$LOG" 2>&1
    log "Neo4j rebuild complete"
fi

log "=== RECOVERY COMPLETE ==="
exit 0
