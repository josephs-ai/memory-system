#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/.openclaw/workspace/.memory-index"
PY="$HOME/.openclaw/venvs/memory-db/bin/python"

export OPENCLAW_MEMORY_DB_DSN="${OPENCLAW_MEMORY_DB_DSN:-dbname=openclaw_memory user=joseph-ding}"
export CLAW_ENV="${CLAW_ENV:-testing}"

$PY -m py_compile "$ROOT/scripts/heartbeat_tuning.py"
$PY -m py_compile "$ROOT/scripts/ci_smoke.py"
$PY "$ROOT/scripts/ci_smoke.py"

