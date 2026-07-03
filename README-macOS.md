# macOS Installation

This guide runs Engram/OpenClaw Memory on macOS with Docker-managed storage and
Python services on the host. It avoids Linux-only `systemctl` assumptions.

## What works on macOS

- PostgreSQL, Qdrant, and Neo4j via Docker Desktop or Colima
- Python memory ingestion/checkpoint scripts
- FastAPI search service (`scripts/search_memory_service.py`)
- Hybrid retrieval across PostgreSQL FTS, Qdrant vectors, and optional Neo4j graph context
- Local CPU embedding/reranking models from `sentence-transformers`

## Requirements

- macOS 13+ recommended
- Python 3.11+
- Git
- Docker Desktop **or** Colima + Docker CLI
- `psql` client for manual schema repair/checks

Install common tools with Homebrew:

```bash
brew install git python@3.11 postgresql@16
```

Choose one container runtime:

```bash
# Option A: Docker Desktop
brew install --cask docker
open -a Docker

# Option B: Colima
brew install docker docker-compose colima
colima start --cpu 4 --memory 8 --disk 60
```

Verify Docker Compose is available:

```bash
docker compose version
```

## Install

```bash
git clone https://github.com/josephs-ai/Engram.git
cd Engram

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

If `pip install -e ".[all]"` fails on Apple Silicon because of a model or ML
wheel, first upgrade packaging tools and retry:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
```

## Start storage services

```bash
docker compose up -d
```

This starts:

- PostgreSQL on `localhost:5432`
- Qdrant on `localhost:6333`
- Neo4j HTTP on `localhost:7474`
- Neo4j Bolt on `localhost:7687`

Check health:

```bash
docker compose ps
curl -sf http://localhost:6333/healthz
```

## Configure environment

Use an explicit DSN on macOS. Do not rely on Linux peer/socket auth defaults.

```bash
export OPENCLAW_MEMORY_DSN="host=localhost port=5432 dbname=openclaw_memory user=openclaw password=openclaw"
export OPENCLAW_MEMORY_DB_DSN="$OPENCLAW_MEMORY_DSN"   # legacy alias, kept for older scripts
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4jpassword"
```

Optional: save these in your shell profile or an untracked local `.env` file.
Do not commit secrets or machine-local DSNs.

## Initialize schema

Fresh Docker volumes run `scripts/memory_db_schema.sql` automatically. If you are
using an existing PostgreSQL volume or an external database, run:

```bash
psql "$OPENCLAW_MEMORY_DSN" -f scripts/memory_db_schema.sql
```

If `psql` is not on PATH after Homebrew install, try:

```bash
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
```

## Run a smoke test

```bash
python -m pytest scripts/test_memory_reentry_and_rotation.py scripts/test_extract_chunk_updates_structural.py -q
python scripts/search_memory_service.py --port 8791
```

In another terminal:

```bash
curl "http://localhost:8791/health"
curl "http://localhost:8791/search?q=memory+retrieval&limit=5"
```

The first model load can take a while because sentence-transformer models are
downloaded and cached locally.

## Background services on macOS

Some Linux maintenance helpers use `systemctl --user`. macOS does not have
systemd. Use one of these instead:

1. OpenClaw cron/reminders for periodic checkpoint commands.
2. A `launchd` LaunchAgent.
3. A terminal/tmux session for development.

Minimal `launchd` example for the search service:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>ai.engram.search</string>
  <key>WorkingDirectory</key><string>/ABSOLUTE/PATH/TO/Engram</string>
  <key>ProgramArguments</key>
  <array>
    <string>/ABSOLUTE/PATH/TO/Engram/.venv/bin/python</string>
    <string>scripts/search_memory_service.py</string>
    <string>--port</string><string>8791</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>OPENCLAW_MEMORY_DSN</key><string>host=localhost port=5432 dbname=openclaw_memory user=openclaw password=openclaw</string>
    <key>QDRANT_HOST</key><string>localhost</string>
    <key>QDRANT_PORT</key><string>6333</string>
    <key>NEO4J_URI</key><string>bolt://localhost:7687</string>
    <key>NEO4J_USER</key><string>neo4j</string>
    <key>NEO4J_PASSWORD</key><string>neo4jpassword</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/engram-search.out.log</string>
  <key>StandardErrorPath</key><string>/tmp/engram-search.err.log</string>
</dict>
</plist>
```

Save it as `~/Library/LaunchAgents/ai.engram.search.plist`, replace the absolute
paths, then run:

```bash
launchctl load ~/Library/LaunchAgents/ai.engram.search.plist
launchctl start ai.engram.search
```

## Troubleshooting

### Port conflicts

If local PostgreSQL already uses `5432`, either stop it or change the Compose
port mapping, for example `55432:5432`, then use:

```bash
export OPENCLAW_MEMORY_DSN="host=localhost port=55432 dbname=openclaw_memory user=openclaw password=openclaw"
```

### Docker memory

Neo4j and embedding/reranking can be memory hungry. Give Docker/Colima at least
8 GB RAM if possible.

### Apple Silicon Python wheels

Use Python 3.11 or 3.12 from Homebrew. If a dependency tries to build from
source and fails, update packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Linux-only scripts

Scripts that explicitly call `systemctl --user` are Linux maintenance wrappers.
They are not required for the core memory system. Replace them with `launchd`,
OpenClaw cron, or manual commands on macOS.
