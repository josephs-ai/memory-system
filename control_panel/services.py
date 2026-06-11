"""Data services for the control panel — each function is independently fault-tolerant."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import requests

from config import (
    DB_DSN,
    MEMORY_INDEX,
    NEO4J_PASS,
    NEO4J_URI,
    NEO4J_USER,
    PY,
    QDRANT_URL,
    SCRIPTS_DIR,
    WORKSPACE,
    load_openclaw_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(fn, default=None):
    """Call fn(), return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


def _db_scalar(sql: str, params: tuple = ()) -> int:
    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def _db_rows(sql: str, params: tuple = (), limit: int = 50) -> list[dict]:
    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchmany(limit)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# System overview
# ---------------------------------------------------------------------------

def system_health() -> dict[str, Any]:
    """Overall system health — containers, DB, services."""
    containers = docker_status()
    running = sum(1 for c in containers if "Up" in c.get("status", ""))
    total = len(containers)

    memory_count = _db_scalar("SELECT COUNT(*) FROM memory_items")
    test_result = last_test_result()

    health = "healthy"
    issues = []

    if running < total:
        health = "degraded"
        issues.append(f"{total - running}/{total} containers down")
    if memory_count == 0:
        health = "degraded"
        issues.append("No memory items in DB")
    if test_result.get("failed", 0) > 0:
        health = "warning"
        issues.append(f"{test_result['failed']} tests failing")

    return {
        "health": health,
        "issues": issues,
        "containers_up": running,
        "containers_total": total,
        "memory_items": memory_count,
    }


# ---------------------------------------------------------------------------
# Docker / Containers
# ---------------------------------------------------------------------------

def docker_status() -> list[dict[str, str]]:
    """Get status of all docker containers."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}|{{.Ports}}|{{.Image}}"],
            capture_output=True, text=True, timeout=5,
        )
        results = []
        for line in proc.stdout.strip().splitlines():
            parts = line.split("|", 3)
            if len(parts) >= 2:
                results.append({
                    "name": parts[0],
                    "status": parts[1],
                    "ports": parts[2] if len(parts) > 2 else "",
                    "image": parts[3] if len(parts) > 3 else "",
                })
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Agents (from real OpenClaw config)
# ---------------------------------------------------------------------------

def agent_list() -> list[dict[str, Any]]:
    """Read agents from openclaw.json."""
    cfg = load_openclaw_config()
    agents_cfg = cfg.get("agents", {})
    defaults = agents_cfg.get("defaults", {})
    default_model = defaults.get("model", {})
    if isinstance(default_model, str):
        default_model_name = default_model
    else:
        default_model_name = default_model.get("primary", "unknown")

    agents = []
    for agent in agents_cfg.get("list", []):
        model = agent.get("model", default_model_name)
        if isinstance(model, dict):
            model = model.get("primary", "unknown")
        agents.append({
            "id": agent.get("id", ""),
            "name": agent.get("name", agent.get("id", "")),
            "model": model,
            "workspace": agent.get("workspace", defaults.get("workspace", "")),
            "has_subagents": bool(agent.get("subagents", {}).get("allowAgents")),
            "tools": agent.get("tools", {}),
        })
    return agents


# ---------------------------------------------------------------------------
# Memory stats
# ---------------------------------------------------------------------------

def memory_stats() -> dict[str, int]:
    return {
        "total": _db_scalar("SELECT COUNT(*) FROM memory_items"),
        "active": _db_scalar("SELECT COUNT(*) FROM memory_items WHERE status = 'active'"),
        "superseded": _db_scalar("SELECT COUNT(*) FROM memory_items WHERE status = 'superseded'"),
        "inbox": _db_scalar("SELECT COUNT(*) FROM memory_inbox"),
        "pending_stable": _db_scalar("SELECT COUNT(*) FROM memory_pending_stable"),
        "discarded": _db_scalar("SELECT COUNT(*) FROM memory_discarded"),
        "embeddings": _db_scalar("SELECT COUNT(*) FROM memory_item_embeddings"),
        "feedback": _db_scalar("SELECT COUNT(*) FROM retrieval_feedback"),
        "code_chunks": _db_scalar("SELECT COUNT(*) FROM code_chunks"),
    }


def recent_memory_items(limit: int = 10) -> list[dict]:
    return _db_rows(
        """SELECT id, text, status, memory_type, entity, property, value,
                  created_at, updated_at
           FROM memory_items ORDER BY created_at DESC LIMIT %s""",
        (limit,),
    )


def memory_search(query: str, top_k: int = 10) -> list[dict]:
    """Run hybrid search via the CLI script."""
    try:
        proc = subprocess.run(
            [PY, str(SCRIPTS_DIR / "hybrid_search_cli.py"), "--query", query, "--top-k", str(top_k)],
            capture_output=True, text=True, timeout=30,
            cwd=str(SCRIPTS_DIR),
            env={**os.environ, "OPENCLAW_MEMORY_DB_DSN": DB_DSN},
        )
        results = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line and line != "NONE":
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Code graph (Neo4j + Qdrant)
# ---------------------------------------------------------------------------

def neo4j_stats() -> dict[str, Any]:
    """Get node/relationship counts from Neo4j."""
    try:
        result = subprocess.run(
            ["docker", "exec", "openclaw-neo4j", "cypher-shell",
             "-u", NEO4J_USER, "-p", NEO4J_PASS,
             "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC"],
            capture_output=True, text=True, timeout=10,
        )
        nodes = {}
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.strip().split(",")
            if len(parts) == 2:
                label = parts[0].strip().strip('"')
                count = int(parts[1].strip().strip('"'))
                if label:
                    nodes[label] = count

        result2 = subprocess.run(
            ["docker", "exec", "openclaw-neo4j", "cypher-shell",
             "-u", NEO4J_USER, "-p", NEO4J_PASS,
             "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC"],
            capture_output=True, text=True, timeout=10,
        )
        rels = {}
        for line in result2.stdout.strip().splitlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) == 2:
                rel = parts[0].strip().strip('"')
                count = int(parts[1].strip().strip('"'))
                if rel:
                    rels[rel] = count

        return {
            "status": "online",
            "nodes": nodes,
            "relationships": rels,
            "total_nodes": sum(nodes.values()),
            "total_relationships": sum(rels.values()),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "nodes": {}, "relationships": {}, "total_nodes": 0, "total_relationships": 0}


def qdrant_stats() -> dict[str, Any]:
    """Get collection info from Qdrant."""
    try:
        resp = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        data = resp.json()
        collections = {}
        for col in data.get("result", {}).get("collections", []):
            name = col["name"]
            detail = requests.get(f"{QDRANT_URL}/collections/{name}", timeout=5).json()
            info = detail.get("result", {})
            collections[name] = {
                "points": info.get("points_count", 0),
                "vectors": info.get("vectors_count", 0),
                "status": info.get("status", "unknown"),
            }
        return {"status": "online", "collections": collections}
    except Exception as e:
        return {"status": "error", "error": str(e), "collections": {}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_last_test_cache: dict[str, Any] = {}


def run_tests() -> dict[str, Any]:
    """Run the full pytest suite and return results."""
    global _last_test_cache
    try:
        proc = subprocess.run(
            ["python3", "-m", "pytest", "scripts/", "--tb=line",
             "-W", "ignore",
             "--ignore=scripts/hybrid_search_cli.py"],
            capture_output=True, text=True, timeout=120,
            cwd=str(MEMORY_INDEX),
        )
        output = proc.stdout + proc.stderr
        # Parse summary line
        passed = failed = 0
        for line in output.splitlines():
            if "passed" in line:
                import re
                m = re.search(r"(\d+)\s+passed", line)
                if m:
                    passed = int(m.group(1))
                m2 = re.search(r"(\d+)\s+failed", line)
                if m2:
                    failed = int(m2.group(1))

        result = {
            "passed": passed,
            "failed": failed,
            "exit_code": proc.returncode,
            "output": output[-3000:],  # last 3K chars
            "run_at": datetime.now(timezone.utc).isoformat(),
        }
        _last_test_cache.update(result)
        return result
    except Exception as e:
        return {"passed": 0, "failed": 0, "exit_code": -1, "output": str(e), "run_at": ""}


def last_test_result() -> dict[str, Any]:
    return _last_test_cache if _last_test_cache else {"passed": 0, "failed": 0, "exit_code": -1, "output": "", "run_at": "never"}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_benchmark() -> dict[str, Any]:
    """Run the retrieval benchmark. Returns the report JSON."""
    report_path = MEMORY_INDEX / "health" / "retrieval_benchmark_report.json"
    try:
        proc = subprocess.run(
            [PY, str(SCRIPTS_DIR / "retrieval_benchmark.py"), "run", "--save-baseline"],
            capture_output=True, text=True, timeout=120,
            cwd=str(SCRIPTS_DIR),
            env={**os.environ, "OPENCLAW_MEMORY_DB_DSN": DB_DSN},
        )
        # Read the full report artifact
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        # Fallback: parse aggregate from stdout
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return {"aggregate": json.loads(line), "output": "parsed from stdout"}
                except Exception:
                    pass
        return {"output": proc.stdout[-2000:], "stderr": proc.stderr[-1000:]}
    except Exception as e:
        return {"error": str(e)}


def last_benchmark_result() -> dict[str, Any]:
    """Load the last saved benchmark report, if any."""
    # Try new benchmark suite results first
    results_dir = MEMORY_INDEX / "benchmarks" / "results"
    suite_results = {}
    if results_dir.exists():
        for name in ["niah", "msc", "hotpotqa", "musique", "crud_rag",
                      "temporalqa", "contradiction", "fever"]:
            import glob as _glob
            files = sorted(_glob.glob(str(results_dir / f"{name}_*.json")))
            if files:
                try:
                    with open(files[-1]) as f:
                        suite_results[name] = json.load(f)
                except Exception:
                    pass
    if suite_results:
        return {"suite": suite_results}

    # Fallback to old single-benchmark report
    report_path = MEMORY_INDEX / "health" / "retrieval_benchmark_report.json"
    try:
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Git status
# ---------------------------------------------------------------------------

def git_info() -> dict[str, Any]:
    """Get git repo status."""
    try:
        cwd = str(MEMORY_INDEX)

        log = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )

        dirty_files = [l for l in status.stdout.strip().splitlines() if l.strip()]

        return {
            "branch": branch.stdout.strip(),
            "remote": remote.stdout.strip(),
            "recent_commits": log.stdout.strip().splitlines(),
            "dirty_files": len(dirty_files),
            "dirty_list": dirty_files[:10],
            "clean": len(dirty_files) == 0,
        }
    except Exception as e:
        return {"branch": "?", "remote": "?", "recent_commits": [], "dirty_files": 0, "clean": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Background services
# ---------------------------------------------------------------------------

def service_status() -> dict[str, Any]:
    """Check background service PIDs and status."""
    services = {}

    # Tracked path watcher
    pid_file = MEMORY_INDEX / "runtime" / "tracked_path_watcher.pid"
    services["watcher"] = _check_pid_service(pid_file, "Tracked Path Watcher")

    # Heartbeat worker
    pid_file_hb = MEMORY_INDEX / "runtime" / "heartbeat_worker.pid"
    services["heartbeat"] = _check_pid_service(pid_file_hb, "Heartbeat Worker")

    return services


def _check_pid_service(pid_file: Path, name: str) -> dict:
    if not pid_file.exists():
        return {"name": name, "running": False, "pid": None}
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        return {"name": name, "running": True, "pid": pid}
    except (ProcessLookupError, ValueError):
        return {"name": name, "running": False, "pid": None}
    except PermissionError:
        return {"name": name, "running": True, "pid": pid}  # exists but owned by another user


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

def run_script(script_name: str) -> str:
    """Run a script and return output."""
    script = SCRIPTS_DIR / script_name
    if not script.exists():
        return f"Script not found: {script_name}"
    try:
        env = {**os.environ, "OPENCLAW_MEMORY_DB_DSN": DB_DSN}
        proc = subprocess.run(
            [PY, str(script)],
            capture_output=True, text=True, timeout=30,
            cwd=str(SCRIPTS_DIR), env=env,
        )
        return (proc.stdout + proc.stderr).strip() or f"exit code {proc.returncode}"
    except Exception as e:
        return str(e)
