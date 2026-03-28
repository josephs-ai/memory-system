from __future__ import annotations

import json
import os
import psycopg
from collections import defaultdict
from typing import Any

from orchestrator.operator_status_summary import build_operator_summary, recent_milestone_operator_summaries

from config import CONFIG_PATH, CONTROL_PANEL_DIR, DEFAULT_CONFIG, PROJECTS_DIR, SCRIPTS_DIR
from data_architecture import ARCHITECTURE_CROSS_EDGES, ARCHITECTURE_TREES


def safe_html(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def ensure_config() -> None:
    CONTROL_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


def load_config() -> dict[str, Any]:
    ensure_config()
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))

    cfg.setdefault("agents", [])
    cfg.setdefault("tracked_paths", [])
    cfg.setdefault("pipeline", DEFAULT_CONFIG["pipeline"].copy())
    cfg.setdefault("heartbeat", DEFAULT_CONFIG["heartbeat"].copy())
    cfg.setdefault("commands", DEFAULT_CONFIG["commands"].copy())

    for agent in cfg["agents"]:
        agent.setdefault("enabled", True)
        agent.setdefault("in_pipeline", True)
        agent.setdefault("role", "custom")
        agent.setdefault("heartbeat_enabled", True)
        agent.setdefault("heartbeat_mode", "standard")
        agent.setdefault("shortcut", agent.get("name", ""))

    return cfg


def save_config(config: dict[str, Any]) -> None:
    CONTROL_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def project_summaries() -> list[dict[str, Any]]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for p in sorted(PROJECTS_DIR.iterdir()):
        if not p.is_dir():
            continue
        current_file = p / "current.md"
        daily_dir = p / "daily"
        milestone_files = sorted(p.glob("milestone-*.md"))
        daily_count = len(list(daily_dir.glob("*.md"))) if daily_dir.exists() else 0
        out.append(
            {
                "name": p.name,
                "path": str(p),
                "has_current": current_file.exists(),
                "milestones": len(milestone_files),
                "daily_count": daily_count,
            }
        )
    return out


def get_all_script_files() -> list[str]:
    if not SCRIPTS_DIR.exists():
        return []
    return sorted(p.name for p in SCRIPTS_DIR.glob("*.py"))


def classify_script(script_name: str) -> str:
    s = script_name.lower()
    if any(x in s for x in ["dehydrate", "chunk", "extract_chunk", "filter_memory", "judge_memory", "extract_memory_fields"]):
        return "Ingest + Dehydrate"
    if any(x in s for x in ["route_memory", "approve_pending", "reject_pending", "promote_memory", "demote_memory", "restore_memory", "auto_promote", "process_auto"]):
        return "Routing + Review"
    if any(x in s for x in ["search_memory", "embed_memory", "mark_retrieval", "feedback", "top_retrieved", "low_utility"]):
        return "Retrieval + Tensor"
    if any(x in s for x in ["sync_registry", "write_memory", "select_best", "match_memory", "reconcile", "patch_memory", "memory_db"]):
        return "Canonical Registry"
    if any(x in s for x in ["conflict", "maintenance", "show_", "why_was", "report_"]):
        return "Inspection + Diagnostics"
    if "migrate_" in s:
        return "Migration Utilities"
    return "Other / Unmapped"


def script_inventory_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for script_name in get_all_script_files():
        groups[classify_script(script_name)].append(script_name)
    return dict(groups)


def all_stage_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for tree in ARCHITECTURE_TREES:
        for node in tree["nodes"]:
            merged = dict(node)
            merged["tree_id"] = tree["id"]
            merged["tree_title"] = tree["title"]
            merged["tree_color"] = tree["color"]
            nodes.append(merged)
    return nodes


def mermaid_id(node_id: str) -> str:
    return "node_" + node_id.replace("-", "_")


def build_mermaid_diagram() -> str:
    lines = [
        "flowchart TB",
        "classDef source fill:#15243d,stroke:#6ea8fe,color:#edf2ff,stroke-width:2px;",
        "classDef policy fill:#1f2940,stroke:#8cb4ff,color:#edf2ff,stroke-width:2px;",
        "classDef pipeline fill:#1c2a35,stroke:#4dd4ac,color:#edf2ff,stroke-width:2px;",
        "classDef queue fill:#2f2a1c,stroke:#ffd166,color:#edf2ff,stroke-width:2px;",
        "classDef database fill:#271d35,stroke:#c084fc,color:#edf2ff,stroke-width:2px;",
        "classDef tensor fill:#183535,stroke:#2dd4bf,color:#edf2ff,stroke-width:2px;",
        "classDef project fill:#2b2233,stroke:#fb7185,color:#edf2ff,stroke-width:2px;",
        "classDef inventory fill:#2b2233,stroke:#fb7185,color:#edf2ff,stroke-width:2px;",
        "classDef component fill:#162642,stroke:#6ea8fe,color:#edf2ff,stroke-width:2px;",
    ]

    click_lines = []

    for tree in ARCHITECTURE_TREES:
        lines.append(f"subgraph {tree['id']}[{tree['title']}]")
        lines.append("direction LR")
        for node in tree["nodes"]:
            node_mid = mermaid_id(node["id"])
            label = f"{node['title']}<br/><span style='font-size:11px'>{node['subtitle']}</span>"
            lines.append(f'    {node_mid}["{label}"]')
            click_lines.append(f"click {node_mid} call openArchNode('{node['id']}')")
        for src, dst, edge_label in tree["edges"]:
            lines.append(f"    {mermaid_id(src)} -- {edge_label} --> {mermaid_id(dst)}")
        lines.append("end")

    for src, dst, edge_label in ARCHITECTURE_CROSS_EDGES:
        lines.append(f"{mermaid_id(src)} -. {edge_label} .-> {mermaid_id(dst)}")

    for node in all_stage_nodes():
        lines.append(f"class {mermaid_id(node['id'])} {node['kind']};")

    lines.extend(click_lines)
    return "\n".join(lines)

def find_node(node_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for tree in ARCHITECTURE_TREES:
        for node in tree["nodes"]:
            if node["id"] == node_id:
                merged = dict(node)
                merged["tree_title"] = tree["title"]
                merged["tree_color"] = tree["color"]
                return tree, merged
    raise KeyError(node_id)


def get_db_dsn() -> str:
    return os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory user=postgres")


def db_scalar(sql: str, params: tuple = ()) -> int:
    try:
        with psycopg.connect(get_db_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def memory_status_counts() -> dict[str, int]:
    return {
        "active_memories": db_scalar("SELECT COUNT(*) FROM memory_items WHERE status = %s", ("active",)),
        "superseded_memories": db_scalar("SELECT COUNT(*) FROM memory_items WHERE status = %s", ("superseded",)),
        "inbox_queue": db_scalar("SELECT COUNT(*) FROM memory_inbox"),
        "pending_stable_queue": db_scalar("SELECT COUNT(*) FROM memory_pending_stable"),
        "discarded_queue": db_scalar("SELECT COUNT(*) FROM memory_discarded"),
        "embedding_rows": db_scalar("SELECT COUNT(*) FROM memory_item_embeddings"),
        "retrieval_feedback_rows": db_scalar("SELECT COUNT(*) FROM retrieval_feedback"),
    }


def project_db_rows(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        with psycopg.connect(get_db_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def orchestrator_operator_overview(limit: int = 6) -> dict[str, Any]:
    rows = project_db_rows(
        """
        SELECT
            work_id,
            title,
            status,
            next_action,
            COALESCE(metadata_json, '{}'::jsonb) AS metadata_json
        FROM work_items
        ORDER BY updated_at DESC NULLS LAST, created_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    work_items = []
    for row in rows:
        metadata = dict(row.get("metadata_json") or {})
        work_items.append(
            {
                "work_id": row.get("work_id"),
                "title": row.get("title"),
                "status": row.get("status"),
                "next_action": row.get("next_action"),
                "summary": build_operator_summary(
                    work_context={
                        "metadata_json": metadata,
                        "execution_rollup": metadata.get("execution_rollup") or {},
                        "execution_governance": metadata.get("execution_governance") or {},
                        "next_action": row.get("next_action"),
                    }
                ),
            }
        )

    return {
        "work_items": work_items,
        "milestones": recent_milestone_operator_summaries(limit=4),
    }
