"""
Generate Rolling Summaries — utility for the OpenClaw memory system.

Key functions: now_iso, fetch_active_projects, fetch_domain_tags_for_project, fetch_durable_reinforced_items
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import POOL, upsert_memory_item

def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def fetch_active_projects():
    sql = "SELECT project_id FROM projects WHERE status = 'active'"
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]

def fetch_domain_tags_for_project(project_id: str) -> list[str]:
    sql = """
        SELECT DISTINCT t
        FROM memory_items m, jsonb_array_elements_text(m.tags) t
        WHERE m.scope = %s
          AND m.status = 'active'
          AND m.lifecycle_state IN ('durable', 'reinforced')
    """
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (project_id,))
            return [row[0] for row in cur.fetchall() if "." in row[0]] # Domain tags convention (e.g. gameplay.movement)


def fetch_durable_reinforced_items(project_id: str, tag: str | None = None):
    sql = """
        SELECT id, text, memory_type, lifecycle_state, tags
        FROM memory_items
        WHERE scope = %s
          AND lifecycle_state IN ('durable', 'reinforced')
          AND status = 'active'
    """
    params = [project_id]
    
    if tag:
        sql += " AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(tags) t WHERE t = %s)"
        params.append(tag)
        
    sql += " ORDER BY retention_priority DESC, created_at DESC"
    
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                rows.append(dict(zip(cols, rec)))
            return rows

def generate_summary(project_id: str, tag: str | None, items: list[dict]) -> str:
    arch = []
    decisions = []
    lessons = []
    active = []
    
    for item in items:
        mtype = item.get("memory_type") or "general"
        mtype = mtype.lower()
        text = item.get("text", "").strip()
        if not text:
            continue
            
        if mtype in {"architecture_rule", "rule"}:
            arch.append(text)
        elif mtype == "decision":
            decisions.append(text)
        elif mtype in {"lesson_learned", "learned_fix"}:
            lessons.append(text)
        else:
            active.append(text)
            
    lines = []
    title = f"Feature Summary: {tag}" if tag else f"Project Summary: {project_id}"
    lines.append(f"# {title}")
    lines.append(f"Generated at: {now_iso()}\n")
    
    if arch:
        lines.append("## Current Architecture & Rules")
        for x in arch:
            lines.append(f"- {x}")
        lines.append("")
        
    if decisions:
        lines.append("## Recent Decisions")
        for x in decisions:
            lines.append(f"- {x}")
        lines.append("")
        
    if lessons:
        lines.append("## Lessons Learned")
        for x in lessons:
            lines.append(f"- {x}")
        lines.append("")
        
    if active:
        lines.append("## Active Systems & Context")
        for x in active:
            lines.append(f"- {x}")
        lines.append("")
        
    if not items:
        lines.append("*No durable or reinforced memory items found for this context.*")
        
    return "\n".join(lines)

def process_project(project_id: str, feature_tag: str | None = None):
    # 1. Generate main project summary
    if not feature_tag:
        items = fetch_durable_reinforced_items(project_id, None)
        if items:
            summary_text = generate_summary(project_id, None, items)
            project_dir = SCRIPT_DIR.parent / "projects" / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            
            out_file = project_dir / "project_summary.md"
            memory_type = "project_summary"
            item_id = f"summary-{project_id}-main"
            
            out_file.write_text(summary_text, encoding="utf-8")
            print(f"Wrote summary to {out_file}")
            
            upsert_memory_item({
                "id": item_id,
                "text": summary_text,
                "memory_type": memory_type,
                "lifecycle_state": "durable",
                "scope": project_id,
                "status": "active",
                "confidence": 1.0,
                "retention_priority": 0.9,
                "source_agent": "system",
                "source_session": "maintenance_cycle",
            })
            print(f"Upserted memory item {item_id}")
            
        # Also auto-generate feature summaries for all domain tags
        tags = fetch_domain_tags_for_project(project_id)
        for t in tags:
            process_project(project_id, feature_tag=t)
        return

    # 2. Generate specific feature summary
    items = fetch_durable_reinforced_items(project_id, feature_tag)
    if not items:
        return
        
    summary_text = generate_summary(project_id, feature_tag, items)
    project_dir = SCRIPT_DIR.parent / "projects" / project_id
    features_dir = project_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    safe_tag = feature_tag.replace(".", "_").replace("/", "_")
    out_file = features_dir / f"{safe_tag}.md"
    memory_type = "feature_summary"
    
    import hashlib
    h = hashlib.sha1(feature_tag.encode("utf-8")).hexdigest()[:8]
    item_id = f"summary-{project_id}-feat-{h}"
    
    out_file.write_text(summary_text, encoding="utf-8")
    print(f"Wrote feature summary to {out_file}")
    
    upsert_memory_item({
        "id": item_id,
        "text": summary_text,
        "memory_type": memory_type,
        "lifecycle_state": "durable",
        "scope": project_id,
        "status": "active",
        "confidence": 1.0,
        "retention_priority": 0.9,
        "source_agent": "system",
        "source_session": "maintenance_cycle",
        "tags": [feature_tag]
    })
    print(f"Upserted memory item {item_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", help="Target project id. If omitted, runs for all active projects.")
    parser.add_argument("--feature-tag", help="Optional domain tag to summarize a specific feature")
    args = parser.parse_args()
    
    projects = [args.project_id] if args.project_id else fetch_active_projects()
    
    for pid in projects:
        process_project(pid, args.feature_tag)

if __name__ == "__main__":
    main()
