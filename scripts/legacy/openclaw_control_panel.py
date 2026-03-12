from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI(title="OpenClaw Control Panel")

WORKSPACE = Path.home() / ".openclaw" / "workspace"
CONTROL_DIR = WORKSPACE / ".memory-index" / "control-panel"
CONFIG_PATH = CONTROL_DIR / "control_panel_config.json"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
MEMORY_DIR = WORKSPACE / "memory"
PROJECTS_DIR = MEMORY_DIR / "projects"

DEFAULT_CONFIG: dict[str, Any] = {
    "agents": [
        {
            "name": "main",
            "enabled": True,
            "in_pipeline": True,
            "role": "orchestrator",
            "heartbeat_enabled": True,
            "heartbeat_mode": "strict",
            "shortcut": "main",
        },
        {
            "name": "builder_ai",
            "enabled": True,
            "in_pipeline": True,
            "role": "builder",
            "heartbeat_enabled": True,
            "heartbeat_mode": "standard",
            "shortcut": "build",
        },
    ],
    "tracked_paths": [
        str(WORKSPACE / "memory"),
        str(WORKSPACE / "MEMORY.md"),
    ],
    "pipeline": {
        "judge_enabled": True,
        "embedding_enabled": True,
        "auto_promote_enabled": True,
        "pending_stable_required": True,
    },
    "heartbeat": {
        "heartbeat_enabled": True,
        "heartbeat_mode": "strict",
        "heartbeat_interval_seconds": 300,
        "heartbeat_file": str(WORKSPACE / "HEARTBEAT.md"),
        "heartbeat_agent_file": str(WORKSPACE / "HEARTBEAT_AGENTS.md"),
    },
    "commands": {
        "maintenance_cycle": f"python3 {SCRIPTS_DIR / 'run_memory_maintenance_cycle.py'}",
        "embed_active": f"python3 {SCRIPTS_DIR / 'embed_memory_items.py'} --batch-size 64",
        "sync_markdown": f"python3 {SCRIPTS_DIR / 'sync_registry_to_markdown.py'}",
        "show_queue": f"python3 {SCRIPTS_DIR / 'show_review_queue.py'} --limit 20",
        "report_maintenance": f"python3 {SCRIPTS_DIR / 'report_memory_maintenance.py'}",
    },
}

ROLE_OPTIONS = [
    "orchestrator",
    "builder",
    "reviewer",
    "tester",
    "planner",
    "researcher",
    "memory",
    "custom",
]
HEARTBEAT_MODES = ["off", "light", "standard", "strict"]

ARCHITECTURE_NODES = [
    {
        "id": "client",
        "title": "Client / Control UI",
        "subtitle": "Dashboard, browser entry, quick commands",
        "x": 80,
        "y": 40,
        "w": 230,
        "h": 76,
        "group": "entry",
        "scripts": ["openclaw_control_panel.py"],
        "details": [
            "Agent registry and heartbeat settings",
            "Project tracker and command runner",
            "Architecture explorer with clickable blocks",
        ],
    },
    {
        "id": "heartbeat",
        "title": "Heartbeat Layer",
        "subtitle": "Global + agent heartbeat policy",
        "x": 390,
        "y": 40,
        "w": 230,
        "h": 76,
        "group": "entry",
        "scripts": ["HEARTBEAT.md", "HEARTBEAT_AGENTS.md"],
        "details": [
            "Global heartbeat mode and interval",
            "Per-agent heartbeat mode and enablement",
            "Tracked path awareness",
        ],
    },
    {
        "id": "dehydrate",
        "title": "Dehydration / Extraction",
        "subtitle": "Transcript to candidate pipeline",
        "x": 80,
        "y": 180,
        "w": 260,
        "h": 96,
        "group": "pipeline",
        "scripts": [
            "dehydrate_transcript.py",
            "chunk_by_topic.py",
            "extract_chunk_updates.py",
            "filter_memory_candidates.py",
            "judge_memory_candidates.py",
            "extract_memory_fields.py",
        ],
        "details": [
            "Chunks latest transcript state",
            "Filters raw transcript blobs and vague facts",
            "Classifies memory type, scope, entity, and route hints",
        ],
    },
    {
        "id": "routing",
        "title": "Routing / Approval Queues",
        "subtitle": "Inbox, discarded, pending stable",
        "x": 400,
        "y": 180,
        "w": 270,
        "h": 96,
        "group": "pipeline",
        "scripts": [
            "route_memory_item.py",
            "route_memory_items_batch.py",
            "approve_pending_memory_item.py",
            "reject_pending_memory_item.py",
            "auto_promote_safe_items.py",
            "process_auto_memory_items.py",
        ],
        "details": [
            "Routes judged candidates into DB queues",
            "Manual approval path for strong stable memories",
            "Auto-promotion and queue processing",
        ],
    },
    {
        "id": "canonical",
        "title": "Canonical Registry / DB",
        "subtitle": "Postgres + queue tables + active memory",
        "x": 730,
        "y": 180,
        "w": 270,
        "h": 96,
        "group": "pipeline",
        "scripts": [
            "memory_db.py",
            "promote_memory_item.py",
            "demote_memory_item.py",
            "restore_memory_item.py",
            "write_memory_item.py",
            "sync_registry_to_markdown.py",
        ],
        "details": [
            "memory_items source of truth",
            "pending, inbox, discarded queues",
            "Markdown sync for human-readable memory files",
        ],
    },
    {
        "id": "retrieval",
        "title": "Retrieval / Ranking",
        "subtitle": "FTS + vector + structured bonuses",
        "x": 250,
        "y": 350,
        "w": 280,
        "h": 96,
        "group": "retrieval",
        "scripts": [
            "search_memory.py",
            "search_memory_rbac.py",
            "select_best_registry_item.py",
            "match_memory_items.py",
            "reconcile_memory_items.py",
            "patch_memory_file.py",
        ],
        "details": [
            "Hybrid retrieval with pgvector and FTS",
            "RBAC-aware search path",
            "Registry item selection and reconciliation",
        ],
    },
    {
        "id": "embeddings",
        "title": "Embeddings / Tensor Path",
        "subtitle": "Sentence-Transformers + vector table",
        "x": 610,
        "y": 350,
        "w": 280,
        "h": 96,
        "group": "retrieval",
        "scripts": [
            "embed_memory_items.py",
            "memory_item_embeddings table",
            "all-MiniLM-L6-v2",
        ],
        "details": [
            "Local sentence-transformers model",
            "Backfills and updates vector rows",
            "Hybrid ranking combines vector, FTS, and structure",
        ],
    },
    {
        "id": "maintenance",
        "title": "Maintenance / Inspection",
        "subtitle": "Reports, top retrievals, conflicts, review",
        "x": 120,
        "y": 520,
        "w": 310,
        "h": 112,
        "group": "ops",
        "scripts": [
            "show_review_queue.py",
            "report_memory_maintenance.py",
            "report_feedback_actions.py",
            "show_top_retrieved.py",
            "show_conflicts.py",
            "show_conflict_candidates.py",
            "show_superseded.py",
            "show_recent_writes.py",
            "show_demotion_candidates.py",
            "show_promotion_candidates.py",
            "why_was_this_written.py",
            "why_was_this_retrieved.py",
            "report_semantic_conflicts.py",
        ],
        "details": [
            "Operational reports and debugging tools",
            "Promotion, demotion, conflict, and recent write views",
            "Why-written and why-retrieved explainers",
        ],
    },
    {
        "id": "projects",
        "title": "Project Tracker / File Memory",
        "subtitle": "Tracked projects, current.md, milestone, daily",
        "x": 520,
        "y": 540,
        "w": 320,
        "h": 112,
        "group": "ops",
        "scripts": [
            "memory/projects/*/current.md",
            "memory/projects/*/milestone-*.md",
            "memory/projects/*/daily/*.md",
        ],
        "details": [
            "Project-scoped memory lives alongside canonical DB memory",
            "Control panel can create new projects and files",
            "Search can blend canonical and project files",
        ],
    },
]

ARCHITECTURE_EDGES = [
    ("client", "heartbeat"),
    ("client", "dehydrate"),
    ("heartbeat", "dehydrate"),
    ("dehydrate", "routing"),
    ("routing", "canonical"),
    ("canonical", "retrieval"),
    ("embeddings", "retrieval"),
    ("canonical", "embeddings"),
    ("canonical", "maintenance"),
    ("retrieval", "maintenance"),
    ("projects", "retrieval"),
    ("client", "projects"),
]


def ensure_config() -> None:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
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
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def safe_html(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nav_bar(active: str) -> str:
    items = [
        ("/", "Overview", "overview"),
        ("/agents", "Agents", "agents"),
        ("/heartbeat", "Heartbeat", "heartbeat"),
        ("/projects", "Projects", "projects"),
        ("/architecture", "Architecture", "architecture"),
        ("/commands", "Commands", "commands"),
    ]
    links = []
    for href, label, key in items:
        cls = "nav-link active" if active == key else "nav-link"
        links.append(f"<a class='{cls}' href='{href}'>{label}</a>")
    return "<div class='nav'>" + "".join(links) + "</div>"


def page_shell(title: str, active: str, body: str, last_output: str = "") -> str:
    escaped_output = safe_html(last_output or "No command run yet.")
    return f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>{safe_html(title)}</title>
  <style>
    :root {{ --bg:#08111f; --card:#111a2b; --card2:#162238; --border:#2c3a57; --text:#edf2ff; --muted:#9fb0d5; --accent:#6ea8fe; --accent-soft:#17345f; --green:#30c48d; --yellow:#ffd166; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, Arial, sans-serif; background:linear-gradient(180deg,#07101c,#0d1627); color:var(--text); }}
    .wrap {{ max-width:1360px; margin:0 auto; padding:20px; }}
    .nav {{ display:flex; gap:10px; margin-bottom:18px; flex-wrap:wrap; }}
    .nav-link {{ text-decoration:none; color:var(--text); background:rgba(255,255,255,0.04); border:1px solid var(--border); padding:10px 14px; border-radius:12px; }}
    .nav-link.active {{ background:var(--accent); color:#08111f; border-color:var(--accent); font-weight:700; }}
    .hero {{ margin-bottom:18px; }}
    .hero h1 {{ margin:0 0 6px 0; font-size:30px; }}
    .hero p {{ margin:0; color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(12, 1fr); gap:16px; }}
    .card {{ background:rgba(17,26,43,0.92); border:1px solid var(--border); border-radius:18px; padding:18px; box-shadow:0 2px 18px rgba(0,0,0,0.25); }}
    .span-12 {{ grid-column:span 12; }} .span-8 {{ grid-column:span 8; }} .span-6 {{ grid-column:span 6; }} .span-4 {{ grid-column:span 4; }}
    h2,h3 {{ margin-top:0; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ border-bottom:1px solid rgba(255,255,255,0.08); padding:10px 8px; text-align:left; vertical-align:top; }}
    .muted {{ color:var(--muted); }}
    .pill {{ display:inline-block; padding:4px 8px; background:var(--accent-soft); color:#cfe1ff; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }}
    .pill.gray {{ background:#273146; color:#d7e2ff; }}
    button {{ padding:8px 12px; border-radius:10px; border:1px solid #5f7299; background:#1b2740; color:#fff; cursor:pointer; }}
    button.primary {{ background:var(--accent); color:#07101c; border-color:var(--accent); font-weight:700; }}
    input[type=text], input[type=number], select {{ width:100%; padding:10px 12px; border:1px solid var(--border); border-radius:10px; background:#0f1829; color:#fff; }}
    label {{ display:block; margin-bottom:8px; font-size:14px; }}
    .form-row {{ display:grid; grid-template-columns:repeat(12, 1fr); gap:12px; margin-bottom:10px; }}
    .col-3 {{ grid-column:span 3; }} .col-4 {{ grid-column:span 4; }} .col-6 {{ grid-column:span 6; }} .col-12 {{ grid-column:span 12; }}
    .actions {{ display:flex; gap:8px; flex-wrap:wrap; }}
    code, pre {{ background:#0b1322; border-radius:10px; }}
    code {{ padding:2px 6px; }}
    pre {{ padding:12px; overflow:auto; white-space:pre-wrap; }}
    ul.clean {{ list-style:none; padding:0; margin:0; }}
    ul.clean li {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.08); }}
    .stats {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; }}
    .stat {{ background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:16px; padding:14px; }}
    .stat .num {{ font-size:28px; font-weight:700; }}
    .small {{ font-size:12px; }}
    .diagram-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:18px; background:#07101c; padding:14px; }}
    .diagram-grid {{ display:grid; grid-template-columns:1fr 340px; gap:16px; }}
    .diagram-side {{ position:sticky; top:12px; align-self:start; }}
    .arch-node {{ cursor:pointer; transition:transform .12s ease, filter .12s ease; }}
    .arch-node:hover {{ transform:translateY(-2px); filter:brightness(1.08); }}
    .arch-box {{ fill:#121b2e; stroke:#3c5179; stroke-width:1.5; rx:14; ry:14; }}
    .arch-node.entry .arch-box {{ fill:#162642; stroke:#5f88d9; }}
    .arch-node.pipeline .arch-box {{ fill:#1a2437; stroke:#4b6a9c; }}
    .arch-node.retrieval .arch-box {{ fill:#182b33; stroke:#52a9a5; }}
    .arch-node.ops .arch-box {{ fill:#2a2334; stroke:#9073d7; }}
    .arch-title {{ fill:#f1f5ff; font-size:14px; font-weight:700; }}
    .arch-sub {{ fill:#adc0e5; font-size:11px; }}
    .arch-edge {{ stroke:#5a6f98; stroke-width:2; fill:none; opacity:0.9; marker-end:url(#arrow); }}
    .detail-list li {{ margin-bottom:6px; }}
    @media (max-width: 980px) {{ .span-8,.span-6,.span-4,.col-6,.col-4,.col-3 {{ grid-column:span 12; }} .stats {{ grid-template-columns:1fr 1fr; }} .diagram-grid {{ grid-template-columns:1fr; }} .diagram-side {{ position:static; }} }}
  </style>
</head>
<body>
  <div class='wrap'>
    {nav_bar(active)}
    <div class='hero'>
      <h1>{safe_html(title)}</h1>
      <p>Manage the memory pipeline, agents, heartbeat rules, tracked files, and project state from one place.</p>
    </div>
    {body}
    <div class='card span-12' style='margin-top:16px;'>
      <h2>Last command output</h2>
      <pre>{escaped_output}</pre>
    </div>
  </div>
</body>
</html>
"""


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
        out.append({"name": p.name, "path": str(p), "has_current": current_file.exists(), "milestones": len(milestone_files), "daily_count": daily_count})
    return out


def overview_body(config: dict[str, Any]) -> str:
    agents = config.get("agents", [])
    projects = project_summaries()
    tracked = config.get("tracked_paths", [])
    heartbeat = config.get("heartbeat", {})
    return f"""
    <div class='stats'>
      <div class='stat'><div class='muted small'>Agents</div><div class='num'>{len(agents)}</div></div>
      <div class='stat'><div class='muted small'>In pipeline</div><div class='num'>{sum(1 for a in agents if a.get('in_pipeline'))}</div></div>
      <div class='stat'><div class='muted small'>Tracked paths</div><div class='num'>{len(tracked)}</div></div>
      <div class='stat'><div class='muted small'>Projects</div><div class='num'>{len(projects)}</div></div>
    </div>
    <div class='grid' style='margin-top:16px;'>
      <div class='card span-8'>
        <h2>Quick status</h2>
        <p><span class='pill'>Heartbeat {'on' if heartbeat.get('heartbeat_enabled') else 'off'}</span><span class='pill gray'>mode: {safe_html(heartbeat.get('heartbeat_mode', 'standard'))}</span><span class='pill gray'>interval: {safe_html(str(heartbeat.get('heartbeat_interval_seconds', 300)))}s</span></p>
        <p class='muted'>Use the tabs above to change agents, heartbeat behavior, tracked files, projects, and architecture view.</p>
      </div>
      <div class='card span-4'>
        <h2>Shortcuts</h2>
        <div class='actions'>
          <a class='nav-link' href='/agents'>Manage agents</a>
          <a class='nav-link' href='/heartbeat'>Heartbeat settings</a>
          <a class='nav-link' href='/projects'>Project tracker</a>
          <a class='nav-link' href='/architecture'>Pipeline map</a>
        </div>
      </div>
      <div class='card span-6'>
        <h2>Agent roles</h2>
        <ul class='clean'>{''.join(f"<li><strong>{safe_html(a['name'])}</strong> <span class='pill gray'>{safe_html(a.get('role','custom'))}</span> <span class='muted'>shortcut: {safe_html(a.get('shortcut',''))}</span></li>" for a in agents) or '<li>No agents configured.</li>'}</ul>
      </div>
      <div class='card span-6'>
        <h2>Projects</h2>
        <ul class='clean'>{''.join(f"<li><strong>{safe_html(p['name'])}</strong> <span class='muted'>{p['milestones']} milestones, {p['daily_count']} daily files</span></li>" for p in projects) or '<li>No projects yet.</li>'}</ul>
      </div>
    </div>
    """


def agents_body(config: dict[str, Any]) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td><strong>{safe_html(a['name'])}</strong><div class='small muted'>shortcut: {safe_html(a.get('shortcut',''))}</div></td>
          <td><span class='pill gray'>{safe_html(a.get('role','custom'))}</span></td>
          <td>{'yes' if a.get('enabled') else 'no'}</td>
          <td>{'yes' if a.get('in_pipeline') else 'no'}</td>
          <td>{'yes' if a.get('heartbeat_enabled') else 'no'}</td>
          <td>{safe_html(a.get('heartbeat_mode','standard'))}</td>
          <td><div class='actions'>
            <form method='post' action='/agents/toggle'><input type='hidden' name='name' value='{safe_html(a['name'])}' /><input type='hidden' name='field' value='enabled' /><button type='submit'>Toggle enabled</button></form>
            <form method='post' action='/agents/toggle'><input type='hidden' name='name' value='{safe_html(a['name'])}' /><input type='hidden' name='field' value='in_pipeline' /><button type='submit'>Toggle pipeline</button></form>
            <form method='post' action='/agents/toggle'><input type='hidden' name='name' value='{safe_html(a['name'])}' /><input type='hidden' name='field' value='heartbeat_enabled' /><button type='submit'>Toggle heartbeat</button></form>
          </div></td>
        </tr>
        """
        for a in config.get("agents", [])
    )
    role_options = "".join(f"<option value='{r}'>{r}</option>" for r in ROLE_OPTIONS)
    hb_options = "".join(f"<option value='{m}'>{m}</option>" for m in HEARTBEAT_MODES)
    return f"""
    <div class='grid'>
      <div class='card span-8'>
        <h2>Agent registry</h2>
        <table><thead><tr><th>Name</th><th>Role</th><th>Enabled</th><th>In pipeline</th><th>Heartbeat</th><th>HB mode</th><th>Actions</th></tr></thead><tbody>{rows or '<tr><td colspan="7">No agents configured.</td></tr>'}</tbody></table>
      </div>
      <div class='card span-4'>
        <h2>Add agent</h2>
        <form method='post' action='/agents/add'>
          <label>Name<input type='text' name='name' required></label>
          <label>Role<select name='role'>{role_options}</select></label>
          <label>Shortcut<input type='text' name='shortcut' placeholder='build, review, mem'></label>
          <label>Heartbeat mode<select name='heartbeat_mode'>{hb_options}</select></label>
          <button class='primary' type='submit'>Add agent</button>
        </form>
      </div>
      <div class='card span-12'>
        <h2>Update agent role / shortcut</h2>
        <form method='post' action='/agents/update'>
          <div class='form-row'>
            <div class='col-3'><label>Agent<select name='name'>{''.join(f"<option value='{safe_html(a['name'])}'>{safe_html(a['name'])}</option>" for a in config.get('agents', []))}</select></label></div>
            <div class='col-3'><label>Role<select name='role'>{role_options}</select></label></div>
            <div class='col-3'><label>Shortcut<input type='text' name='shortcut'></label></div>
            <div class='col-3'><label>Heartbeat mode<select name='heartbeat_mode'>{hb_options}</select></label></div>
          </div>
          <button class='primary' type='submit'>Save agent settings</button>
        </form>
      </div>
    </div>
    """


def heartbeat_body(config: dict[str, Any]) -> str:
    hb = config.get("heartbeat", {})
    hb_options = "".join(f"<option value='{m}' {'selected' if hb.get('heartbeat_mode') == m else ''}>{m}</option>" for m in HEARTBEAT_MODES)
    tracked_rows = "".join(f"<li><code>{safe_html(p)}</code></li>" for p in config.get("tracked_paths", [])) or "<li>None</li>"
    return f"""
    <div class='grid'>
      <div class='card span-6'>
        <h2>Global heartbeat settings</h2>
        <form method='post' action='/heartbeat/save'>
          <label><input type='checkbox' name='heartbeat_enabled' {'checked' if hb.get('heartbeat_enabled') else ''}/> heartbeat enabled</label>
          <label>Mode<select name='heartbeat_mode'>{hb_options}</select></label>
          <label>Interval seconds<input type='number' name='heartbeat_interval_seconds' value='{safe_html(str(hb.get('heartbeat_interval_seconds', 300)))}'></label>
          <label>Heartbeat file<input type='text' name='heartbeat_file' value='{safe_html(hb.get('heartbeat_file', ''))}'></label>
          <label>Heartbeat agent file<input type='text' name='heartbeat_agent_file' value='{safe_html(hb.get('heartbeat_agent_file', ''))}'></label>
          <button class='primary' type='submit'>Save heartbeat settings</button>
        </form>
      </div>
      <div class='card span-6'>
        <h2>Tracked paths</h2>
        <ul class='clean'>{tracked_rows}</ul>
        <form method='post' action='/tracked/add'>
          <label>Add tracked path<input type='text' name='path' placeholder='/absolute/or/workspace/relative/path' required></label>
          <button type='submit'>Add tracked path</button>
        </form>
      </div>
    </div>
    """


def projects_body() -> str:
    projects = project_summaries()
    rows = "".join(
        f"<tr><td><a href='/projects/{safe_html(p['name'])}'>{safe_html(p['name'])}</a></td><td><code>{safe_html(p['path'])}</code></td><td>{'yes' if p['has_current'] else 'no'}</td><td>{p['milestones']}</td><td>{p['daily_count']}</td></tr>"
        for p in projects
    )
    return f"""
    <div class='grid'>
      <div class='card span-8'>
        <h2>Project tracker</h2>
        <table><thead><tr><th>Project</th><th>Path</th><th>current.md</th><th>Milestones</th><th>Daily files</th></tr></thead><tbody>{rows or '<tr><td colspan="5">No projects yet.</td></tr>'}</tbody></table>
      </div>
      <div class='card span-4'>
        <h2>Add new project</h2>
        <form method='post' action='/projects/add'>
          <label>Project name<input type='text' name='name' placeholder='game-ai-combat' required></label>
          <button class='primary' type='submit'>Create project</button>
        </form>
        <p class='muted small'>This creates the project folder, <code>current.md</code>, and <code>daily/</code>.</p>
      </div>
    </div>
    """


def project_detail_body(project_name: str) -> str:
    p = PROJECTS_DIR / project_name
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    current_file = p / "current.md"
    milestone_files = sorted(p.glob("milestone-*.md"))
    daily_dir = p / "daily"
    daily_files = sorted(daily_dir.glob("*.md")) if daily_dir.exists() else []
    def file_li(fp: Path) -> str:
        return f"<li><code>{safe_html(str(fp))}</code></li>"
    return f"""
    <div class='grid'>
      <div class='card span-6'>
        <h2>{safe_html(project_name)}</h2>
        <p class='muted'>Project path: <code>{safe_html(str(p))}</code></p>
        <p><span class='pill gray'>current.md: {'yes' if current_file.exists() else 'no'}</span><span class='pill gray'>milestones: {len(milestone_files)}</span><span class='pill gray'>daily files: {len(daily_files)}</span></p>
      </div>
      <div class='card span-6'>
        <h2>Quick add files</h2>
        <form method='post' action='/projects/file/add'>
          <input type='hidden' name='project_name' value='{safe_html(project_name)}'>
          <label>Relative file path<input type='text' name='relative_path' placeholder='milestone-01.md or notes/design.md' required></label>
          <button type='submit'>Create file</button>
        </form>
      </div>
      <div class='card span-4'><h3>Current</h3><ul class='clean'>{file_li(current_file) if current_file.exists() else '<li>Missing</li>'}</ul></div>
      <div class='card span-4'><h3>Milestones</h3><ul class='clean'>{''.join(file_li(fp) for fp in milestone_files) or '<li>None</li>'}</ul></div>
      <div class='card span-4'><h3>Daily</h3><ul class='clean'>{''.join(file_li(fp) for fp in daily_files[-15:]) or '<li>None</li>'}</ul></div>
    </div>
    """


def architecture_body() -> str:
    node_lookup = {n["id"]: n for n in ARCHITECTURE_NODES}
    svg_edges = []
    for src, dst in ARCHITECTURE_EDGES:
        a = node_lookup[src]
        b = node_lookup[dst]
        x1 = a["x"] + a["w"] / 2
        y1 = a["y"] + a["h"]
        x2 = b["x"] + b["w"] / 2
        y2 = b["y"]
        midy = (y1 + y2) / 2
        path = f"M {x1} {y1} C {x1} {midy}, {x2} {midy}, {x2} {y2}"
        svg_edges.append(f"<path class='arch-edge' d='{path}'></path>")

    svg_nodes = []
    details = []
    for idx, n in enumerate(ARCHITECTURE_NODES):
        scripts = "".join(f"<li><code>{safe_html(s)}</code></li>" for s in n["scripts"])
        bullets = "".join(f"<li>{safe_html(d)}</li>" for d in n["details"])
        svg_nodes.append(
            f"""
            <g class='arch-node {safe_html(n['group'])}' data-target='detail-{safe_html(n['id'])}'>
              <rect class='arch-box' x='{n['x']}' y='{n['y']}' width='{n['w']}' height='{n['h']}' rx='16' ry='16'></rect>
              <text class='arch-title' x='{n['x'] + 16}' y='{n['y'] + 28}'>{safe_html(n['title'])}</text>
              <text class='arch-sub' x='{n['x'] + 16}' y='{n['y'] + 50}'>{safe_html(n['subtitle'])}</text>
            </g>
            """
        )
        details.append(
            f"""
            <div id='detail-{safe_html(n['id'])}' class='card detail-panel' style='display:{'block' if idx == 0 else 'none'};'>
              <h3>{safe_html(n['title'])}</h3>
              <p class='muted'>{safe_html(n['subtitle'])}</p>
              <h4>Scripts / components</h4>
              <ul class='detail-list'>{scripts}</ul>
              <h4>What it does</h4>
              <ul class='detail-list'>{bullets}</ul>
            </div>
            """
        )

    return f"""
    <div class='card span-12'>
      <h2>Full memory system map</h2>
      <p class='muted'>Each block is clickable. The right-side panel shows the scripts and responsibilities for that stage of the pipeline.</p>
      <div class='diagram-grid'>
        <div class='diagram-wrap'>
          <svg width='1100' height='700' viewBox='0 0 1100 700' xmlns='http://www.w3.org/2000/svg'>
            <defs>
              <marker id='arrow' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto' markerUnits='strokeWidth'>
                <path d='M0,0 L0,6 L9,3 z' fill='#6c82ac'></path>
              </marker>
            </defs>
            {''.join(svg_edges)}
            {''.join(svg_nodes)}
          </svg>
        </div>
        <div class='diagram-side'>
          {''.join(details)}
        </div>
      </div>
      <script>
        document.querySelectorAll('.arch-node').forEach((node) => {{
          node.addEventListener('click', () => {{
            const target = node.getAttribute('data-target');
            document.querySelectorAll('.detail-panel').forEach((panel) => panel.style.display = 'none');
            const el = document.getElementById(target);
            if (el) el.style.display = 'block';
          }});
        }});
      </script>
    </div>
    """


def commands_body(config: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{safe_html(name)}</td><td><code>{safe_html(cmd)}</code></td><td><form method='post' action='/run'><input type='hidden' name='command_name' value='{safe_html(name)}'><button type='submit'>Run</button></form></td></tr>"
        for name, cmd in config.get("commands", {}).items()
    )
    return f"<div class='card span-12'><h2>Commands</h2><table><thead><tr><th>Name</th><th>Command</th><th>Run</th></tr></thead><tbody>{rows}</tbody></table></div>"


LAST_OUTPUT = ""


@app.get("/", response_class=HTMLResponse)
def overview() -> str:
    config = load_config()
    return page_shell("OpenClaw Control Panel", "overview", overview_body(config), LAST_OUTPUT)


@app.get("/agents", response_class=HTMLResponse)
def agents_page() -> str:
    config = load_config()
    return page_shell("Agent Registry", "agents", agents_body(config), LAST_OUTPUT)


@app.get("/heartbeat", response_class=HTMLResponse)
def heartbeat_page() -> str:
    config = load_config()
    return page_shell("Heartbeat Settings", "heartbeat", heartbeat_body(config), LAST_OUTPUT)


@app.get("/projects", response_class=HTMLResponse)
def projects_page() -> str:
    return page_shell("Project Tracker", "projects", projects_body(), LAST_OUTPUT)


@app.get("/projects/{project_name}", response_class=HTMLResponse)
def project_detail(project_name: str) -> str:
    return page_shell(f"Project: {project_name}", "projects", project_detail_body(project_name), LAST_OUTPUT)


@app.get("/architecture", response_class=HTMLResponse)
def architecture_page() -> str:
    return page_shell("Architecture Explorer", "architecture", architecture_body(), LAST_OUTPUT)


@app.get("/commands", response_class=HTMLResponse)
def commands_page() -> str:
    config = load_config()
    return page_shell("Commands", "commands", commands_body(config), LAST_OUTPUT)


@app.post("/agents/add")
def add_agent(name: str = Form(...), role: str = Form("custom"), shortcut: str = Form(""), heartbeat_mode: str = Form("standard")):
    config = load_config()
    agents = config.setdefault("agents", [])
    clean_name = name.strip()
    if any(a["name"] == clean_name for a in agents):
        return RedirectResponse(url="/agents", status_code=303)
    agents.append({"name": clean_name, "enabled": True, "in_pipeline": True, "role": role, "heartbeat_enabled": heartbeat_mode != "off", "heartbeat_mode": heartbeat_mode, "shortcut": shortcut.strip() or clean_name})
    save_config(config)
    return RedirectResponse(url="/agents", status_code=303)


@app.post("/agents/toggle")
def toggle_agent(name: str = Form(...), field: str = Form(...)):
    config = load_config()
    for agent in config.get("agents", []):
        if agent.get("name") == name:
            if field not in {"enabled", "in_pipeline", "heartbeat_enabled"}:
                raise HTTPException(status_code=400, detail="Bad field")
            agent[field] = not bool(agent.get(field))
            save_config(config)
            return RedirectResponse(url="/agents", status_code=303)
    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/agents/update")
def update_agent(name: str = Form(...), role: str = Form(...), shortcut: str = Form(""), heartbeat_mode: str = Form(...)):
    config = load_config()
    for agent in config.get("agents", []):
        if agent.get("name") == name:
            agent["role"] = role
            agent["shortcut"] = shortcut.strip() or name
            agent["heartbeat_mode"] = heartbeat_mode
            agent["heartbeat_enabled"] = heartbeat_mode != "off"
            save_config(config)
            return RedirectResponse(url="/agents", status_code=303)
    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/tracked/add")
def add_tracked_path(path: str = Form(...)):
    config = load_config()
    tracked = config.setdefault("tracked_paths", [])
    p = path.strip()
    if p and p not in tracked:
        tracked.append(p)
        save_config(config)
    return RedirectResponse(url="/heartbeat", status_code=303)


@app.post("/tracked/remove")
def remove_tracked_path(path: str = Form(...)):
    config = load_config()
    config["tracked_paths"] = [p for p in config.get("tracked_paths", []) if p != path]
    save_config(config)
    return RedirectResponse(url="/heartbeat", status_code=303)


@app.post("/heartbeat/save")
def save_heartbeat(
    heartbeat_enabled: str | None = Form(None),
    heartbeat_mode: str = Form(...),
    heartbeat_interval_seconds: int = Form(...),
    heartbeat_file: str = Form(...),
    heartbeat_agent_file: str = Form(...),
):
    config = load_config()
    config["heartbeat"] = {
        "heartbeat_enabled": heartbeat_enabled is not None,
        "heartbeat_mode": heartbeat_mode,
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "heartbeat_file": heartbeat_file,
        "heartbeat_agent_file": heartbeat_agent_file,
    }
    save_config(config)
    return RedirectResponse(url="/heartbeat", status_code=303)


@app.post("/projects/add")
def add_project(name: str = Form(...)):
    project_name = name.strip()
    p = PROJECTS_DIR / project_name
    p.mkdir(parents=True, exist_ok=True)
    (p / "daily").mkdir(parents=True, exist_ok=True)
    (p / "current.md").touch(exist_ok=True)
    return RedirectResponse(url=f"/projects/{project_name}", status_code=303)


@app.post("/projects/file/add")
def add_project_file(project_name: str = Form(...), relative_path: str = Form(...)):
    p = PROJECTS_DIR / project_name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    rel = Path(relative_path.strip())
    target = p / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=True)
    return RedirectResponse(url=f"/projects/{project_name}", status_code=303)


@app.post("/run")
def run_command(command_name: str = Form(...)):
    global LAST_OUTPUT
    config = load_config()
    cmd = config.get("commands", {}).get(command_name)
    if not cmd:
        raise HTTPException(status_code=404, detail="Unknown command")
    proc = subprocess.run(cmd, shell=True, cwd=str(WORKSPACE), capture_output=True, text=True, env=os.environ.copy())
    LAST_OUTPUT = f"$ {cmd}\n\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nRETURN CODE: {proc.returncode}"
    return RedirectResponse(url="/commands", status_code=303)


def pick_port() -> int:
    preferred = [int(os.environ.get("OPENCLAW_CONTROL_PANEL_PORT", "8787")), 8788, 8790, 8791, 8800]
    seen = set()
    for port in preferred:
        if port in seen:
            continue
        seen.add(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 8788


if __name__ == "__main__":
    import uvicorn

    port = pick_port()
    print(f"OpenClaw Control Panel running on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
