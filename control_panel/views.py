from __future__ import annotations

import sys
from live_status import get_live_status
import os
import subprocess
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Form

from config import DEFAULT_PORT, HEARTBEAT_MODES, PROJECTS_DIR, ROLE_OPTIONS, WORKSPACE, SCRIPTS_DIR
from data_architecture import ARCHITECTURE_CROSS_EDGES, ARCHITECTURE_TREES

from datetime import datetime, timezone

from checkpoint_db import list_recent_checkpoints, latest_checkpoint

from services import (
    build_mermaid_diagram,
    find_node,
    get_all_script_files,
    load_config,
    memory_status_counts,
    orchestrator_operator_overview,
    project_summaries,
    save_config,
    script_inventory_groups,
)

router = APIRouter()
templates = Jinja2Templates(directory=str((WORKSPACE / ".memory-index" / "control_panel" / "templates")))

LAST_OUTPUT = ""
LAST_STATUS = "idle"
LAST_RUN_AT = ""

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_INDEX = WORKSPACE / ".memory-index"
SCRIPTS_DIR = MEMORY_INDEX / "scripts"
PY = str(Path.home() / ".openclaw" / "venvs" / "memory-db" / "bin" / "python")

def services_page_data() -> dict:
    watcher = run_json_script("tracked_path_watcher_status.py")
    heartbeat = run_json_script("heartbeat_worker_status.py")
    search = run_json_script("search_service_status.py")
    tuning = read_heartbeat_tuning_state()
    return {
        "tracked_path_watcher_status": watcher,
        "heartbeat_worker_status": heartbeat,
        "search_service_status": search,
        "heartbeat_tuning_state": tuning,
    }

def build_architecture_detail_map() -> dict[str, dict]:
    all_nodes = {}
    inbound_map: dict[str, list[dict]] = {}
    outbound_map: dict[str, list[dict]] = {}

    for tree in ARCHITECTURE_TREES:
        for node in tree["nodes"]:
            merged = dict(node)
            merged["tree_title"] = tree["title"]
            merged["tree_color"] = tree["color"]
            all_nodes[node["id"]] = merged
            inbound_map[node["id"]] = []
            outbound_map[node["id"]] = []

    def add_edge(src: str, dst: str, label: str, kind: str):
        if src in all_nodes and dst in all_nodes:
            outbound_map[src].append(
                {"id": dst, "title": all_nodes[dst]["title"], "label": label, "kind": kind}
            )
            inbound_map[dst].append(
                {"id": src, "title": all_nodes[src]["title"], "label": label, "kind": kind}
            )

    for tree in ARCHITECTURE_TREES:
        for src, dst, label in tree["edges"]:
            add_edge(src, dst, label, "tree")

    for src, dst, label in ARCHITECTURE_CROSS_EDGES:
        add_edge(src, dst, label, "cross")

    detail_map = {}
    for node_id, node in all_nodes.items():
        detail_map[node_id] = {
            "id": node_id,
            "title": node["title"],
            "subtitle": node["subtitle"],
            "tree_title": node["tree_title"],
            "tree_color": node["tree_color"],
            "kind": node["kind"],
            "scripts": node.get("scripts", []),
            "purpose": node.get("details", []),
            "inputs": [f"{x['title']} ({x['label']})" for x in inbound_map[node_id]],
            "outputs": [f"{x['title']} ({x['label']})" for x in outbound_map[node_id]],
            "connected_from": inbound_map[node_id],
            "connected_to": outbound_map[node_id],
        }

    return detail_map


def run_json_script(script_name: str) -> dict:
    script = SCRIPTS_DIR / script_name
    env = os.environ.copy()
    env.setdefault("OPENCLAW_MEMORY_DB_DSN", f"dbname=openclaw_memory user={os.environ.get('USER', 'joseph-ding')}")
    proc = subprocess.run(
        [PY, str(script)],
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
        env=env,
    )
    text = (proc.stdout or proc.stderr or "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text, "returncode": proc.returncode}

def read_heartbeat_tuning_state() -> dict:
    code = (
        "import sys, json; from pathlib import Path; "
        "sys.path.insert(0, str(Path.home()/'.openclaw'/'workspace'/'.memory-index'/'scripts')); "
        "from checkpoint_db import get_state, close_pool; "
        "print(json.dumps({'heartbeat_tuning': get_state('heartbeat_tuning', {}), 'effective_interval': get_state('heartbeat_effective_interval_seconds', None)})); "
        "close_pool()"
    )
    proc = subprocess.run(
        [PY, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
        env=os.environ.copy(),
    )
    try:
        return json.loads((proc.stdout or "{}").strip())
    except Exception:
        return {"heartbeat_tuning": {}, "effective_interval": None}

def _run_json_script(script_name: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / script_name),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        return {
            "status": "error",
            "stderr": proc.stderr,
            "stdout": proc.stdout,
            "returncode": proc.returncode,
        }
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {
            "status": "invalid_json",
            "stdout": proc.stdout,
        }


def heartbeat_worker_status_data() -> dict[str, Any]:
    return _run_json_script("heartbeat_worker_status.py")


def tracked_path_watcher_status_data() -> dict[str, Any]:
    return _run_json_script("tracked_path_watcher_status.py")


def checkpoint_page_data(limit: int = 20) -> dict[str, Any]:
    latest = latest_checkpoint()
    recent = list_recent_checkpoints(limit=limit)
    return {
        "latest_checkpoint": latest,
        "recent_checkpoints": recent,
    }

def render(
    request: Request,
    title: str,
    page: str,
    extra_context: dict | None = None,
    **kwargs,
):
    cfg = load_config()

    context = {
        "heartbeat_worker_status": heartbeat_worker_status_data(),
        "tracked_path_watcher_status": tracked_path_watcher_status_data(),
        **checkpoint_page_data(),
        "request": request,
        "title": title,
        "page": page,
        "config": cfg,
        "live_status": get_live_status(),
        "default_port": DEFAULT_PORT,
        "role_options": ROLE_OPTIONS,
        "heartbeat_modes": HEARTBEAT_MODES,
        "projects": project_summaries(),
        "db_counts": memory_status_counts(),
        "operator_overview": orchestrator_operator_overview(),
        "last_output": LAST_OUTPUT or "No command run yet.",
    }

    if extra_context:
        context.update(extra_context)
    if kwargs:
        context.update(kwargs)

    return templates.TemplateResponse("base.html", context)

def run_stack_command(command: str) -> str:
    env = os.environ.copy()
    env.setdefault("OPENCLAW_MEMORY_DB_DSN", f"dbname=openclaw_memory user={os.environ.get('USER', 'joseph-ding')}")
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(SCRIPTS_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return out.strip() or f"returncode={proc.returncode}"


def run_stack_action(action: str) -> str:
    actions = {
        "ensure_memory_stack": [
            f"{PY} {SCRIPTS_DIR / 'ensure_tracked_path_watcher.py'}",
            f"{PY} {SCRIPTS_DIR / 'ensure_heartbeat_worker.py'}",
            f"{PY} {SCRIPTS_DIR / 'start_search_service.py'}",
        ],
        "stop_memory_stack": [
            f"{PY} {SCRIPTS_DIR / 'stop_tracked_path_watcher.py'}",
            f"{PY} {SCRIPTS_DIR / 'stop_heartbeat_worker.py'}",
            f"{PY} {SCRIPTS_DIR / 'stop_search_service.py'}",
        ],
    }
    cmds = actions.get(action, [])
    parts = []
    for cmd in cmds:
        parts.append(f"$ {cmd}\n{run_stack_command(cmd)}")
    return "\n\n".join(parts)


def read_heartbeat_tuning_state() -> dict:
    code = (
        "import sys, json; from pathlib import Path; "
        "sys.path.insert(0, str(Path.home()/'.openclaw'/'workspace'/'.memory-index'/'scripts')); "
        "from checkpoint_db import get_state, close_pool; "
        "print(json.dumps({'heartbeat_tuning': get_state('heartbeat_tuning', {}), 'effective_interval': get_state('heartbeat_effective_interval_seconds', None)})); "
        "close_pool()"
    )
    proc = subprocess.run(
        [PY, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
        env=os.environ.copy(),
    )
    try:
        return json.loads((proc.stdout or "{}").strip())
    except Exception:
        return {"heartbeat_tuning": {}, "effective_interval": None}

@router.get("/")
def overview(request: Request):
    return render(request, "OpenClaw Control Panel", "overview")


@router.get("/agents")
def agents_page(request: Request):
    return render(request, "Agent Registry", "agents")



@router.get("/projects")
def projects_page(request: Request):
    return render(request, "Project Tracker", "projects")


@router.get("/projects/{project_name}")
def project_detail(request: Request, project_name: str):
    p = PROJECTS_DIR / project_name
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    current_file = p / "current.md"
    milestone_files = sorted(p.glob("milestone-*.md"))
    daily_dir = p / "daily"
    daily_files = sorted(daily_dir.glob("*.md")) if daily_dir.exists() else []

    return render(
        request,
        f"Project: {project_name}",
        "project_detail",
        project_name=project_name,
        project_path=str(p),
        has_current=current_file.exists(),
        milestone_files=[str(x) for x in milestone_files],
        daily_files=[str(x) for x in daily_files[-15:]],
        current_file=str(current_file),
    )


@router.get("/architecture")
def architecture_page(request: Request):
    detail_map = build_architecture_detail_map()
    first_node_id = next(iter(detail_map)) if detail_map else None
    return render(
        request,
        "Memory System Graph",
        "architecture",
        mermaid_graph=build_mermaid_diagram(),
        script_groups=script_inventory_groups(),
        total_scripts=len(get_all_script_files()),
        architecture_details=detail_map,
        first_node_id=first_node_id,
    )


@router.get("/architecture/node/{node_id}")
def architecture_node_page(request: Request, node_id: str):
    try:
        tree, node = find_node(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Node not found")
    return render(request, f"Graph Node: {node['title']}", "architecture_node", tree=tree, node=node)


@router.get("/commands")
def commands_page(request: Request):
    return render(request, "Commands", "commands")


@router.post("/agents/add")
def add_agent(
    name: str = Form(...),
    role: str = Form("custom"),
    shortcut: str = Form(""),
    heartbeat_mode: str = Form("standard"),
):
    cfg = load_config()
    agents = cfg.setdefault("agents", [])
    clean_name = name.strip()
    if not any(a["name"] == clean_name for a in agents):
        agents.append(
            {
                "name": clean_name,
                "enabled": True,
                "in_pipeline": True,
                "role": role,
                "heartbeat_enabled": heartbeat_mode != "off",
                "heartbeat_mode": heartbeat_mode,
                "shortcut": shortcut.strip() or clean_name,
            }
        )
        save_config(cfg)
    return RedirectResponse(url="/agents", status_code=303)


@router.post("/agents/toggle")
def toggle_agent(name: str = Form(...), field: str = Form(...)):
    cfg = load_config()
    for agent in cfg.get("agents", []):
        if agent.get("name") == name:
            if field not in {"enabled", "in_pipeline", "heartbeat_enabled"}:
                raise HTTPException(status_code=400, detail="Bad field")
            agent[field] = not bool(agent.get(field))
            save_config(cfg)
            return RedirectResponse(url="/agents", status_code=303)
    raise HTTPException(status_code=404, detail="Agent not found")


@router.post("/agents/update")
def update_agent(
    name: str = Form(...),
    role: str = Form(...),
    shortcut: str = Form(""),
    heartbeat_mode: str = Form(...),
):
    cfg = load_config()
    for agent in cfg.get("agents", []):
        if agent.get("name") == name:
            agent["role"] = role
            agent["shortcut"] = shortcut.strip() or name
            agent["heartbeat_mode"] = heartbeat_mode
            agent["heartbeat_enabled"] = heartbeat_mode != "off"
            save_config(cfg)
            return RedirectResponse(url="/agents", status_code=303)
    raise HTTPException(status_code=404, detail="Agent not found")


@router.post("/tracked/add")
def add_tracked_path(path: str = Form(...)):
    cfg = load_config()
    tracked = cfg.setdefault("tracked_paths", [])
    p = path.strip()
    if p and p not in tracked:
        tracked.append(p)
        save_config(cfg)
    return RedirectResponse(url="/heartbeat", status_code=303)


@router.post("/heartbeat/save")
def save_heartbeat(
    heartbeat_enabled: str | None = Form(None),
    heartbeat_mode: str = Form(...),
    heartbeat_interval_seconds: int = Form(...),
    heartbeat_file: str = Form(...),
    heartbeat_agent_file: str = Form(...),
):
    cfg = load_config()
    cfg["heartbeat"] = {
        "heartbeat_enabled": heartbeat_enabled is not None,
        "heartbeat_mode": heartbeat_mode,
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "heartbeat_file": heartbeat_file,
        "heartbeat_agent_file": heartbeat_agent_file,
    }
    save_config(cfg)
    return RedirectResponse(url="/heartbeat", status_code=303)


@router.post("/projects/add")
def add_project(name: str = Form(...)):
    project_name = name.strip()
    p = PROJECTS_DIR / project_name
    p.mkdir(parents=True, exist_ok=True)
    (p / "daily").mkdir(parents=True, exist_ok=True)
    (p / "current.md").touch(exist_ok=True)
    return RedirectResponse(url=f"/projects/{project_name}", status_code=303)


@router.post("/projects/file/add")
def add_project_file(project_name: str = Form(...), relative_path: str = Form(...)):
    p = PROJECTS_DIR / project_name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    rel = Path(relative_path.strip())
    target = p / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=True)
    return RedirectResponse(url=f"/projects/{project_name}", status_code=303)


@router.post("/run")
def run_command(command_name: str = Form(...)):
    global LAST_OUTPUT, LAST_STATUS, LAST_RUN_AT

    cfg = load_config()
    cmd = cfg.get("commands", {}).get(command_name)
    if not cmd:
        raise HTTPException(status_code=404, detail="Unknown command")

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    LAST_STATUS = "success" if proc.returncode == 0 else "error"
    LAST_RUN_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    LAST_OUTPUT = (
        f"$ {cmd}\n\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}\n"
        f"RETURN CODE: {proc.returncode}"
    )

@router.post("/conflicts/demote")

def conflict_demote(item_id: str = Form(...)):
    global LAST_OUTPUT, LAST_STATUS, LAST_RUN_AT

    cmd = (
        f"{WORKSPACE / '.openclaw' / 'venvs' / 'memory-db' / 'bin' / 'python'} "
        f"{SCRIPTS_DIR / 'demote_memory_item.py'} --item-id {item_id}"
    )

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    LAST_STATUS = "success" if proc.returncode == 0 else "error"
    LAST_RUN_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    LAST_OUTPUT = (
        f"$ {cmd}\n\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}\n"
        f"RETURN CODE: {proc.returncode}"
    )

    return RedirectResponse(url="/", status_code=303)

@router.get("/search")
def search_page(request: Request):
    return render(
        request,
        "Memory Search",
        "search",
        search_results=[],
        search_query="",
    )


@router.post("/search")
def search_run(request: Request, query: str = Form(...)):
    env = os.environ.copy()
    env.setdefault("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory user=joseph-ding")
    env.setdefault("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")

    cmd = [
        str(Path.home() / ".openclaw" / "venvs" / "memory-db" / "bin" / "python"),
        str(SCRIPTS_DIR / "search_memory.py"),
        "--query",
        query,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=env,
    )

    results = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line == "NONE":
            continue
        try:
            results.append(json.loads(line))
        except Exception:
            pass

    if proc.returncode != 0:
        results = [
            {
                "score": 0.0,
                "source_type": "error",
                "path": "control_panel:search",
                "text": (proc.stderr or proc.stdout or "Search failed").strip(),
            }
        ]

    return render(
        request,
        "Memory Search",
        "search",
        search_results=results,
        search_query=query,
    )

@router.get("/checkpoints")
def checkpoints_page(request: Request):
    return render(request, "Memory Checkpoints", "checkpoints")

@router.get("/heartbeat", response_class=HTMLResponse)
def heartbeat_page(request: Request):
    return render(
        request,
        "Heartbeat Settings",
        "heartbeat",
        extra_context=services_page_data(),
    )

@router.get("/services", response_class=HTMLResponse)
def services_page(request: Request):
    return render(
        request,
        "Memory Stack Services",
        "services",
        extra_context=services_page_data(),
    )

@router.post("/stack/action")
def stack_action(action: str = Form(...)):
    output = run_stack_action(action)
    return RedirectResponse(url=f"/services?output={action}", status_code=303)


@router.post("/stack/run-script")
def stack_run_script(script_name: str = Form(...)):
    output = run_stack_command(f"{PY} {SCRIPTS_DIR / script_name}")
    return RedirectResponse(url=f"/heartbeat?output={script_name}", status_code=303)

@router.post("/conflicts/restore")
def conflict_restore(item_id: str = Form(...)):
    global LAST_OUTPUT, LAST_STATUS, LAST_RUN_AT

    cmd = (
        f"{Path.home() / '.openclaw' / 'venvs' / 'memory-db' / 'bin' / 'python'} "
        f"{SCRIPTS_DIR / 'restore_memory_item.py'} --item-id {item_id}"
    )

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    LAST_STATUS = "success" if proc.returncode == 0 else "error"
    LAST_RUN_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    LAST_OUTPUT = (
        f"$ {cmd}\n\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}\n"
        f"RETURN CODE: {proc.returncode}"
    )

    return RedirectResponse(url="/", status_code=303)


    return RedirectResponse(url="/commands", status_code=303)
