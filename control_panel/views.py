"""FastAPI routes for the OpenClaw Memory System Control Panel."""
from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import CONTROL_PANEL_DIR
from services import (
    agent_list,
    docker_status,
    git_info,
    last_benchmark_result,
    last_test_result,
    memory_search,
    memory_stats,
    neo4j_stats,
    qdrant_stats,
    recent_memory_items,
    run_benchmark,
    run_tests,
    service_status,
    system_health,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(CONTROL_PANEL_DIR / "templates"))


def _render(request: Request, page: str, title: str, **extra):
    ctx = {"request": request, "page": page, "title": title, **extra}
    return templates.TemplateResponse(request, "base.html", context=ctx)


# ── Dashboard ──────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return _render(
        request, "dashboard", "Dashboard",
        health=system_health(),
        stats=memory_stats(),
        containers=docker_status(),
        git=git_info(),
        tests=last_test_result(),
        services=service_status(),
    )


# ── Agents ─────────────────────────────────────────────────────────────────

@router.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request):
    return _render(request, "agents", "Agents", agents=agent_list())


# ── Memory ─────────────────────────────────────────────────────────────────

@router.get("/memory", response_class=HTMLResponse)
def memory_page(request: Request):
    return _render(
        request, "memory", "Memory",
        stats=memory_stats(),
        recent=recent_memory_items(15),
        search_results=[],
        search_query="",
    )


@router.post("/memory/search", response_class=HTMLResponse)
def memory_search_action(request: Request, query: str = Form(...)):
    return _render(
        request, "memory", "Memory",
        stats=memory_stats(),
        recent=recent_memory_items(15),
        search_results=memory_search(query),
        search_query=query,
    )


# ── Code Graph ─────────────────────────────────────────────────────────────

@router.get("/code-graph", response_class=HTMLResponse)
def code_graph_page(request: Request):
    return _render(
        request, "code-graph", "Code Graph",
        neo4j=neo4j_stats(),
        qdrant=qdrant_stats(),
    )


# ── Tests ──────────────────────────────────────────────────────────────────

@router.get("/tests", response_class=HTMLResponse)
def tests_page(request: Request):
    return _render(request, "tests", "Tests", tests=last_test_result())


@router.post("/tests/run", response_class=HTMLResponse)
def tests_run(request: Request):
    result = run_tests()
    return _render(request, "tests", "Tests", tests=result)


# ── Benchmarks ─────────────────────────────────────────────────────────────

@router.get("/benchmarks", response_class=HTMLResponse)
def benchmarks_page(request: Request):
    return _render(request, "benchmarks", "Benchmarks", benchmark=last_benchmark_result())


@router.post("/benchmarks/run", response_class=HTMLResponse)
def benchmarks_run(request: Request):
    result = run_benchmark()
    return _render(request, "benchmarks", "Benchmarks", benchmark=result)


# ── Services ───────────────────────────────────────────────────────────────

@router.get("/services", response_class=HTMLResponse)
def services_page(request: Request):
    return _render(
        request, "services", "Services",
        containers=docker_status(),
        services=service_status(),
    )


# ── Git ────────────────────────────────────────────────────────────────────

@router.get("/architecture", response_class=HTMLResponse)
def architecture_page(request: Request):
    return _render(request, "architecture", "System Architecture")


@router.get("/git", response_class=HTMLResponse)
def git_page(request: Request):
    return _render(request, "git", "Repository", git=git_info())
