"""
Path resolution for project-scoped memory storage.
Maps project IDs to filesystem locations.
"""
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"
PROJECTS_DIR = MEMORY_DIR / "projects"


def normalize_project_id(project_id: str) -> str:
    s = (project_id or "").strip().lower().replace(" ", "-").replace("_", "-")
    while "--" in s:
        s = s.replace("--", "-")
    s = s.strip("-")
    if not s:
        raise ValueError("project_id cannot be empty")
    return s


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_project_dir(project_id: str) -> Path:
    pid = normalize_project_id(project_id)
    return PROJECTS_DIR / pid


def ensure_project_dirs(project_id: str):
    project_dir = get_project_dir(project_id)
    daily_dir = project_dir / "daily"
    project_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    return project_dir, daily_dir


def get_project_current_file(project_id: str) -> Path:
    project_dir, _ = ensure_project_dirs(project_id)
    return project_dir / "current.md"


def get_project_daily_file(project_id: str, day: str | None = None) -> Path:
    _, daily_dir = ensure_project_dirs(project_id)
    if day is None:
        day = today_utc()
    return daily_dir / f"{day}.md"


def get_project_snapshot_file(project_id: str) -> Path:
    project_dir, _ = ensure_project_dirs(project_id)
    return project_dir / "latest-workspace-snapshot.json"


def get_project_summary_file(project_id: str) -> Path:
    project_dir, _ = ensure_project_dirs(project_id)
    return project_dir / "latest-workspace-summary.json"


def get_latest_project_daily_file(project_id: str):
    _, daily_dir = ensure_project_dirs(project_id)
    files = sorted(daily_dir.glob("*.md"))
    if not files:
        return None
    return files[-1]
