"""
Configuration for periodic memory refresh cycles —
defines intervals, scopes, and policies for background memory updates.
"""
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"

REFRESH_PROFILES = {
    "coder": {
        "interval_seconds": 300,
        "files": [
            "browser.md",
            "worker.md",
            "gateway.md",
            "learned-fixes.md",
            "memory-system.md",
            "__PROJECT_CURRENT__",
            "__PROJECT_LATEST_DAILY__",
        ],
    },
    "review": {
        "interval_seconds": 420,
        "files": [
            "learned-fixes.md",
            "memory-system.md",
            "preferences.md",
        ],
    },
    "art": {
        "interval_seconds": 900,
        "files": [
            "preferences.md",
            "memory-system.md",
        ],
    },
    "default": {
        "interval_seconds": 600,
        "files": [
            "preferences.md",
            "learned-fixes.md",
            "memory-system.md",
        ],
    },
}

# Optional explicit overrides for special cases
AGENT_CLASSES = {
    "main": "coder",
    "builder": "coder",
    "reviewer": "review",
    "tester": "coder",
    "artist": "art",
    "art": "art",
}

def infer_agent_class(agent_name: str) -> str:
    name = (agent_name or "").strip().lower()

    coder_keywords = [
        "builder", "coder", "dev", "engineer",
        "tester", "implementer", "programmer",
    ]
    review_keywords = [
        "review", "reviewer", "critic", "audit",
    ]
    art_keywords = [
        "artist", "art", "design", "concept", "visual",
    ]

    if any(k in name for k in coder_keywords):
        return "coder"
    if any(k in name for k in review_keywords):
        return "review"
    if any(k in name for k in art_keywords):
        return "art"
    return "default"

def get_agent_profile(agent_name: str):
    klass = AGENT_CLASSES.get(agent_name)
    if klass is None:
        klass = infer_agent_class(agent_name)
    return REFRESH_PROFILES.get(klass, REFRESH_PROFILES["default"])

def get_agent_memory_paths(agent_name: str):
    profile = get_agent_profile(agent_name)
    out = []
    for name in profile["files"]:
        if name in {"__LATEST_DAILY__", "__PROJECT_CURRENT__", "__PROJECT_LATEST_DAILY__"}:
            out.append(Path(name))
        else:
            out.append(MEMORY_DIR / name)
    return out

def get_agent_refresh_interval(agent_name: str) -> int:
    profile = get_agent_profile(agent_name)
    return int(profile["interval_seconds"])
