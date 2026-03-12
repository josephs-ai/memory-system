from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"

CONTROL_PANEL_DIR = WORKSPACE / ".memory-index" / "control_panel"
CONTROL_DIR = CONTROL_PANEL_DIR

CONFIG_PATH = CONTROL_PANEL_DIR / "control_panel_config.json"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
MEMORY_DIR = WORKSPACE / "memory"
PROJECTS_DIR = MEMORY_DIR / "projects"

DEFAULT_PORT = int(os.environ.get("OPENCLAW_CONTROL_PANEL_PORT", "8788"))
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

DEFAULT_CONFIG = {
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
    },
}
