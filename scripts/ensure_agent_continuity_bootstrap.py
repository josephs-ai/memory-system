#!/usr/bin/env python3
"""Ensure every OpenClaw agent workspace has the continuity boot surface.

This is intentionally idempotent and conservative:
- Only targets ~/.openclaw/workspace* directories and configured agent workspaces.
- Adds missing BOOT.md from the canonical main workspace.
- Adds missing MEMORY.md as a symlink to canonical main MEMORY.md.
- Adds/repairs .memory-index/timeline as a symlink to canonical generated chronology.
- Appends the continuity startup override to AGENTS.md only if absent.

It does not overwrite existing non-empty agent MEMORY.md files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HOME = Path.home()
OPENCLAW_ROOT = HOME / ".openclaw"
CANONICAL_WORKSPACE = OPENCLAW_ROOT / "workspace"
CANONICAL_BOOT = CANONICAL_WORKSPACE / "BOOT.md"
CANONICAL_MEMORY = CANONICAL_WORKSPACE / "MEMORY.md"
CANONICAL_TIMELINE = CANONICAL_WORKSPACE / ".memory-index" / "timeline"
CONFIG = OPENCLAW_ROOT / "openclaw.json"

CONTINUITY_OVERRIDE = """

## Continuity startup override (2026-06-22)

This workspace is not fresh. Do not follow any older "Hello, World"/"There is no memory yet" bootstrap text.
At startup and before recency/history answers, read durable memory plus generated chronology:
- `MEMORY.md`
- `.memory-index/timeline/latest.md`
- today's and, when relevant, yesterday's `.memory-index/timeline/daily/YYYY-MM-DD.md`
If timeline files are missing/stale, inspect recent active/reset/deleted transcripts before answering.
""".lstrip("\n")


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def configured_workspaces() -> set[Path]:
    rows: set[Path] = set()
    if not CONFIG.exists():
        return rows
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return rows
    agents = data.get("agents") or {}
    defaults = agents.get("defaults") or {}
    default_ws = defaults.get("workspace")
    if isinstance(default_ws, str):
        rows.add(expand_path(default_ws))
    for item in agents.get("list") or []:
        if isinstance(item, dict) and isinstance(item.get("workspace"), str):
            rows.add(expand_path(item["workspace"]))
    return rows


def discover_workspaces() -> list[Path]:
    rows: set[Path] = set()
    rows.add(CANONICAL_WORKSPACE.resolve())
    rows.update(configured_workspaces())
    for p in OPENCLAW_ROOT.glob("workspace*"):
        if p.is_dir():
            rows.add(p.resolve())
    return sorted(rows, key=str)


def safe_unlink(path: Path, dry_run: bool) -> bool:
    if path.is_symlink():
        if not dry_run:
            path.unlink()
        return True
    return False


def ensure_symlink(path: Path, target: Path, dry_run: bool) -> tuple[bool, str]:
    """Ensure path is a symlink to target when safe.

    Existing regular files/dirs are left alone unless they are known-empty/missing is not handled here.
    """
    target = target.resolve()
    if path.is_symlink():
        current = path.resolve()
        if current == target:
            return False, "ok"
        if not dry_run:
            path.unlink()
            path.symlink_to(target)
        return True, f"relinked {path} -> {target}"
    if path.exists():
        return False, "exists_non_symlink_left_unchanged"
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    return True, f"linked {path} -> {target}"


def ensure_file_copy_if_missing(path: Path, source: Path, dry_run: bool) -> tuple[bool, str]:
    if path.exists() or path.is_symlink():
        return False, "ok"
    if not source.exists():
        return False, f"source_missing:{source}"
    if not dry_run:
        path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return True, f"copied {source.name}"


def ensure_agents_override(path: Path, dry_run: bool) -> tuple[bool, str]:
    if not path.exists() or path.is_symlink():
        # If absent, copy main AGENTS.md when available so the workspace has something sane.
        source = CANONICAL_WORKSPACE / "AGENTS.md"
        if source.exists() and not path.exists():
            if not dry_run:
                path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            return True, "copied AGENTS.md"
        return False, "missing_agents_no_source"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"read_error:{exc}"
    if "Continuity startup override (2026-06-22)" in text:
        return False, "ok"
    if not dry_run:
        with path.open("a", encoding="utf-8") as f:
            if not text.endswith("\n"):
                f.write("\n")
            f.write("\n" + CONTINUITY_OVERRIDE)
    return True, "appended continuity override"


def repair_workspace(ws: Path, dry_run: bool = False) -> dict:
    changes: list[str] = []
    warnings: list[str] = []
    if not ws.exists():
        if not dry_run:
            ws.mkdir(parents=True, exist_ok=True)
        changes.append("created workspace dir")
    if not ws.is_dir():
        return {"workspace": str(ws), "changed": False, "changes": [], "warnings": ["not_a_directory"]}

    changed, msg = ensure_file_copy_if_missing(ws / "BOOT.md", CANONICAL_BOOT, dry_run)
    (changes if changed else warnings if msg.startswith("source_missing") else []).append(msg) if changed or msg.startswith("source_missing") else None

    # MEMORY.md: do not overwrite existing regular memory. Only add if absent.
    changed, msg = ensure_symlink(ws / "MEMORY.md", CANONICAL_MEMORY, dry_run)
    if changed:
        changes.append(msg)
    # Existing regular MEMORY.md files can be legitimate per-agent compact memories; leave them alone.

    changed, msg = ensure_symlink(ws / ".memory-index" / "timeline", CANONICAL_TIMELINE, dry_run)
    if changed:
        changes.append(msg)
    elif msg != "ok" and ws.resolve() != CANONICAL_WORKSPACE.resolve():
        warnings.append(f"timeline:{msg}")

    changed, msg = ensure_agents_override(ws / "AGENTS.md", dry_run)
    if changed:
        changes.append(msg)
    elif msg not in ("ok",):
        warnings.append(f"AGENTS.md:{msg}")

    # Ensure BOOT copy message didn't get lost.
    if not (ws / "BOOT.md").exists() and not (ws / "BOOT.md").is_symlink():
        warnings.append("BOOT.md:missing")

    return {"workspace": str(ws), "changed": bool(changes), "changes": changes, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", action="append", help="Specific workspace to repair; may be passed multiple times")
    parser.add_argument("--all", action="store_true", help="Repair all discovered/configured workspaces")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.workspace:
        workspaces = [expand_path(w) for w in args.workspace]
    else:
        workspaces = discover_workspaces()

    # Default to all discovered workspaces; --all exists for readability/backcompat.
    results = [repair_workspace(ws, dry_run=args.dry_run) for ws in workspaces]
    summary = {
        "checked": len(results),
        "changed": sum(1 for r in results if r["changed"]),
        "warnings": sum(len(r["warnings"]) for r in results),
        "results": results,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"checked={summary['checked']} changed={summary['changed']} warnings={summary['warnings']}")
        for r in results:
            if r["changed"] or r["warnings"]:
                print(f"- {r['workspace']}")
                for c in r["changes"]:
                    print(f"  changed: {c}")
                for w in r["warnings"]:
                    print(f"  warn: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
