"""
scaffold_canonical_memory.py — create the canonical memory layout (Phase 1).

Thin CLI over canonical_memory_paths.scaffold(). Idempotent. Use --dry-run to
preview without touching disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import canonical_memory_paths as cmp


def _planned_dirs() -> list[str]:
    root = cmp.canonical_root()
    dirs = [".", "session", "daily", "resources"]
    dirs += [f"digest/{k}" for k in cmp.DIGEST_KINDS]
    dirs += [f"metadata/{k}" for k in cmp.METADATA_KINDS]
    return [str(root / d if d != "." else root) for d in dirs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scaffold the canonical memory layout")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, do not create")
    ap.add_argument("--no-readme", action="store_true", help="do not write README.md")
    ap.add_argument("--json", action="store_true", help="machine output")
    args = ap.parse_args(argv)

    if args.dry_run:
        plan = {"root": str(cmp.canonical_root()), "would_create": _planned_dirs()}
        print(json.dumps(plan, indent=2) if args.json else "\n".join(plan["would_create"]))
        return 0

    result = cmp.scaffold(write_readme=not args.no_readme)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Canonical memory scaffolded at {result['root']}")
        if result["created"]:
            print("Created:")
            for c in result["created"]:
                print(f"  + {c}")
        else:
            print("(already present — nothing to create)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
