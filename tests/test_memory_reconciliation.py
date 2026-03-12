import json
import subprocess
import tempfile
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS = WORKSPACE / ".memory-index" / "scripts"
RECONCILE = SCRIPTS / "reconcile_memory_items.py"

def write_tmp(obj):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name

def run_reconcile(candidate, existing):
    cpath = write_tmp(candidate)
    epath = write_tmp(existing)

    result = subprocess.run(
        [
            "python3",
            str(RECONCILE),
            "--candidate",
            cpath,
            "--existing",
            epath,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)

def assert_outcome(name, candidate, existing, expected):
    out = run_reconcile(candidate, existing)
    actual = out["outcome"]
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")
    print(f"PASS {name}: {actual}")

def main():
    duplicate_candidate = {
        "id": "cand-dup",
        "text": "Native browser tool access is now CLI fallback through forwarded DevTools port 9224.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "tool_access_mode",
        "value": "cli_fallback_via_devtools_9224",
        "status": "active",
    }
    duplicate_existing = {
        "id": "exist-dup",
        "text": "Native browser tool access may be restricted in-session, but browser control still works through openclaw browser CLI against the forwarded DevTools port 9224.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "tool_access_mode",
        "value": "cli_fallback_via_devtools_9224",
        "status": "active",
    }

    supersede_candidate = {
        "id": "cand-sup",
        "text": "Native browser tool access is now CLI fallback through forwarded DevTools port 9224.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "tool_access_mode",
        "value": "cli_fallback_via_devtools_9224",
        "status": "active",
    }
    supersede_existing = {
        "id": "exist-sup",
        "text": "Native browser tool access is unavailable in-session.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "tool_access_mode",
        "value": "unavailable_in_session",
        "status": "active",
    }

    conflict_candidate = {
        "id": "cand-conflict",
        "text": "Browser capability is available.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "capability",
        "value": "browser_control_available",
        "status": "active",
    }
    conflict_existing = {
        "id": "exist-conflict",
        "text": "Browser capability is unavailable.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "capability",
        "value": "browser_control_unavailable",
        "status": "active",
    }

    merge_candidate = {
        "id": "cand-merge",
        "text": "OpenClaw has browser-control capability through browser automation features and CLI/browser-tool pathways.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "capability",
        "value": "browser_control_available",
        "status": "active",
    }
    merge_existing = {
        "id": "exist-merge",
        "text": "OpenClaw has browser-control capability.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": None,
        "value": None,
        "status": "active",
    }

    assert_outcome("duplicate", duplicate_candidate, duplicate_existing, "IGNORE_DUPLICATE")
    assert_outcome("supersede", supersede_candidate, supersede_existing, "SUPERSEDE_EXISTING")
    assert_outcome("conflict", conflict_candidate, conflict_existing, "FLAG_CONFLICT")
    assert_outcome("merge", merge_candidate, merge_existing, "MERGE_INTO_EXISTING")

    print()
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
