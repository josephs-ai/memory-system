import json
import subprocess
import tempfile
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS = WORKSPACE / ".memory-index" / "scripts"

CHUNK_EXTRACT = SCRIPTS / "extract_chunk_updates.py"
RECONCILE = SCRIPTS / "reconcile_memory_items.py"

def write_text_tmp(text: str, suffix: str = ".txt") -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return tmp.name

def write_json_tmp(obj) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name

def run_extract_on_text(text: str):
    chunk_dir = Path(tempfile.mkdtemp())
    chunk_file = chunk_dir / "case.chunk01.txt"
    chunk_file.write_text(text, encoding="utf-8")

    result = subprocess.run(
        [
            "python3",
            str(CHUNK_EXTRACT),
            "--chunk-dir",
            str(chunk_dir),
            "--source-agent",
            "test",
            "--source-session",
            "test-session",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout

def run_reconcile(candidate: dict, existing: dict):
    cpath = write_json_tmp(candidate)
    epath = write_json_tmp(existing)
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
    return result.stdout

def assert_contains(name: str, text: str, needle: str):
    if needle not in text:
        raise AssertionError(f"{name}: expected to contain {needle!r}\nGot:\n{text}")
    print(f"PASS {name}")

def main():
    startup_case = """USER: A new session was started via /new or /reset. Execute your Session Startup sequence now - read the required files before responding to the user.
ASSISTANT: Hey. I just came online. Who am I? Who are you?
USER: hello?
ASSISTANT: What should my name be?"""

    durable_case = """USER: Canonical memory now uses a structured registry for active items instead of inferring item fields from markdown bullets."""

    preference_case = """USER: Joseph prefers memory automation upgrades to be built before optional polish."""

    browser_case = """USER: Visible browser state is managed through the forwarded DevTools port 9224."""

    project_case = """USER: Worker upload API currently uses a temporary compatibility path."""

    daily_case = """USER: During a checkpoint run, we observed a one-off test output mismatch."""

    startup_out = run_extract_on_text(startup_case)
    assert_contains("startup_case_none", startup_out, "NONE")

    durable_out = run_extract_on_text(durable_case)
    assert_contains("durable_case_decision", durable_out, '"memory_type": "decision"')
    assert_contains("durable_case_entity", durable_out, '"entity": "checkpoint_pipeline"')
    assert_contains("durable_case_route", durable_out, '"suggested_route": "auto"')
    assert_contains("durable_case_freshness", durable_out, '"freshness_class": "permanent"')

    pref_out = run_extract_on_text(preference_case)
    assert_contains("preference_case_type", pref_out, '"memory_type": "preference"')
    assert_contains("preference_case_entity", pref_out, '"entity": "user_preference"')
    assert_contains("preference_case_freshness", pref_out, '"freshness_class": "long_lived"')

    browser_out = run_extract_on_text(browser_case)
    assert_contains("browser_case_entity", browser_out, '"entity": "browser"')
    assert_contains("browser_case_property", browser_out, '"property": "forwarded_devtools_port"')
    assert_contains("browser_case_freshness", browser_out, '"freshness_class": "volatile"')

    project_out = run_extract_on_text(project_case)
    assert_contains("project_case_scope", project_out, '"scope": "project"')

    daily_out = run_extract_on_text(daily_case)
    assert_contains("daily_case_scope", daily_out, '"scope": "daily"')

    uncertain_candidate = {
        "id": "cand-uncertain",
        "text": "Browser tool access may use another mode.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "tool_access_mode",
        "value": "possible_other_mode",
        "status": "active",
        "confidence": 0.60
    }
    uncertain_existing = {
        "id": "exist-uncertain",
        "text": "Native browser tool access may be restricted in-session, but browser control still works through openclaw browser CLI against the forwarded DevTools port 9224.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "tool_access_mode",
        "value": "cli_fallback_via_devtools_9224",
        "status": "active",
        "confidence": 0.91
    }
    uncertain_out = run_reconcile(uncertain_candidate, uncertain_existing)
    assert_contains("uncertain_reconcile", uncertain_out, '"outcome": "MARK_UNCERTAIN"')

    archive_candidate = {
        "id": "cand-archive",
        "text": "Visible browser state is managed through the forwarded DevTools port 9224.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "forwarded_devtools_port",
        "value": "9224",
        "status": "active",
        "confidence": 0.90
    }
    archive_existing = {
        "id": "exist-archive",
        "text": "Visible browser state was previously managed elsewhere.",
        "memory_type": "fact",
        "scope": "stable",
        "entity": "browser",
        "property": "forwarded_devtools_port",
        "value": "9333",
        "status": "superseded",
        "confidence": 0.90
    }
    archive_out = run_reconcile(archive_candidate, archive_existing)
    assert_contains("archive_reconcile", archive_out, '"outcome": "ARCHIVE_OLD"')

    print()
    print("ALL TRANSCRIPT PIPELINE TESTS PASSED")

if __name__ == "__main__":
    main()
