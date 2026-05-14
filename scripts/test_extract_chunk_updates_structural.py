from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_CHUNK_UPDATES = SCRIPT_DIR / "extract_chunk_updates.py"



def test_extract_chunk_updates_uses_structural_path_by_default(tmp_path: Path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "001.txt").write_text("USER: In pipeline Alpha, this rule must stay automated.\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(EXTRACT_CHUNK_UPDATES),
            "--chunk-dir",
            str(chunk_dir),
            "--source-agent",
            "tester",
            "--source-session",
            "session-1",
            "--reduced-only",
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip() and line.strip() != "NONE"]
    assert len(rows) == 1
    assert rows[0]["claim_type"] == "rule"
    assert rows[0]["scope_envelope"]["scope_type"] == "pipeline"



def test_extract_chunk_updates_shadow_compare_reports_structural_vs_legacy_counts(tmp_path: Path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "001.txt").write_text("USER: In pipeline Alpha, this rule must stay automated.\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(EXTRACT_CHUNK_UPDATES),
            "--chunk-dir",
            str(chunk_dir),
            "--source-agent",
            "tester",
            "--source-session",
            "session-1",
            "--reduced-only",
            "--shadow-compare",
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert '"structural_count":' in proc.stderr
    assert '"legacy_count":' in proc.stderr



def test_extract_chunk_updates_dedupes_structural_output(tmp_path: Path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    text = "USER: In pipeline Alpha, this rule must stay automated.\n"
    (chunk_dir / "001.txt").write_text(text, encoding="utf-8")
    (chunk_dir / "002.txt").write_text(text, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(EXTRACT_CHUNK_UPDATES),
            "--chunk-dir",
            str(chunk_dir),
            "--source-agent",
            "tester",
            "--source-session",
            "session-1",
            "--reduced-only",
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip() and line.strip() != "NONE"]
    assert len(rows) == 1
