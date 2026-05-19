from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from memory_candidate_schema import AuthorityBasis, ClaimType, StructuredMemoryCandidate
from memory_validation_gate import validate_candidate
from structured_memory_extractor import extract_structured_candidates

SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_UPDATES = SCRIPT_DIR / "extract_updates.py"



def test_structured_extractor_returns_schema_candidates():
    rows = extract_structured_candidates(
        "USER: In pipeline Alpha, this rule must stay automated.\n",
        source_agent="tester",
        source_session="session-1",
        source_chunk="chunk-1",
    )

    assert len(rows) == 1
    candidate = rows[0]
    assert candidate.claim_type == ClaimType.rule
    assert candidate.scope_envelope.scope_type == "pipeline"
    assert candidate.scope_envelope.pipeline_id == "alpha"



def test_validation_gate_marks_unknown_authority_for_review():
    candidate = StructuredMemoryCandidate(
        candidate_id="cand-1",
        claim_text="Rule: keep Pipeline Alpha automated.",
        authority_basis=AuthorityBasis.unknown,
    )

    validated = validate_candidate(candidate)

    assert validated.validation_status == "needs_review"
    assert "authority_not_explicit" in validated.validation_reasons



def test_extract_updates_is_now_structural_wrapper(tmp_path: Path):
    input_path = tmp_path / "chunk.txt"
    input_path.write_text("USER: In pipeline Alpha, this rule must stay automated.\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(EXTRACT_UPDATES),
            str(input_path),
            "--source-agent",
            "tester",
            "--source-session",
            "session-1",
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip() and line.strip() != "NONE"]
    assert len(rows) == 1
    row = rows[0]
    assert row["claim_type"] == "rule"
    assert row["scope_envelope"]["scope_type"] == "pipeline"
    assert row["durability_class"] == "durable_eligible"
