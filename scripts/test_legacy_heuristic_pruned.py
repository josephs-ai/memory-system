from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE = SCRIPT_DIR / "generate_memory_candidates.py"
FILTER = SCRIPT_DIR / "filter_memory_candidates.py"
JUDGE = SCRIPT_DIR / "judge_memory_candidates.py"



def test_generate_memory_candidates_now_uses_structural_extractor(tmp_path: Path):
    input_path = tmp_path / "chunk.txt"
    input_path.write_text("USER: In pipeline Alpha, this rule must stay automated.\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(GENERATE), str(input_path), "--source-agent", "tester", "--source-session", "session-1"],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip() and line.strip() != "NONE"]
    assert len(rows) == 1
    assert rows[0]["claim_type"] == "rule"



def test_filter_and_judge_are_non_heuristic_passthroughs(tmp_path: Path):
    input_path = tmp_path / "items.jsonl"
    input_path.write_text(json.dumps({"normalized_text": "Rule text", "contains_reasoning": False, "contains_speculation": False}) + "\n", encoding="utf-8")

    filtered = subprocess.run([sys.executable, str(FILTER), str(input_path)], capture_output=True, text=True, cwd=str(SCRIPT_DIR), check=False)
    judged = subprocess.run([sys.executable, str(JUDGE), str(input_path)], capture_output=True, text=True, cwd=str(SCRIPT_DIR), check=False)

    assert filtered.returncode == 0, filtered.stderr
    assert judged.returncode == 0, judged.stderr
    assert "Rule text" in filtered.stdout
    assert "Rule text" in judged.stdout
