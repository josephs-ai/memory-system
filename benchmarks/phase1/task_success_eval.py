#!/usr/bin/env python3
"""
task_success_eval.py — Downstream task success evaluation framework.

Phase 1 item 3: Measure whether memory actually improves agent task performance.

Eval dimensions:
1. Task completion accuracy (does memory help the agent complete the task correctly?)
2. Repeated mistake prevention (does memory reduce error recurrence?)
3. Context efficiency (does memory reduce prompt/context size?)
4. Consistency (does memory improve cross-session answer consistency?)
5. Time-to-solution (does memory speed up task completion?)

This provides the evaluation harness and dataset schema.
Actual test cases are loaded from task_success_dataset.json.

Usage:
    python task_success_eval.py seed        # Generate initial dataset template
    python task_success_eval.py run         # Run eval suite
    python task_success_eval.py report      # Generate report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE1_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1_DIR / "results"
DATASET_PATH = PHASE1_DIR / "task_success_dataset.json"

sys.path.insert(0, str(PHASE1_DIR))
from statistical_utils import bootstrap_ci


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

TASK_SCHEMA = {
    "task_id": "unique identifier",
    "category": "one of: task_completion | mistake_prevention | context_efficiency | consistency | time_to_solution",
    "description": "What this test case measures",
    "setup": {
        "memory_items": "list of memory items to inject before the task",
        "task_prompt": "the prompt/question the agent receives",
        "expected_behavior": "what correct behavior looks like",
    },
    "eval_criteria": {
        "metric": "one of: accuracy | error_rate | context_tokens | answer_match | latency_ms",
        "threshold": "pass/fail threshold",
        "direction": "higher_better or lower_better",
    },
    "variants": [
        {"name": "with_memory", "memory_enabled": True},
        {"name": "without_memory", "memory_enabled": False},
    ],
}


# ---------------------------------------------------------------------------
# Seed dataset
# ---------------------------------------------------------------------------

def seed_dataset():
    """Generate a template task success dataset with example cases."""
    cases = [
        # Category 1: Task completion accuracy
        {
            "task_id": "tc-001",
            "category": "task_completion",
            "description": "Agent should use remembered preference for output format",
            "setup": {
                "memory_items": [
                    {"text": "User prefers JSON output format over YAML", "type": "preference", "confidence": 0.95},
                    {"text": "User works on a Python project called 'memory-engine'", "type": "fact", "confidence": 0.9},
                ],
                "task_prompt": "Generate a config file for the database connection",
                "expected_behavior": "Agent produces JSON format without being asked",
            },
            "eval_criteria": {
                "metric": "accuracy",
                "threshold": 0.8,
                "direction": "higher_better",
                "judge_prompt": "Did the agent produce JSON output? Score 1 if yes, 0 if no.",
            },
        },
        {
            "task_id": "tc-002",
            "category": "task_completion",
            "description": "Agent should remember architectural decision about database choice",
            "setup": {
                "memory_items": [
                    {"text": "Decided to use PostgreSQL for the memory store, not MySQL. Reason: better JSON support, pg_trgm for fuzzy search.", "type": "decision", "confidence": 0.95},
                    {"text": "Neo4j is used only for graph relationships, not as primary store", "type": "architectural_truth", "confidence": 0.9},
                ],
                "task_prompt": "What database should we use for storing memory items?",
                "expected_behavior": "Agent recommends PostgreSQL and explains the prior decision",
            },
            "eval_criteria": {
                "metric": "accuracy",
                "threshold": 0.9,
                "direction": "higher_better",
                "judge_prompt": "Did the agent mention PostgreSQL as the chosen database and reference the prior decision? Score 1 if yes, 0 if no.",
            },
        },

        # Category 2: Mistake prevention
        {
            "task_id": "mp-001",
            "category": "mistake_prevention",
            "description": "Agent should avoid a previously documented pitfall",
            "setup": {
                "memory_items": [
                    {"text": "BUG: Using psycopg2 with asyncio causes connection pool exhaustion. Must use psycopg (v3) async API instead.", "type": "learned_fix", "confidence": 0.95},
                ],
                "task_prompt": "Write a function to query the database asynchronously",
                "expected_behavior": "Agent uses psycopg (v3) async, NOT psycopg2",
            },
            "eval_criteria": {
                "metric": "error_rate",
                "threshold": 0.1,
                "direction": "lower_better",
                "judge_prompt": "Does the code use psycopg2 (bad) or psycopg v3 async (good)? Score 0 if psycopg2, 1 if psycopg v3.",
            },
        },

        # Category 3: Context efficiency
        {
            "task_id": "ce-001",
            "category": "context_efficiency",
            "description": "Memory should reduce the context needed to answer correctly",
            "setup": {
                "memory_items": [
                    {"text": "The search service runs on port 8765", "type": "fact", "confidence": 0.9},
                    {"text": "Qdrant runs on port 6333", "type": "fact", "confidence": 0.9},
                    {"text": "Neo4j bolt port is 7687", "type": "fact", "confidence": 0.9},
                ],
                "task_prompt": "What ports do our services run on?",
                "expected_behavior": "Agent answers from memory without needing to read config files",
            },
            "eval_criteria": {
                "metric": "context_tokens",
                "threshold": 500,
                "direction": "lower_better",
            },
        },

        # Category 4: Cross-session consistency
        {
            "task_id": "cs-001",
            "category": "consistency",
            "description": "Same question across sessions should produce consistent answers",
            "setup": {
                "memory_items": [
                    {"text": "Project uses MIT license", "type": "fact", "confidence": 0.95},
                ],
                "task_prompt": "What license does the project use?",
                "expected_behavior": "Agent consistently says MIT across multiple sessions",
                "n_sessions": 5,
            },
            "eval_criteria": {
                "metric": "answer_match",
                "threshold": 0.9,
                "direction": "higher_better",
                "judge_prompt": "Do all answers agree that the license is MIT? Score = fraction of matching answers.",
            },
        },
    ]

    DATASET_PATH.write_text(json.dumps(cases, indent=2))
    print(f"Seed dataset written to {DATASET_PATH} ({len(cases)} cases)", flush=True)
    return cases


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

@staticmethod
def _count_tokens_approx(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars)."""
    return len(text) // 4


class TaskEvaluator:
    """Runs task success evaluations with and without memory."""

    def __init__(self, openai_key: str | None = None):
        self.api_key = openai_key or os.environ.get("OPENAI_API_KEY", "")

    def evaluate_case(self, case: dict) -> dict:
        """Run a single evaluation case (with and without memory)."""
        task_id = case["task_id"]
        category = case["category"]
        criteria = case["eval_criteria"]

        print(f"\n[{task_id}] {case['description']}", flush=True)

        results = {
            "task_id": task_id,
            "category": category,
            "description": case["description"],
            "variants": {},
        }

        for variant in case.get("variants", [{"name": "with_memory", "memory_enabled": True}, {"name": "without_memory", "memory_enabled": False}]):
            vname = variant["name"]
            memory_enabled = variant.get("memory_enabled", True)

            # Build context
            setup = case["setup"]
            context = ""
            if memory_enabled and "memory_items" in setup:
                context = "Relevant memories:\n"
                for item in setup["memory_items"]:
                    context += f"- [{item.get('type', 'fact')}] {item['text']}\n"
                context += "\n"

            prompt = f"{context}Task: {setup['task_prompt']}"

            # Get LLM response
            t0 = time.monotonic()
            response = self._call_llm(prompt)
            latency = time.monotonic() - t0

            # Judge the response
            score = self._judge_response(response, case, variant)

            results["variants"][vname] = {
                "memory_enabled": memory_enabled,
                "prompt_tokens": _count_tokens_approx(prompt),
                "response": response[:500],
                "latency_ms": round(latency * 1000, 1),
                "score": score,
            }

            status = "✓" if score >= criteria.get("threshold", 0.5) else "✗"
            print(f"  {vname}: {status} score={score:.2f} latency={latency*1000:.0f}ms", flush=True)

        # Compute delta
        with_score = results["variants"].get("with_memory", {}).get("score", 0)
        without_score = results["variants"].get("without_memory", {}).get("score", 0)
        results["memory_delta"] = round(with_score - without_score, 4)
        results["memory_helps"] = with_score > without_score

        return results

    def _call_llm(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Call OpenAI API for task execution."""
        if not self.api_key:
            return "[NO API KEY — SIMULATED RESPONSE]"

        import requests
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as ex:
            return f"[ERROR: {ex}]"

    def _judge_response(self, response: str, case: dict, variant: dict) -> float:
        """Judge the response quality. Uses LLM-as-judge if judge_prompt available."""
        criteria = case["eval_criteria"]
        judge_prompt = criteria.get("judge_prompt")

        if not judge_prompt:
            # Heuristic: check if expected keywords appear
            expected = case["setup"].get("expected_behavior", "")
            if not expected:
                return 0.5
            expected_words = set(expected.lower().split())
            response_words = set(response.lower().split())
            overlap = len(expected_words & response_words) / max(len(expected_words), 1)
            return round(overlap, 4)

        if not self.api_key:
            return 0.5  # Cannot judge without API

        # LLM-as-judge
        judge_input = f"""Evaluate this response:

TASK: {case['setup']['task_prompt']}
EXPECTED: {case['setup']['expected_behavior']}
RESPONSE: {response}

{judge_prompt}

Return ONLY a number between 0 and 1."""

        import requests
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": judge_input}],
                    "max_tokens": 10,
                    "temperature": 0,
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return float(raw)
        except Exception:
            return 0.5


def _count_tokens_approx(text: str) -> int:
    return len(text) // 4


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_task_report(results: list[dict]) -> str:
    """Generate markdown report from task eval results."""
    lines = [
        "# Phase 1 — Task Success Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Does Memory Improve Task Success?",
        "",
        "| Task | Category | With Memory | Without | Delta | Helps? |",
        "|------|----------|-------------|---------|-------|--------|",
    ]

    helps_count = 0
    total = 0

    for r in results:
        with_s = r["variants"].get("with_memory", {}).get("score", 0)
        without_s = r["variants"].get("without_memory", {}).get("score", 0)
        delta = r.get("memory_delta", 0)
        helps = "✓" if r.get("memory_helps") else "✗"
        if r.get("memory_helps"):
            helps_count += 1
        total += 1

        lines.append(f"| {r['task_id']} | {r['category']} | {with_s:.2f} | {without_s:.2f} | {delta:+.2f} | {helps} |")

    lines.extend([
        "",
        f"**Memory helps in {helps_count}/{total} cases ({helps_count/max(total,1)*100:.0f}%)**",
        "",
        "## By Category",
        "",
    ])

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    for cat, items in sorted(by_cat.items()):
        avg_delta = sum(r.get("memory_delta", 0) for r in items) / max(len(items), 1)
        pct_helps = sum(1 for r in items if r.get("memory_helps")) / max(len(items), 1)
        lines.append(f"### {cat}")
        lines.append(f"- n={len(items)}, avg delta={avg_delta:+.3f}, helps={pct_helps*100:.0f}%")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task Success Evaluation")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("seed", help="Generate seed dataset")
    run_p = sub.add_parser("run", help="Run evaluation")
    run_p.add_argument("--dataset", default=str(DATASET_PATH))
    sub.add_parser("report", help="Generate report from latest results")

    args = parser.parse_args()

    if args.command == "seed":
        seed_dataset()

    elif args.command == "run":
        if not Path(args.dataset).exists():
            print("No dataset found. Run 'seed' first.", flush=True)
            return

        with open(args.dataset) as f:
            cases = json.load(f)

        evaluator = TaskEvaluator()
        results = []
        for case in cases:
            result = evaluator.evaluate_case(case)
            results.append(result)

        # Save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"task_success_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}", flush=True)

        # Report
        report = generate_task_report(results)
        report_path = RESULTS_DIR / f"task_success_report_{ts}.md"
        report_path.write_text(report)
        print(report, flush=True)

    elif args.command == "report":
        # Load latest results
        files = sorted(RESULTS_DIR.glob("task_success_2*.json"), reverse=True)
        if not files:
            print("No results found. Run eval first.", flush=True)
            return

        with open(files[0]) as f:
            results = json.load(f)
        report = generate_task_report(results)
        print(report, flush=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
