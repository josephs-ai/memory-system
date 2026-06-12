# Phase 1 — Make Performance Undeniable

Benchmark suite, ablation framework, and baseline comparison tooling
for the memory engine retrieval pipeline.

## Contents

- `benchmark_suite.py` — Expanded benchmark runner (all query classes, full dataset, per-class breakdown)
- `ablation_runner.py` — Configurable ablation framework with feature toggles
- `baseline_comparisons.py` — Side-by-side comparison of retrieval strategies
- `statistical_utils.py` — Confidence intervals, significance tests
- `report_generator.py` — Generates markdown/JSON reports from benchmark runs
- `task_success_eval.py` — Downstream task success measurement framework

## Running

```bash
# Full benchmark (all 500 questions, all classes)
python benchmarks/phase1/benchmark_suite.py run --dataset benchmarks/longmemeval/longmemeval_s.json

# Ablation study
python benchmarks/phase1/ablation_runner.py run --ablations all

# Baseline comparison
python benchmarks/phase1/baseline_comparisons.py run

# Generate consolidated report
python benchmarks/phase1/report_generator.py
```
