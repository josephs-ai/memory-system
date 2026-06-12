"""
statistical_utils.py — Statistical utilities for benchmark analysis.

Provides:
- Bootstrap confidence intervals
- Paired significance testing (permutation test)
- Effect size computation
- Aggregation with error bars
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class MetricSummary:
    """Aggregated metric with confidence interval."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int
    values: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "n": self.n,
        }


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> MetricSummary:
    """Compute bootstrap confidence interval for the mean."""
    if not values:
        return MetricSummary(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n=0)

    vals = list(values)
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    std = math.sqrt(variance)

    if n < 3:
        return MetricSummary(mean=mean, std=std, ci_lower=mean, ci_upper=mean, n=n, values=vals)

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(vals) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1 - ci
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    return MetricSummary(
        mean=mean,
        std=std,
        ci_lower=boot_means[lo_idx],
        ci_upper=boot_means[hi_idx],
        n=n,
        values=vals,
    )


def permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_permutations: int = 5000,
    seed: int | None = 42,
) -> dict:
    """
    Paired permutation test for difference in means.
    Returns p-value and observed difference (a - b).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired test requires equal-length sequences")

    n = len(scores_a)
    if n == 0:
        return {"p_value": 1.0, "observed_diff": 0.0, "n": 0}

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    observed = sum(diffs) / n

    rng = random.Random(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        perm_diffs = [d * (1 if rng.random() > 0.5 else -1) for d in diffs]
        perm_mean = sum(perm_diffs) / n
        if abs(perm_mean) >= abs(observed):
            count_extreme += 1

    p_value = count_extreme / n_permutations

    return {
        "p_value": round(p_value, 4),
        "observed_diff": round(observed, 4),
        "n": n,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def cohens_d(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Compute Cohen's d effect size using pooled standard deviation.

    Uses the standard (not paired/d_z) formula:
        d = (mean_a - mean_b) / s_pooled
    where s_pooled = sqrt(((n_a-1)*s_a^2 + (n_b-1)*s_b^2) / (n_a + n_b - 2))
    """
    n_a = len(scores_a)
    n_b = len(scores_b)
    if n_a == 0 or n_b == 0:
        return 0.0

    mean_a = sum(scores_a) / n_a
    mean_b = sum(scores_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in scores_a) / max(n_a - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / max(n_b - 1, 1)

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    s_pooled = math.sqrt(pooled_var)

    if s_pooled == 0:
        return float("inf") if mean_a != mean_b else 0.0
    return round((mean_a - mean_b) / s_pooled, 4)


def aggregate_metrics(
    results: list[dict],
    metric_keys: list[str],
    group_key: str | None = None,
) -> dict[str, dict[str, MetricSummary]]:
    """
    Aggregate metrics from a list of result dicts, optionally grouped.
    Returns {group: {metric: MetricSummary}}.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        key = r.get(group_key, "ALL") if group_key else "ALL"
        groups.setdefault(key, []).append(r)

    # Always include ALL
    if "ALL" not in groups and group_key:
        groups["ALL"] = results

    output = {}
    for group, items in groups.items():
        output[group] = {}
        for mk in metric_keys:
            vals = [r[mk] for r in items if mk in r and r[mk] is not None]
            output[group][mk] = bootstrap_ci(vals)

    return output
