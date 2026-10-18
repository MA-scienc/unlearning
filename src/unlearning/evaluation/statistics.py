"""Paired, cluster-aware summaries for counterfactual unlearning comparisons."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any


def paired_cluster_bootstrap(
    baseline: Mapping[str, float],
    comparison: Mapping[str, float],
    *,
    seed: int,
    samples: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate a paired mean delta with resampling at the cluster level.

    Each key is one independent cluster, such as a synthetic patient record or
    a benchmark question. Template-level scores should be aggregated before
    calling this function, so correlated probes are not treated as independent
    observations. ``mean_delta`` is comparison minus baseline.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    baseline_ids = set(baseline)
    comparison_ids = set(comparison)
    if not baseline_ids:
        raise ValueError("paired bootstrap requires at least one cluster")
    if baseline_ids != comparison_ids:
        missing_from_comparison = sorted(baseline_ids - comparison_ids)
        missing_from_baseline = sorted(comparison_ids - baseline_ids)
        raise ValueError(
            "Paired bootstrap requires identical cluster IDs; "
            f"missing_from_comparison={missing_from_comparison[:5]}, "
            f"missing_from_baseline={missing_from_baseline[:5]}"
        )

    cluster_ids = sorted(baseline_ids)
    baseline_values = [float(baseline[cluster_id]) for cluster_id in cluster_ids]
    comparison_values = [float(comparison[cluster_id]) for cluster_id in cluster_ids]
    if not all(math.isfinite(value) for value in [*baseline_values, *comparison_values]):
        raise ValueError("paired bootstrap requires finite cluster scores")
    deltas = [comparison_value - baseline_value for baseline_value, comparison_value in zip(
        baseline_values, comparison_values, strict=True
    )]
    rng = random.Random(seed)
    count = len(cluster_ids)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(deltas[rng.randrange(count)] for _ in range(count)) / count)
    estimates.sort()
    lower_index = int((alpha / 2) * (samples - 1))
    upper_index = int((1 - (alpha / 2)) * (samples - 1))
    return {
        "cluster_count": count,
        "baseline_mean": sum(baseline_values) / count,
        "comparison_mean": sum(comparison_values) / count,
        "mean_delta": sum(deltas) / count,
        "delta_ci": (estimates[lower_index], estimates[upper_index]),
        "resampling_unit": "cluster",
        "samples": samples,
        "alpha": alpha,
        "seed": seed,
    }


def aggregate_scores_by_cluster(
    rows: list[Mapping[str, Any]],
    *,
    cluster_key: str,
    value_key: str,
) -> dict[str, float]:
    """Mean correlated probe scores within each cluster before paired inference."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        if cluster_key not in row or value_key not in row:
            raise ValueError(f"Each row must contain {cluster_key!r} and {value_key!r}")
        cluster_id = str(row[cluster_key])
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("Cluster score values must be finite")
        grouped.setdefault(cluster_id, []).append(value)
    if not grouped:
        raise ValueError("Cannot aggregate an empty row set")
    return {
        cluster_id: sum(values) / len(values)
        for cluster_id, values in grouped.items()
    }
