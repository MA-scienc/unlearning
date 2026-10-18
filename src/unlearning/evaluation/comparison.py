"""Strict paired comparisons for checkpoint-level evaluation artifacts.

The pilot has correlated probes per synthetic patient.  These helpers require
identical evaluated items on both sides, aggregate templates within each record,
and then call the record-cluster bootstrap.  They intentionally refuse missing
or mismatched rows instead of silently producing a favorable subset.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unlearning.evaluation.statistics import paired_cluster_bootstrap
from unlearning.utils.io import read_json


def _finite_score(value: Any, name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite")
    return score


def _row_identity(row: Mapping[str, Any], item_key: str) -> str:
    value = row.get(item_key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Every evaluation row needs a non-empty {item_key!r}")
    return value


def _index_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    item_key: str,
    cluster_key: str,
) -> dict[str, tuple[str, float]]:
    """Select exactly the rows for a metric and index them by stable item ID."""

    indexed: dict[str, tuple[str, float]] = {}
    for row in rows:
        if metric not in row or row[metric] is None:
            continue
        item_id = _row_identity(row, item_key)
        if item_id in indexed:
            raise ValueError(f"Duplicate evaluation item {item_id!r} for metric {metric!r}")
        cluster_id = row.get(cluster_key)
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError(f"Every evaluation row needs a non-empty {cluster_key!r}")
        indexed[item_id] = (cluster_id, _finite_score(row[metric], f"{metric} for {item_id}"))
    if not indexed:
        raise ValueError(f"No non-null rows found for metric {metric!r}")
    return indexed


def paired_cluster_metric_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    seed: int,
    cluster_key: str = "record_id",
    item_key: str = "example_id",
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Compare two evaluations at matched items and record-level clusters.

    ``mean_delta`` is comparison minus baseline. For leakage metrics a negative
    value means the comparison model leaks less; for utility accuracy it means
    lower retained performance. Interpretation belongs in the analysis plan,
    not in this generic function.
    """

    baseline = _index_metric_rows(
        baseline_rows,
        metric=metric,
        item_key=item_key,
        cluster_key=cluster_key,
    )
    comparison = _index_metric_rows(
        comparison_rows,
        metric=metric,
        item_key=item_key,
        cluster_key=cluster_key,
    )
    if set(baseline) != set(comparison):
        raise ValueError(
            "Paired comparison requires identical evaluated item IDs; "
            f"missing_from_comparison={sorted(set(baseline) - set(comparison))[:5]}, "
            f"missing_from_baseline={sorted(set(comparison) - set(baseline))[:5]}"
        )
    baseline_by_cluster: dict[str, list[float]] = defaultdict(list)
    comparison_by_cluster: dict[str, list[float]] = defaultdict(list)
    for item_id in sorted(baseline):
        baseline_cluster, baseline_value = baseline[item_id]
        comparison_cluster, comparison_value = comparison[item_id]
        if baseline_cluster != comparison_cluster:
            raise ValueError(
                f"Item {item_id!r} belongs to different clusters: "
                f"{baseline_cluster!r} vs {comparison_cluster!r}"
            )
        baseline_by_cluster[baseline_cluster].append(baseline_value)
        comparison_by_cluster[comparison_cluster].append(comparison_value)
    baseline_cluster_means = {
        cluster_id: sum(values) / len(values)
        for cluster_id, values in baseline_by_cluster.items()
    }
    comparison_cluster_means = {
        cluster_id: sum(values) / len(values)
        for cluster_id, values in comparison_by_cluster.items()
    }
    bootstrap = paired_cluster_bootstrap(
        baseline_cluster_means,
        comparison_cluster_means,
        seed=seed,
        samples=bootstrap_samples,
    )
    return {
        "metric": metric,
        "item_key": item_key,
        "cluster_key": cluster_key,
        "item_count": len(baseline),
        "cluster_scores": {
            "baseline": baseline_cluster_means,
            "comparison": comparison_cluster_means,
        },
        "paired_bootstrap": bootstrap,
    }


def _checkpoint_identity(payload: Mapping[str, Any]) -> str | None:
    provenance = payload.get("checkpoint_provenance")
    if not isinstance(provenance, Mapping):
        return None
    manifest = provenance.get("manifest")
    if isinstance(manifest, Mapping):
        checkpoint_id = manifest.get("checkpoint_id")
        return checkpoint_id if isinstance(checkpoint_id, str) and checkpoint_id else None
    return None


def validate_unlearning_triplet(
    clean_payload: Mapping[str, Any],
    contaminated_payload: Mapping[str, Any],
    unlearned_payload: Mapping[str, Any],
) -> dict[str, str]:
    """Verify that a result triplet is C_i -> U_i and references its own S_i."""

    clean_id = _checkpoint_identity(clean_payload)
    contaminated_id = _checkpoint_identity(contaminated_payload)
    unlearned_id = _checkpoint_identity(unlearned_payload)
    if not clean_id or not contaminated_id or not unlearned_id:
        raise ValueError("Clean, contaminated, and unlearned results need manifest-backed checkpoint IDs")
    unlearned_manifest = unlearned_payload["checkpoint_provenance"]["manifest"]
    if unlearned_manifest.get("source_contaminated_checkpoint_id") != contaminated_id:
        raise ValueError("Unlearned result does not descend from the supplied contaminated result")
    if unlearned_manifest.get("clean_counterfactual_checkpoint_id") != clean_id:
        raise ValueError("Unlearned result does not reference the supplied clean counterfactual")
    return {
        "clean_checkpoint_id": clean_id,
        "contaminated_checkpoint_id": contaminated_id,
        "unlearned_checkpoint_id": unlearned_id,
    }


def _shared_data_provenance(
    baseline_payload: Mapping[str, Any], comparison_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Require matching F or retention-data identity where result types expose it."""

    baseline_forget = baseline_payload.get("forget_dataset")
    comparison_forget = comparison_payload.get("forget_dataset")
    if isinstance(baseline_forget, Mapping) or isinstance(comparison_forget, Mapping):
        if not isinstance(baseline_forget, Mapping) or not isinstance(comparison_forget, Mapping):
            raise ValueError("Cannot pair a target-forgetting result with a non-forgetting result")
        keys = ("eval_examples_fingerprint", "records_fingerprint")
        if any(baseline_forget.get(key) != comparison_forget.get(key) for key in keys):
            raise ValueError("Target-forgetting results were evaluated on different F artifacts")
        return {key: baseline_forget[key] for key in keys}
    baseline_data = baseline_payload.get("evaluation_dataset")
    comparison_data = comparison_payload.get("evaluation_dataset")
    if isinstance(baseline_data, Mapping) or isinstance(comparison_data, Mapping):
        if not isinstance(baseline_data, Mapping) or not isinstance(comparison_data, Mapping):
            raise ValueError("Cannot pair a retention result with a non-retention result")
        if baseline_data.get("records_fingerprint") != comparison_data.get("records_fingerprint"):
            raise ValueError("Retention results were evaluated on different dataset artifacts")
        return {"records_fingerprint": baseline_data["records_fingerprint"]}
    return {}


def compare_result_files(
    baseline_path: str | Path,
    comparison_path: str | Path,
    *,
    metric: str,
    seed: int,
    cluster_key: str = "record_id",
    item_key: str = "example_id",
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Load two result JSON files and return a provenance-bound paired comparison."""

    baseline_payload = read_json(baseline_path)
    comparison_payload = read_json(comparison_path)
    if not isinstance(baseline_payload, Mapping) or not isinstance(comparison_payload, Mapping):
        raise ValueError("Evaluation result files must be JSON objects")
    baseline_rows = baseline_payload.get("records")
    comparison_rows = comparison_payload.get("records")
    if not isinstance(baseline_rows, list) or not isinstance(comparison_rows, list):
        raise ValueError("Evaluation result files must contain a records list")
    result = paired_cluster_metric_comparison(
        baseline_rows,
        comparison_rows,
        metric=metric,
        seed=seed,
        cluster_key=cluster_key,
        item_key=item_key,
        bootstrap_samples=bootstrap_samples,
    )
    result.update(
        {
            "baseline_result_path": str(baseline_path),
            "comparison_result_path": str(comparison_path),
            "shared_data_provenance": _shared_data_provenance(
                baseline_payload, comparison_payload
            ),
            "baseline_checkpoint_id": _checkpoint_identity(baseline_payload),
            "comparison_checkpoint_id": _checkpoint_identity(comparison_payload),
        }
    )
    return result
