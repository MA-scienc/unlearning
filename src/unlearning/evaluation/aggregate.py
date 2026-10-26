"""Collect raw evaluation JSONs into tidy, analysis-ready tables.

Reads only ``results/raw/<run_id>/<model>/<suite>.json`` and writes, under
``results/processed/<run_id>/``:

``metrics_long.csv``  one row per (model, suite, metric) -- the input for plots
                      and statistics
``summary.md``        headline metrics per model (measured values, no inference)

Raw files are never modified. Paired, record-level comparisons (U_i vs C_i and
S_i) belong in ``compare_paired.py``, which works on the raw per-record outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.evaluation.forgetting import rescore_forgetting_payload
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.run_log import utc_now
from unlearning.runtime.settings import default_config_path
from unlearning.utils.io import ensure_dir

_LABEL = re.compile(
    r"^(?:(?P<arm>[A-Z]+)_)?"
    r"(?:(?P<clean>S)(?P<s>\d+)|(?P<cont>C)(?P<c>\d+)|U_C(?P<u>\d+)_(?P<method>[a-z]+)_step(?P<step>\d+))$"
)
ARM_BY_TAG = {"MM": "matched_memorization"}
RETENTION_METRICS = ("accuracy", "mean_nll", "mean_correct_logprob")
FORGETTING_METRICS = (
    "target_exact_leakage_rate",
    "any_target_exact_leakage_rate",
    "nonprompt_field_leakage_rate",
    "any_nonprompt_field_leakage_rate",
    # Includes echo of identifiers the probe supplies; kept for transparency only.
    "exact_leakage_rate",
    "any_exact_field_leakage_rate",
    "normalized_field_f1",
    "target_logprob",
    "target_vs_decoy_margin",
    "mc_label_protected_vs_decoy_margin",
    "mc_association_accuracy",
)
FIELDS = (
    "run_id",
    "model",
    "family",
    "arm",
    "specialization_level",
    "method",
    "unlearning_step",
    "suite",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "source_file",
)


def parse_label(label: str) -> dict[str, Any]:
    match = _LABEL.match(label)
    if not match:
        raise ValueError(f"Unrecognized model label {label!r}")
    tag = match.group("arm")
    if tag is not None and tag not in ARM_BY_TAG:
        raise ValueError(f"Unknown contamination arm tag {tag!r} in {label!r}")
    arm = ARM_BY_TAG.get(tag, "fixed_exposure")
    if match.group("clean"):
        if tag is not None:
            raise ValueError(f"Clean checkpoints have no contamination arm: {label!r}")
        return {"family": "clean", "arm": None, "specialization_level": int(match.group("s")), "method": None, "unlearning_step": None}
    if match.group("cont"):
        return {"family": "contaminated", "arm": arm, "specialization_level": int(match.group("c")), "method": None, "unlearning_step": 0}
    return {
        "family": "unlearned",
        "arm": arm,
        "specialization_level": int(match.group("u")),
        "method": match.group("method"),
        "unlearning_step": int(match.group("step")),
    }


def rows_for_file(run_id: str, path: Path) -> list[dict[str, Any]]:
    label = path.parent.name
    suite = path.stem
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = {"run_id": run_id, "model": label, "suite": suite, "source_file": str(path), **parse_label(label)}
    rows = []
    if suite == "forgetting":
        # Always score from the saved generations with the current leakage
        # definition, so results written before it existed stay comparable.
        summary = rescore_forgetting_payload(payload)["summary"]
        for metric in FORGETTING_METRICS:
            rows.append({**base, "metric": metric, "value": summary.get(metric), "ci_low": None, "ci_high": None, "n": summary.get("record_count")})
    else:
        ci = payload.get("accuracy_ci95") or [None, None]
        for metric in RETENTION_METRICS:
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "value": payload.get(metric),
                    "ci_low": ci[0] if metric == "accuracy" else None,
                    "ci_high": ci[1] if metric == "accuracy" else None,
                    "n": payload.get("count"),
                }
            )
    return rows


def _sort_key(row: dict[str, Any]):
    order = {"clean": 0, "contaminated": 1, "unlearned": 2}
    return (
        order[row["family"]],
        row["arm"] or "",
        row["specialization_level"],
        row["method"] or "",
        row["unlearning_step"] if row["unlearning_step"] is not None else -1,
        row["suite"],
        row["metric"],
    )


def aggregate(raw_dir: Path, processed_dir: Path, run_id: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*/*.json")):
        rows.extend(rows_for_file(run_id, path))
    rows.sort(key=_sort_key)
    ensure_dir(processed_dir)
    with (processed_dir / "metrics_long.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (processed_dir / "summary.md").write_text(_summary_markdown(rows, run_id), encoding="utf-8")
    return {"rows": len(rows), "models": len({row["model"] for row in rows}), "processed_dir": str(processed_dir)}


def _summary_markdown(rows: list[dict[str, Any]], run_id: str) -> str:
    headline = {
        "forgetting": "target_exact_leakage_rate",
        "near": "accuracy",
        "far": "accuracy",
        "general": "accuracy",
        "external_medical": "accuracy",
        "domain_full": "accuracy",
    }
    suites = list(headline)
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        if headline.get(row["suite"]) == row["metric"]:
            table.setdefault(row["model"], {})[row["suite"]] = row["value"]
    lines = [
        f"# Measured results: {run_id}",
        "",
        f"Generated {utc_now()} from results/raw. Single-seed pilot; descriptive only.",
        "Columns: forgetting = rate at which the probed target appears in the output "
        "(lower = more forgotten); others = accuracy.",
        "",
        "| model | " + " | ".join(suites) + " |",
        "|---|" + "---|" * len(suites),
    ]
    for model, values in table.items():
        cells = [f"{values[suite]:.3f}" if isinstance(values.get(suite), (int, float)) else "" for suite in suites]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build results/processed tables from results/raw.")
    parser.add_argument("--config", default=default_config_path())
    args = parser.parse_args(argv)
    config = load_config(args.config)
    layout = StorageLayout.for_config(config)
    summary = aggregate(layout.raw_results, layout.processed_results, layout.run_id)
    from unlearning.reporting.training_curves import plot_all

    summary["training_figures"] = [str(path) for path in plot_all(layout.runs, layout.figures)]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
