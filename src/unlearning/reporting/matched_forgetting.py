"""Tables for unlearning revision 3 (matched forgetting), built only from results and manifests.

Writes ``results/processed/<run_id>/matched_forgetting_summary.md``:

1. steps each run needed to reach 25/50/75/100% of the gap to S_i (training probes);
2. every outcome at each matched point (held-out gap closed, leakage, decoy margin,
   degenerate outputs, KL to S_i, retention vs S_i);
3. the primary estimand (docs/MATCHED_FORGETTING_DESIGN.md, 2): letter-bias-free
   retention loss vs S_i at 100%, with paired bootstrap CIs over the shared questions,
   and the S50 - S0 and S100 - S0 contrasts of that loss.

Measured values from one seed: descriptive, not confirmatory. The bootstrap fixes each
model's answer-position prior on the full set and resamples questions (paired).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.config.unlearning_revisions import load_unlearning_revisions
from unlearning.evaluation.multiple_choice import bias_removed_correctness
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.run_log import utc_now
from unlearning.runtime.settings import REPO_ROOT, default_config_path

LEVELS = ("0", "50", "100")
METHODS = ("ga", "npo")
RETENTION = ("near", "far", "general")
SAMPLES = 2000


def _metrics(processed: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    with (processed / "metrics_long.csv").open(encoding="utf-8") as handle:
        return {(row["model"], row["suite"], row["metric"]): row for row in csv.DictReader(handle)}


def _value(metrics, model: str, suite: str, metric: str) -> float | None:
    row = metrics.get((model, suite, metric))
    return float(row["value"]) if row and row["value"] not in ("", None) else None


def _correctness(raw: Path, model: str, suite: str) -> list[bool] | None:
    path = raw / model / f"{suite}.json"
    if not path.is_file():
        return None
    return bias_removed_correctness(json.loads(path.read_text(encoding="utf-8"))["records"])


def paired_ci(differences: list[float], *, seed: int, samples: int = SAMPLES) -> tuple[float, float, float]:
    """Mean of per-question differences with a percentile bootstrap over questions."""

    rng = random.Random(seed)
    n = len(differences)
    estimates = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return sum(differences) / n, estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]


def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def build_summary(raw: Path, processed: Path, run_id: str, *, tag: str, names: tuple[str, ...], seed: int) -> str:
    metrics = _metrics(processed)
    lines = [
        f"# Matched-forgetting unlearning ({tag}): measured results",
        "",
        f"Run `{run_id}`. Generated {utc_now()} from results/raw, results/processed and snapshot manifests.",
        "One seed, Qwen2.5-0.5B: descriptive pilot results, not confirmatory. Retention is letter-bias-free",
        "accuracy; differences are vs the level's own clean model S_i.",
        "",
        "## Steps to each threshold (gap closed on training-template probes)",
        "",
        "| run | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for level in LEVELS:
        for method in METHODS:
            cells = []
            for name in names:
                model = f"MR_{tag}_C{level}_{method}_{name}"
                step, gap = _value(metrics, model, "forgetting", "snapshot_step"), _value(metrics, model, "forgetting", "training_probe_gap_closed")
                cells.append("not reached" if step is None else f"{int(step)} ({gap:.3f})")
            lines.append(f"| {method.upper()} C{level} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Outcomes at each matched point",
        "",
        "| run | point | held-out gap closed | exact leakage | any-target leakage | decoy margin | degenerate | KL to S_i | "
        "Δ near | Δ far | Δ general |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for level in LEVELS:
        clean, contaminated = f"S{level}", f"MR_C{level}"
        rows = [(clean, "S_i"), (contaminated, "C_i (0%)")]
        rows += [(f"MR_{tag}_C{level}_{method}_{name}", f"{method.upper()} {name}") for method in METHODS for name in names]
        for model, point in rows:
            deltas = []
            for suite in RETENTION:
                value, reference = _value(metrics, model, suite, "bias_removed_accuracy"), _value(metrics, clean, suite, "bias_removed_accuracy")
                deltas.append(None if value is None or reference is None else value - reference)
            lines.append(
                f"| C{level} | {point} | {_fmt(_value(metrics, model, 'forgetting', 'heldout_gap_closed'), 2)} | "
                f"{_fmt(_value(metrics, model, 'forgetting', 'target_exact_leakage_rate'))} | "
                f"{_fmt(_value(metrics, model, 'forgetting', 'any_target_exact_leakage_rate'))} | "
                f"{_fmt(_value(metrics, model, 'forgetting', 'target_vs_decoy_margin'), 2)} | "
                f"{_fmt(_value(metrics, model, 'forgetting', 'degenerate_output_rate'))} | "
                f"{_fmt(_value(metrics, model, 'closeness', 'kl_to_clean_answer_start'), 2)} | "
                + " | ".join(_fmt(delta, signed=True) for delta in deltas)
                + " |"
            )

    final = names[-1]
    lines += [
        "",
        f"## Primary estimand: retention loss vs S_i at {final} (paired bootstrap over questions, 95% CI)",
        "",
        "| method | suite | C0 | C50 | C100 | C50 - C0 | C100 - C0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        for suite in RETENTION:
            per_level = {}
            for level in LEVELS:
                unlearned = _correctness(raw, f"MR_{tag}_C{level}_{method}_{final}", suite)
                clean = _correctness(raw, f"S{level}", suite)
                if unlearned is None or clean is None:
                    per_level[level] = None
                    continue
                per_level[level] = [float(u) - float(c) for u, c in zip(unlearned, clean)]
            cells = []
            for level in LEVELS:
                diffs = per_level[level]
                cells.append("n/a" if diffs is None else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*paired_ci(diffs, seed=seed)))
            for level in ("50", "100"):
                if per_level[level] is None or per_level["0"] is None:
                    cells.append("n/a")
                    continue
                contrast = [a - b for a, b in zip(per_level[level], per_level["0"])]
                cells.append("{:+.3f} [{:+.3f}, {:+.3f}]".format(*paired_ci(contrast, seed=seed)))
            lines.append(f"| {method.upper()} | {suite} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Δ near/far/general = letter-bias-free accuracy minus S_i's. Held-out gap closed = (lp_C - lp_U) / (lp_C - lp_S)",
        "on held-out forgetting templates (1 = the clean model's mean target log-prob). Degenerate = share of",
        "generations with distinct character 4-gram ratio < 0.2. KL = mean KL(S_i || model) over the first 16 answer",
        "positions of S_i's own continuation (0 = predicts like S_i).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize matched-forgetting (revision 3) results.")
    parser.add_argument("--config", default=default_config_path())
    args = parser.parse_args(argv)
    config = load_config(args.config)
    layout = StorageLayout.for_config(config)
    revision = next(r for r in load_unlearning_revisions(REPO_ROOT / "configs" / "pilot" / "unlearning") if r.is_matched)
    text = build_summary(
        layout.raw_results, layout.processed_results, layout.run_id,
        tag=revision.tag, names=revision.matched.snapshot_names, seed=config.seed,
    )
    output = layout.processed_results / "matched_forgetting_summary.md"
    output.write_text(text, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
