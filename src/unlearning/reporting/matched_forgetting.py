"""Tables for unlearning revision 3 (matched forgetting), built only from results and manifests.

Writes ``results/processed/<run_id>/matched_forgetting_summary.md``:

1. steps each run needed to reach 25/50/75/100% of the gap to S_i (training probes);
2. every outcome at each matched point (held-out gap closed, leakage, decoy margin,
   degenerate outputs, KL to S_i, retention vs S_i);
3. what was forgotten: gap closed per fact type (probe template) and per patient
   (record), from the snapshot manifests and the run's references;
4. the primary estimand (docs/MATCHED_FORGETTING_DESIGN.md, 2): letter-bias-free
   retention loss vs S_i at 100%, with paired bootstrap CIs over the shared questions,
   the S50 - S0 and S100 - S0 contrasts, their bootstrap p-values and Holm-adjusted
   p-values over the 12 contrasts;
5. secondary: damage from unlearning alone (U - C_i) and the contamination's own cost
   (C_i - S_i), which add up to the primary loss;
6. probe sensitivity: multiple-choice association accuracy against chance, and free-text
   diagnosis leakage against a guessing baseline.

Measured values from one seed: descriptive, not confirmatory. The bootstrap fixes each
model's answer-position prior on the full set and resamples questions (paired).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from unlearning.config import load_config
from unlearning.config.unlearning_revisions import load_unlearning_revisions
from unlearning.evaluation.forgetting import rescore_forgetting_payload
from unlearning.evaluation.multiple_choice import bias_removed_correctness
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.run_log import utc_now
from unlearning.runtime.settings import REPO_ROOT, default_config_path
from unlearning.training.matched_forgetting import REFERENCES_FILE, per_example_references

LEVELS = ("0", "50", "100")
METHODS = ("ga", "npo")
RETENTION = ("near", "far", "general")
# 10,000 resamples: Monte Carlo error of a p near 0.05 is about ±0.004 (±0.01 at 2,000).
SAMPLES = 10000
FACT_TYPES = {"condition_lookup": "diagnosis", "medication_lookup": "medication", "private_code_lookup": "code"}
DIAGNOSIS_PROBE = "eval_paraphrase_condition"


# --- reading -------------------------------------------------------------------------------


def _metrics(processed: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    with (processed / "metrics_long.csv").open(encoding="utf-8") as handle:
        return {(row["model"], row["suite"], row["metric"]): row for row in csv.DictReader(handle)}


def _value(metrics, model: str, suite: str, metric: str) -> float | None:
    row = metrics.get((model, suite, metric))
    return float(row["value"]) if row and row["value"] not in ("", None) else None


def _payload(raw: Path, model: str, suite: str) -> dict[str, Any] | None:
    path = raw / model / f"{suite}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _correctness(raw: Path, model: str, suite: str) -> list[bool] | None:
    payload = _payload(raw, model, suite)
    return None if payload is None else bias_removed_correctness(payload["records"])


# --- statistics ----------------------------------------------------------------------------


def paired_bootstrap(differences, *, seed: int, samples: int = SAMPLES) -> tuple[float, float, float, float]:
    """Mean of per-question differences, percentile 95% CI over resampled questions, two-sided p.

    p = twice the smaller share of bootstrap means on either side of zero (capped at 1);
    0 means no resample crossed zero, i.e. p < 1/samples.
    """

    values = np.asarray(differences, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    chunks, remaining = [], samples
    while remaining:  # chunks keep memory small for 1,000-question suites
        size = min(remaining, 2000)
        chunks.append(values[rng.integers(0, n, size=(size, n))].mean(axis=1))
        remaining -= size
    estimates = np.sort(np.concatenate(chunks))
    p = min(1.0, 2.0 * min(float((estimates <= 0).mean()), float((estimates >= 0).mean())))
    return float(values.mean()), float(estimates[int(0.025 * samples)]), float(estimates[int(0.975 * samples) - 1]), p


def paired_ci(differences, *, seed: int, samples: int = SAMPLES) -> tuple[float, float, float]:
    mean, low, high, _ = paired_bootstrap(differences, seed=seed, samples=samples)
    return mean, low, high


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error), in the input order."""

    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


# --- formatting ----------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _fmt_p(p: float, samples: int = SAMPLES) -> str:
    return f"<{1 / samples:.4f}" if p == 0 else f"{p:.3f}"


def _ci_cell(stats: tuple | None) -> str:
    return "n/a" if stats is None else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*stats[:3])


# --- what was forgotten (manifests) ----------------------------------------------------------


def forgetting_composition(manifest_path: Path) -> dict[str, Any] | None:
    """Gap closed per fact type and per patient for one matched-forgetting snapshot.

    Per type: (l(C_t) - l(U_t)) / (l(C_t) - l(S_t)) on that template's mean, the same
    kernel as the stopping signal. Per patient: the share whose probe is at or below
    S_i's (``U <= S``, i.e. gap closed >= 1) per type, and the distribution of each
    patient's gap on its mean over the three probes. Patients whose C_i does not rate
    the answer above S_i have no gap and are excluded (counted).
    """

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    progress = json.loads(manifest_path.read_text(encoding="utf-8")).get("forgetting_progress")
    trajectory = manifest_path.parent.parent
    references_path = trajectory / REFERENCES_FILE
    if not progress or not references_path.is_file():
        return None
    references = json.loads(references_path.read_text(encoding="utf-8"))
    contaminated, clean = references["contaminated"]["by_template"], references["clean"]["by_template"]
    unlearned = progress["by_template"]
    composition: dict[str, Any] = {
        "by_type": {t: (contaminated[t] - unlearned[t]) / (contaminated[t] - clean[t]) for t in unlearned},
        "patients_at_clean": None,
        "patient_gap_quantiles": None,
        "patients": None,
        "patients_without_gap": None,
    }
    per_example = per_example_references(trajectory)
    if per_example is None or not progress.get("per_example"):
        return composition
    lp: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for role, values in (("U", progress["per_example"]), ("C", per_example["contaminated"]), ("S", per_example["clean"])):
        for key, value in values.items():
            record, template = key.split("::", 1)
            lp[record][template][role] = float(value)
    at_clean: dict[str, list[bool]] = defaultdict(list)
    gaps, without_gap = [], 0
    for templates in lp.values():
        for template, v in templates.items():
            if v["C"] > v["S"]:
                at_clean[template].append(v["U"] <= v["S"])
        c, u, s = (np.mean([v[role] for v in templates.values()]) for role in ("C", "U", "S"))
        if c > s:
            gaps.append((c - u) / (c - s))
        else:
            without_gap += 1
    composition["patients_at_clean"] = {t: float(np.mean(flags)) for t, flags in at_clean.items()}
    composition["patient_gap_quantiles"] = tuple(float(q) for q in np.percentile(gaps, [10, 50, 90])) if gaps else None
    composition["patients"] = len(lp)
    composition["patients_without_gap"] = without_gap
    return composition


def _manifest_path(raw: Path, model: str) -> Path | None:
    payload = _payload(raw, model, "forgetting")
    path = ((payload or {}).get("checkpoint_provenance") or {}).get("manifest_path")
    return Path(path) if path else None


# --- the report ----------------------------------------------------------------------------


def _loss_by_level(raw: Path, model_of: dict[str, str], reference_of: dict[str, str], suite: str):
    """Per-question (model - reference) letter-bias-free correctness, per level."""

    per_level = {}
    for level in LEVELS:
        model, reference = _correctness(raw, model_of[level], suite), _correctness(raw, reference_of[level], suite)
        per_level[level] = None if model is None or reference is None else [float(a) - float(b) for a, b in zip(model, reference)]
    return per_level


def _contrast_table(raw: Path, *, model_of, reference_of, seed: int, adjust: bool) -> list[str]:
    """Rows: per-level loss with CI, and C50 - C0 / C100 - C0 contrasts with p (and Holm)."""

    cells: dict[tuple[str, str], list[str]] = {}
    contrasts: list[tuple[tuple[str, str], int, float]] = []
    for method in METHODS:
        for suite in RETENTION:
            per_level = _loss_by_level(raw, model_of(method), reference_of, suite)
            row = [_ci_cell(None if per_level[level] is None else paired_ci(per_level[level], seed=seed)) for level in LEVELS]
            for level in ("50", "100"):
                if per_level[level] is None or per_level["0"] is None:
                    row.append("n/a")
                    continue
                stats = paired_bootstrap([a - b for a, b in zip(per_level[level], per_level["0"])], seed=seed)
                row.append(f"{_ci_cell(stats)} p={_fmt_p(stats[3])}")
                contrasts.append(((method, suite), len(row) - 1, stats[3]))
            cells[(method, suite)] = row
    if adjust and contrasts:
        for (key, column, _), adjusted in zip(contrasts, holm([p for *_, p in contrasts])):
            cells[key][column] += f", Holm {_fmt_p(adjusted)}"
    return [f"| {method.upper()} | {suite} | " + " | ".join(cells[(method, suite)]) + " |" for method, suite in cells]


def build_summary(
    raw: Path,
    processed: Path,
    run_id: str,
    *,
    tag: str,
    names: tuple[str, ...],
    seed: int,
    mc_chance: float | None = None,
    figure: Path | None = None,
) -> str:
    metrics = _metrics(processed)
    final = names[-1]
    lines = [
        f"# Matched-forgetting unlearning ({tag}): measured results",
        "",
        f"Run `{run_id}`. Generated {utc_now()} from results/raw, results/processed and snapshot manifests.",
        "One seed, Qwen2.5-0.5B: descriptive pilot results, not confirmatory. Retention is letter-bias-free",
        "accuracy; differences are vs the level's own clean model S_i unless stated.",
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

    lines += [
        "",
        "## What was forgotten: gap closed per fact type (training probes)",
        "",
        "The stopping signal averages three fact types. Per type, 1 = that type's mean log-prob equals S_i's;",
        "values above 1 overshoot the clean model. Exploratory for this seed: the breakdown was examined after",
        "the first results.",
        "",
        "| run | point | " + " | ".join(FACT_TYPES.values()) + " | overall (stopping signal) |",
        "|---|---|" + "---|" * (len(FACT_TYPES) + 1),
    ]
    compositions = {}
    for level in LEVELS:
        for method in METHODS:
            for name in names:
                model = f"MR_{tag}_C{level}_{method}_{name}"
                manifest = _manifest_path(raw, model)
                composition = forgetting_composition(manifest) if manifest else None
                compositions[model] = composition
                by_type = (composition or {}).get("by_type") or {}
                lines.append(
                    f"| {method.upper()} C{level} | {name} | "
                    + " | ".join(_fmt(by_type.get(t), 2) for t in FACT_TYPES)
                    + f" | {_fmt(_value(metrics, model, 'forgetting', 'training_probe_gap_closed'), 2)} |"
                )
    lines += [
        "",
        f"Per patient at {final}: share of patients whose probe is at or below S_i's (forgotten to the clean",
        "level) per fact type, and the 10th / 50th / 90th percentile of each patient's gap closed (mean of its",
        "three probes).",
        "",
        "| run | " + " | ".join(f"{label} at clean" for label in FACT_TYPES.values()) + " | patient gap p10 / p50 / p90 | patients |",
        "|---|" + "---|" * (len(FACT_TYPES) + 2),
    ]
    for level in LEVELS:
        for method in METHODS:
            composition = compositions.get(f"MR_{tag}_C{level}_{method}_{final}") or {}
            at_clean = composition.get("patients_at_clean") or {}
            quantiles = composition.get("patient_gap_quantiles")
            excluded = composition.get("patients_without_gap")
            lines.append(
                f"| {method.upper()} C{level} | "
                + " | ".join(_fmt(at_clean.get(t), 2) for t in FACT_TYPES)
                + f" | {'n/a' if quantiles is None else ' / '.join(f'{q:.2f}' for q in quantiles)} | "
                + f"{composition.get('patients') or 'n/a'}{f' ({excluded} without a gap)' if excluded else ''} |"
            )

    lines += [
        "",
        f"## Primary estimand: retention loss vs S_i at {final} (paired bootstrap over questions, 95% CI)",
        "",
        "Contrasts carry a two-sided bootstrap p and a Holm-adjusted p over all 12 contrasts. No",
        "multiple-comparison rule was registered for this pilot; Holm shows how the contrasts fare under a",
        "standard family-wise correction.",
        "",
        "| method | suite | C0 | C50 | C100 | C50 - C0 | C100 - C0 |",
        "|---|---|---|---|---|---|---|",
    ]
    clean_of = {level: f"S{level}" for level in LEVELS}
    contaminated_of = {level: f"MR_C{level}" for level in LEVELS}
    unlearned_of = lambda method: {level: f"MR_{tag}_C{level}_{method}_{final}" for level in LEVELS}  # noqa: E731
    lines += _contrast_table(raw, model_of=unlearned_of, reference_of=clean_of, seed=seed, adjust=True)

    lines += [
        "",
        f"## Secondary: damage from unlearning alone (U - C_i at {final}) and contamination's cost (C_i - S_i)",
        "",
        "Primary loss = (C_i - S_i) + (U - C_i). The first part was there before unlearning started.",
        "",
        "| method | suite | C0 | C50 | C100 | C50 - C0 | C100 - C0 |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += _contrast_table(raw, model_of=unlearned_of, reference_of=contaminated_of, seed=seed, adjust=False)
    lines += ["", "| contamination | suite | C0 | C50 | C100 |", "|---|---|---|---|---|"]
    for suite in RETENTION:
        per_level = _loss_by_level(raw, contaminated_of, clean_of, suite)
        cells = [_ci_cell(None if per_level[level] is None else paired_ci(per_level[level], seed=seed)) for level in LEVELS]
        lines.append(f"| C_i - S_i | {suite} | " + " | ".join(cells) + " |")

    lines += _probe_sensitivity(raw, metrics, tag=tag, final=final, mc_chance=mc_chance)
    if figure is not None:
        lines += ["", "## Figure", "", f"Forgetting-retention curves: `{figure}`"]
    lines += [
        "",
        "Δ near/far/general = letter-bias-free accuracy minus S_i's. Held-out gap closed = (lp_C - lp_U) / (lp_C - lp_S)",
        "on held-out forgetting templates (1 = the clean model's mean target log-prob). Degenerate = share of",
        "generations with distinct character 4-gram ratio < 0.2. KL = mean KL(S_i || model) over the first 16 answer",
        "positions of S_i's own continuation (0 = predicts like S_i). Near = MedMCQA medicine and pharmacology;",
        "far = other MedMCQA subjects (still medical); general = MMLU-CF (non-medical).",
    ]
    return "\n".join(lines) + "\n"


def _probe_sensitivity(raw: Path, metrics, *, tag: str, final: str, mc_chance: float | None) -> list[str]:
    lines = [
        "",
        "## Probe sensitivity",
        "",
        "A probe can only show forgetting if the contaminated model passes it. Multiple-choice association",
        f"accuracy (chance {_fmt(mc_chance, 2)}) next to free-text diagnosis leakage on the held-out paraphrase probe.",
        "",
        "| model | MC association accuracy | diagnosis leakage (free text) |",
        "|---|---|---|",
    ]
    guessing = None
    for level in LEVELS:
        for model in (f"S{level}", f"MR_C{level}", *(f"MR_{tag}_C{level}_{m}_{final}" for m in METHODS)):
            payload = _payload(raw, model, "forgetting")
            leakage = None
            if payload is not None:
                rescored = rescore_forgetting_payload(payload)
                leakage = rescored["summary"]["by_template"].get(DIAGNOSIS_PROBE, {}).get("target_exact_leakage_rate")
                if guessing is None:
                    targets = [row["target"] for row in rescored["records"] if row["template_id"] == DIAGNOSIS_PROBE]
                    if targets:
                        counts = Counter(targets)
                        guessing = (len(counts), max(counts.values()) / len(targets))
            lines.append(f"| {model} | {_fmt(_value(metrics, model, 'forgetting', 'mc_association_accuracy'))} | {_fmt(leakage)} |")
    if guessing is not None:
        lines += [
            "",
            f"F has {guessing[0]} distinct diagnoses; always naming the most common one would leak for "
            f"{guessing[1]:.2f} of patients (the guessing baseline for diagnosis leakage).",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize matched-forgetting (revision 3) results.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--unblind", action="store_true", help="Stage 2 only: allowed after every seed is done (cross-seed analysis).")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.data_order == "seeded" and not args.unblind:
        # Pre-registration, section 10: Stage-2 results stay sealed until all seeds are done.
        raise SystemExit("Stage-2 results are blinded; the cross-seed analysis unblinds them (--unblind).")
    layout = StorageLayout.for_config(config)
    revision = next(r for r in load_unlearning_revisions(REPO_ROOT / "configs" / "pilot" / "unlearning") if r.is_matched)
    figure = None
    if not args.no_figure:
        from unlearning.reporting.matched_forgetting_figure import plot_forgetting_retention

        figure = plot_forgetting_retention(
            layout.raw_results, layout.processed_results, layout.figures / "matched_forgetting",
            tag=revision.tag, names=revision.matched.snapshot_names, seed=config.seed,
        )
    text = build_summary(
        layout.raw_results, layout.processed_results, layout.run_id,
        tag=revision.tag, names=revision.matched.snapshot_names, seed=config.seed,
        mc_chance=1.0 / (config.forget.mc_decoy_count + 1), figure=figure,
    )
    output = layout.processed_results / "matched_forgetting_summary.md"
    if output.is_file():
        # Keep the previous version: summaries are regenerated, never silently replaced.
        previous = output.with_name(f"matched_forgetting_summary__{output.stat().st_mtime_ns}.md")
        output.replace(previous)
        print(f"previous summary kept as {previous}")
    output.write_text(text, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
