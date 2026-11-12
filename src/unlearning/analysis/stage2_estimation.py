"""Stage-2 sensitivity analyses and estimation-only tables (pre-registration 8.1, 8.2, 8.4 family 6, 8.5, 7).

None of these changes a registered decision; they are reported beside the tests.

* H1 sensitivity: t over per-seed means ("conditional on this question set"); linear
  interpolation to exactly g = 1 between f075 and f100; excluding seeds with any resumed run;
  the highest threshold every level of a seed reached.
* Size: loss as a fraction of the specialization gain (MedMCQA full dev, higher level minus its
  reference), "undefined" if the gain's 95% CI includes 0.
* 8.5 covariates and the family-6 regression: ordinary least squares on per-run mean losses
  with level/seed fixed effects.
* Family 6: S5 - S0 and S25 - S0 crossed estimates; the C_i - S_i and U - C_i decomposition;
  per-level forgetting, closeness, real-facts, recovery and degenerate-output summaries; steps to
  each threshold; the continuous co-measure; PubMedQA and MedMCQA full-dev losses.
* Section 7 reading rule: a level is specialized beyond format if its content-token dev loss is
  below both S0's and G25's, paired bootstrap CI over dev items excluding 0.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as _stats

from unlearning.analysis.stage2 import (
    METHODS, SUITES, Results, contaminated, crossed_test, question_contrast, unlearned,
)
from unlearning.analysis.stats import ANALYSIS_SEED, BOOTSTRAP_SAMPLES, crossed_anova, crossed_bootstrap, percentile_ci
from unlearning.evaluation.aggregate import is_degenerate
from unlearning.evaluation.forgetting import rescore_forgetting_payload
from unlearning.evaluation.multiple_choice import bias_removed_correctness, option_position_prior
from unlearning.utils.io import read_json

THRESHOLDS = ("f025", "f050", "f075", "f100")


def _mean_ci_over_seeds(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    out = {"n": len(values), "mean": float(np.mean(values)), "per_seed": values}
    if len(values) >= 2:
        half = _stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
        out["ci95_t_over_seeds"] = (out["mean"] - half, out["mean"] + half)
    return out


# --- H1 sensitivity ----------------------------------------------------------------------


def _snapshot_gap(results: Results, label: str) -> float | None:
    for suite in ("near", "forgetting"):
        payload = results.payload(label, suite)
        path = ((payload or {}).get("checkpoint_provenance") or {}).get("manifest_path")
        if path and Path(path).is_file():
            return (read_json(path).get("forgetting_progress") or {}).get("gap_closed")
    return None


def interpolated_contrast(results: Results, suite: str, method: str, *, high: str, low: str) -> dict[str, float] | None:
    """Per question, loss at exactly g = 1: linear in recorded g between the f075 and f100 snapshots."""

    def loss(clean):
        c75 = results.correctness(unlearned(clean, method, "f075"), suite)
        c100 = results.correctness(unlearned(clean, method, "f100"), suite)
        g75, g100 = _snapshot_gap(results, unlearned(clean, method, "f075")), _snapshot_gap(results, unlearned(clean, method, "f100"))
        s = results.correctness(clean, suite)
        if None in (c75, c100, g75, g100, s) or g100 == g75:
            return None
        w = (1.0 - g75) / (g100 - g75)
        return {q: c75[q] + w * (c100[q] - c75[q]) - s[q] for q in s}

    a, b = loss(high), loss(low)
    return None if a is None or b is None else {q: a[q] - b[q] for q in a}


def _resumed(results: Results, labels: list[str]) -> bool:
    """True if any manifest behind these results records more than one execution segment."""

    for label in labels:
        for suite in ("near", "forgetting"):
            payload = results.payload(label, suite)
            manifest = ((payload or {}).get("checkpoint_provenance") or {}).get("manifest") or {}
            execution = manifest.get("execution") or {}
            if execution.get("resumed") or (execution.get("segment_count") or 1) > 1:
                return True
    return False


def h1_sensitivity(results_by_seed: dict[int, Results]) -> dict[str, Any]:
    out = {}
    for method in METHODS:
        for suite in SUITES:
            base = {s: c for s, r in results_by_seed.items()
                    if (c := question_contrast(r, suite, method, high="S100", low="S0", reference="S")) is not None}
            entry: dict[str, Any] = {}
            means = [float(np.mean(list(base[s].values()))) for s in sorted(base)]
            entry["t_over_seed_means"] = _mean_ci_over_seeds(means)
            interp = {s: c for s, r in results_by_seed.items()
                      if (c := interpolated_contrast(r, suite, method, high="S100", low="S0")) is not None}
            entry["interpolated_g1"] = _strip(crossed_test(interp))
            kept = {s: c for s, c in base.items() if not _resumed(results_by_seed[s], [
                "S100", "S0", contaminated("S100"), contaminated("S0"), unlearned("S100", method), unlearned("S0", method)])}
            entry["excluding_resumed"] = _strip(crossed_test(kept)) | {"seeds_kept": sorted(kept)}
            common = {}
            for s, r in results_by_seed.items():
                for point in reversed(THRESHOLDS):
                    contrast = question_contrast(r, suite, method, high="S100", low="S0", reference="S", point=point)
                    if contrast is not None:
                        common[s] = contrast
                        break
            entry["highest_common_threshold"] = _strip(crossed_test(common))
            out[f"{method}/{suite}"] = entry
    return out


def _strip(test: dict[str, Any]) -> dict[str, Any]:
    """Crossed estimate with a 95% CI (wider of t and bootstrap), no decision."""

    out = {k: v for k, v in test.items() if not k.startswith("_")}
    if test.get("estimable"):
        out["ci95"] = (min(test["_estimate"].ci(0.95)[0], percentile_ci(test["_boot"], 0.95)[0]),
                       max(test["_estimate"].ci(0.95)[1], percentile_ci(test["_boot"], 0.95)[1]))
    return out


# --- size relative to the specialization gain (8.2) --------------------------------------------


def specialization_gain(results_by_seed: dict[int, Results], high: str, low: str) -> dict[str, Any]:
    per_seed = {}
    for s, r in results_by_seed.items():
        a, b = r.correctness(high, "domain_full"), r.correctness(low, "domain_full")
        if a is not None and b is not None:
            per_seed[s] = {q: a[q] - b[q] for q in a}
    test = crossed_test(per_seed)
    out = _strip(test)
    if test.get("estimable"):
        out["defined"] = not (out["ci95"][0] <= 0 <= out["ci95"][1])
    return out


# --- OLS with fixed effects (8.5, family 6 regression) ------------------------------------------


def ols(y: list[float], columns: dict[str, list[float]]) -> dict[str, float]:
    x = np.column_stack([np.ones(len(y))] + [np.asarray(v, float) for v in columns.values()])
    coef, *_ = np.linalg.lstsq(x, np.asarray(y, float), rcond=None)
    return {"intercept": float(coef[0]), **{name: float(c) for name, c in zip(columns, coef[1:])}}


def _dummies(values: list, reference) -> dict[str, list[float]]:
    return {f"is_{v}": [float(x == v) for x in values] for v in sorted(set(values), key=str) if v != reference}


def covariate_sensitivity(results_by_seed: dict[int, Results], levels: list[str]) -> dict[str, Any]:
    out = {}
    for method in METHODS:
        for suite in SUITES:
            rows = []
            for s, r in results_by_seed.items():
                for level in levels:
                    u, c = r.correctness(unlearned(level, method), suite), r.correctness(level, suite)
                    if u is None or c is None:
                        continue
                    rows.append({"seed": s, "level": level, "loss": float(np.mean([u[q] - c[q] for q in c])),
                                 "recall": _recall_at_stop(r, level), "heldout_gap": _heldout_gap(r, level, method)})
            if len({row["seed"] for row in rows}) < 2 or "S100" not in {row["level"] for row in rows}:
                out[f"{method}/{suite}"] = {"estimable": False}
                continue
            base = {**_dummies([row["level"] for row in rows], "S0"), **_dummies([row["seed"] for row in rows], rows[0]["seed"])}
            entry = {"without": ols([row["loss"] for row in rows], base)["is_S100"]}
            for name in ("recall", "heldout_gap"):
                if all(row[name] is not None for row in rows):
                    entry[f"with_{name}"] = ols([row["loss"] for row in rows], {**base, name: [row[name] for row in rows]})["is_S100"]
            out[f"{method}/{suite}"] = entry
    return out


def _recall_at_stop(results: Results, level: str) -> float | None:
    payload = results.payload(contaminated(level), "near")
    path = ((payload or {}).get("checkpoint_provenance") or {}).get("manifest_path")
    if not path or not Path(path).is_file():
        return None
    history = (read_json(path).get("memorization") or {}).get("history") or []
    return float(history[-1]["mean_recall"]) if history else None


def _forgetting_summary(results: Results, label: str) -> dict | None:
    payload = results.payload(label, "forgetting")
    return None if payload is None else rescore_forgetting_payload(payload)["summary"]


def _heldout_gap(results: Results, level: str, method: str, point: str = "f100") -> float | None:
    u, c, s = (_forgetting_summary(results, label) for label in (unlearned(level, method, point), contaminated(level), level))
    if None in (u, c, s) or c["target_logprob"] == s["target_logprob"]:
        return None
    return (c["target_logprob"] - u["target_logprob"]) / (c["target_logprob"] - s["target_logprob"])


def _decoy_gap(results: Results, level: str, method: str, point: str = "f100") -> float | None:
    u, c, s = (_forgetting_summary(results, label) for label in (unlearned(level, method, point), contaminated(level), level))
    if None in (u, c, s) or c["target_vs_decoy_margin"] == s["target_vs_decoy_margin"]:
        return None
    return (c["target_vs_decoy_margin"] - u["target_vs_decoy_margin"]) / (c["target_vs_decoy_margin"] - s["target_vs_decoy_margin"])


def regression_on_specialization(results_by_seed: dict[int, Results], levels: list[str]) -> dict[str, Any]:
    out = {}
    for method in METHODS:
        for suite in SUITES:
            y, x, seeds = [], [], []
            for s, r in results_by_seed.items():
                for level in (lvl for lvl in levels if lvl.startswith("S")):
                    u, c, m = r.correctness(unlearned(level, method), suite), r.correctness(level, suite), r.payload(level, "manipulation")
                    if u is None or c is None or m is None:
                        continue
                    y.append(float(np.mean([u[q] - c[q] for q in c])))
                    x.append(float(m["summary"]["content_dev_loss"]))
                    seeds.append(s)
            out[f"{method}/{suite}"] = (
                {"slope_per_nat": ols(y, {"content_dev_loss": x, **_dummies(seeds, seeds[0])})["content_dev_loss"], "points": len(y)}
                if len(set(seeds)) >= 2 else {"estimable": False}
            )
    return out


# --- family 6 tables ---------------------------------------------------------------------


def co_measure(results: Results, label: str, suite: str) -> dict[str, float] | None:
    """Per question: letter-bias-free log-prob of the correct option minus the mean of the others."""

    payload = results.payload(label, suite)
    if payload is None:
        return None
    records = payload["records"]
    prior = option_position_prior(records)
    out = {}
    for record in records:
        adjusted = [lp - prior[i] for i, lp in enumerate(record["option_logprobs"])]
        correct = record["answer_index"]
        others = [v for i, v in enumerate(adjusted) if i != correct]
        out[str(record["record_id"])] = adjusted[correct] - float(np.mean(others))
    return out


def estimation_tables(results_by_seed: dict[int, Results], levels: list[str]) -> dict[str, Any]:
    tables: dict[str, Any] = {"contrasts": {}, "decomposition": {}, "per_level": defaultdict(dict), "steps": {}, "co_measure": {}}
    for high in ("S5", "S25"):
        if high in levels:
            for method in METHODS:
                for suite in (*SUITES, "external_medical", "domain_full"):
                    per = {s: c for s, r in results_by_seed.items()
                           if (c := question_contrast(r, suite, method, high=high, low="S0", reference="S")) is not None}
                    tables["contrasts"][f"{high}-S0/{method}/{suite}"] = _strip(crossed_test(per))
    for level in levels:
        for suite in (*SUITES, "external_medical", "domain_full"):
            cs, uc = defaultdict(list), defaultdict(list)
            for s, r in results_by_seed.items():
                c_s, c_c = r.correctness(level, suite), r.correctness(contaminated(level), suite)
                if c_s is None or c_c is None:
                    continue
                cs["all"].append(float(np.mean([c_c[q] - c_s[q] for q in c_s])))
                for method in METHODS:
                    c_u = r.correctness(unlearned(level, method), suite)
                    if c_u is not None:
                        uc[method].append(float(np.mean([c_u[q] - c_c[q] for q in c_c])))
            tables["decomposition"][f"{level}/{suite}"] = {"C-S": _mean_ci_over_seeds(cs["all"]),
                                                           **{f"U-C/{m}": _mean_ci_over_seeds(uc[m]) for m in METHODS}}
        for method in METHODS:
            measures = defaultdict(list)
            for s, r in results_by_seed.items():
                summary = _forgetting_summary(r, unlearned(level, method))
                if summary is not None:
                    measures["target_exact_leakage"].append(summary["target_exact_leakage_rate"])
                    measures["any_target_leakage"].append(summary["any_target_exact_leakage_rate"])
                    measures["token_f1"].append(summary["normalized_field_f1"])
                    rows = rescore_forgetting_payload(r.payload(unlearned(level, method), "forgetting"))["records"]
                    texts = [row["generated_text"] for row in rows if row.get("generated_text") is not None]
                    measures["degenerate_rate"].append(float(np.mean([is_degenerate(t) for t in texts])) if texts else None)
                for name, value in (("heldout_gap_closed", _heldout_gap(r, level, method)), ("decoy_margin_gap_closed", _decoy_gap(r, level, method))):
                    if value is not None:
                        measures[name].append(value)
                for suite, metric in (("closeness", "kl_to_clean_answer_start"), ("closeness_nonf", "kl_to_clean_answer_start")):
                    payload = r.payload(unlearned(level, method), suite)
                    if payload is not None:
                        measures[f"{suite}"].append(payload["summary"][metric])
                facts = r.payload(unlearned(level, method), "real_facts")
                if facts is not None:
                    measures["real_facts_logprob"].append(facts["summary"]["mean_target_logprob"])
                for treatment in ("relearn", "benign"):
                    rec = r.payload(unlearned(level, method), f"recovery_{treatment}")
                    if rec is not None:
                        measures[f"recovery_{treatment}_rise"].append(rec["summary"]["mean_leakage_rise"])
                steps = {}
                for point in (*THRESHOLDS, "f_all", "final"):
                    payload = r.payload(unlearned(level, method, point), "near")
                    path = ((payload or {}).get("checkpoint_provenance") or {}).get("manifest_path")
                    if path and Path(path).is_file():
                        steps[point] = read_json(path).get("snapshot_step")
                tables["steps"][f"{level}/{method}/{s}"] = steps
            tables["per_level"][f"{level}/{method}"] = {name: _mean_ci_over_seeds([v for v in values if v is not None])
                                                        for name, values in measures.items()}
        for suite in SUITES:
            for method in METHODS:
                per = []
                for s, r in results_by_seed.items():
                    u, c = co_measure(r, unlearned(level, method), suite), co_measure(r, level, suite)
                    if u is not None and c is not None:
                        per.append(float(np.mean([u[q] - c[q] for q in c])))
                tables["co_measure"][f"{level}/{method}/{suite}"] = _mean_ci_over_seeds(per)
    tables["per_level"] = dict(tables["per_level"])
    return tables


def manipulation_reading(results_by_seed: dict[int, Results], levels: list[str]) -> dict[str, Any]:
    """Specialized beyond format: content-token dev loss below both S0's and G25's (paired item bootstrap)."""

    def item_losses(level):
        per_seed = [r.payload(level, "manipulation") for r in results_by_seed.values()]
        if any(p is None for p in per_seed):
            return None
        return np.mean(np.array([[v if v is not None else np.nan for v in p["per_item"]["content_dev_loss"]] for p in per_seed]), axis=0)

    out = {}
    references = {ref: item_losses(ref) for ref in ("S0", "G25") if ref in levels}
    rng = np.random.default_rng(ANALYSIS_SEED)
    for level in (lvl for lvl in levels if lvl.startswith("S") and lvl != "S0"):
        losses = item_losses(level)
        entry = {}
        for ref, ref_losses in references.items():
            if losses is None or ref_losses is None:
                continue
            diff = losses - ref_losses
            diff = diff[~np.isnan(diff)]
            boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOTSTRAP_SAMPLES)])
            entry[f"vs_{ref}"] = {"mean": float(diff.mean()), "ci95": percentile_ci(boot, 0.95)}
        if not entry or len(entry) < 2:
            # needs both references (S0 and G25) with results; otherwise the rule cannot be read
            out[level] = {**entry, "estimable": False}
            continue
        entry["specialized_beyond_format"] = all(v["ci95"][1] < 0 for v in entry.values())
        out[level] = {**entry, "estimable": True}
    return out
