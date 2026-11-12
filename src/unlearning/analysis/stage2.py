"""Cross-seed Stage-2 analysis (docs/PREREGISTRATION_STAGE2.md, section 8).

Reads every seed's raw results and manifests and applies the registered procedures in
``analysis/stats.py``. Writes ``results/processed/stage2/analysis.json`` and ``analysis.md``.

Families: H1 (primary; 6 tests), H2 (4), and the secondary families 1a, 1b, 1c, 2, 3, 4 and
5, each its own Holm family; family 6 is estimation only. A contrast whose required results
are missing for a seed is complete-case; a test needs >= 2 complete seeds (H1 needs all
requested seeds), otherwise it is reported "not estimable". Every result file must record
committed code (``git_dirty: false``); ``allow_unversioned`` relaxes that for dry runs on
the pilot and is recorded in the output.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unlearning.analysis.stats import (
    ALPHA, ANALYSIS_SEED, BOOTSTRAP_SAMPLES, bootstrap_p, crossed_anova, crossed_bootstrap, hierarchical_bootstrap,
    holm, holm_levels, percentile_ci, symmetric_bound,
)
from unlearning.config import load_config
from unlearning.evaluation.forgetting import exact_field_leakage, rescore_forgetting_payload
from unlearning.evaluation.multiple_choice import bias_removed_correctness
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.settings import REPO_ROOT
from unlearning.utils.code_version import code_version
from unlearning.utils.io import read_json, write_json

ARM, REVISION = "MR", "U3"
METHODS = ("ga", "npo")
SUITES = ("near", "far", "general")
POINT = "f100"
DIAGNOSIS_PROBE = "eval_paraphrase_condition"
CODE, DIAGNOSIS = "private_code_lookup", "condition_lookup"


@dataclass(frozen=True)
class SeedRun:
    seed: int
    run_id: str
    raw: Path
    levels: tuple[str, ...]  # clean names, e.g. S0, S5, S25, S100, G25


def load_runs(config_paths: list[Path]) -> list[SeedRun]:
    from unlearning.training.levels import clean_names

    runs = []
    for path in config_paths:
        config = load_config(path)
        layout = StorageLayout.for_config(config)
        runs.append(SeedRun(config.seed, layout.run_id, layout.raw_results, clean_names(config)))
    return runs


def contaminated(clean: str) -> str:
    return f"{ARM}_C{clean[1:]}" if clean.startswith("S") else f"{ARM}_C{clean}"


def unlearned(clean: str, method: str, point: str = POINT) -> str:
    level = clean[1:] if clean.startswith("S") else clean
    return f"{ARM}_{REVISION}_C{level}_{method}_{point}"


class Results:
    """Cached, provenance-checked access to one seed's raw result files."""

    def __init__(self, run: SeedRun, *, allow_unversioned: bool):
        self.run, self.allow_unversioned, self._cache = run, allow_unversioned, {}
        self.unversioned: list[str] = []

    def payload(self, label: str, suite: str) -> dict | None:
        key = (label, suite)
        if key not in self._cache:
            path = self.run.raw / label / f"{suite}.json"
            payload = read_json(path) if path.is_file() else None
            if payload is not None and (payload.get("code_version") or {}).get("git_dirty") is not False:
                if not self.allow_unversioned:
                    raise ValueError(f"{path} does not record committed code (git_dirty false); refusing")
                self.unversioned.append(str(path))
            self._cache[key] = payload
        return self._cache[key]

    def correctness(self, label: str, suite: str) -> dict[str, float] | None:
        payload = self.payload(label, suite)
        if payload is None:
            return None
        records = payload["records"]
        return {str(r["record_id"]): float(ok) for r, ok in zip(records, bias_removed_correctness(records))}


# --- per-question contrasts (H1, families 1a-1c, 5, family 6) -------------------------------


def question_contrast(results: Results, suite: str, method: str, *, high: str, low: str, reference: str,
                      point: str = POINT) -> dict[str, float] | None:
    """Per question: loss(high) - loss(low); loss = c_U - c_reference(level), reference "S" or "C"."""

    def loss(clean: str):
        u = results.correctness(unlearned(clean, method, point), suite)
        r = results.correctness(clean if reference == "S" else contaminated(clean), suite)
        if u is None or r is None:
            return None
        return {q: u[q] - r[q] for q in u}

    a, b = loss(high), loss(low)
    if a is None or b is None:
        return None
    return {q: a[q] - b[q] for q in a}


def crossed_test(per_seed: dict[int, dict[str, float]]) -> dict[str, Any]:
    seeds = sorted(per_seed)
    if len(seeds) < 2:
        return {"estimable": False, "seeds": seeds, "per_seed": {s: float(np.mean(list(per_seed[s].values()))) for s in seeds}}
    questions = sorted(set.intersection(*(set(per_seed[s]) for s in seeds)))
    data = np.array([[per_seed[s][q] for s in seeds] for q in questions])
    estimate = crossed_anova(data)
    boot = crossed_bootstrap(data)
    return {
        "estimable": True, "seeds": seeds, "questions": len(questions), "mean": estimate.mean, "p": estimate.p,
        "df": estimate.df, "variance": estimate.variance, "truncated": list(estimate.truncated),
        "components": estimate.components, "_boot": boot, "_estimate": estimate,
        "per_seed": {s: float(data[:, i].mean()) for i, s in enumerate(seeds)},
        "seeds_same_sign": int(sum(np.sign(data[:, i].mean()) == np.sign(estimate.mean) for i in range(len(seeds)))),
    }


def finish_crossed_family(tests: dict[str, dict[str, Any]], *, require_all: int | None = None) -> None:
    """Holm over the estimable tests; decision = Holm p < alpha and bootstrap CI at the Holm level excludes 0."""

    names = [name for name, test in tests.items() if test["estimable"] and (require_all is None or len(test["seeds"]) >= require_all)]
    for name, test in tests.items():
        if name not in names:
            test["estimable"] = False
    if not names:
        return
    adjusted = holm([tests[n]["p"] for n in names])
    levels = holm_levels([tests[n]["p"] for n in names])
    for name, p_holm, level in zip(names, adjusted, levels):
        test = tests[name]
        boot, estimate = test.pop("_boot"), test.pop("_estimate")
        step_ci = percentile_ci(boot, level)
        ci95 = _wider(estimate.ci(0.95), percentile_ci(boot, 0.95))
        ci90 = _wider(estimate.ci(0.90), percentile_ci(boot, 0.90))
        test.update({
            "p_holm": p_holm, "holm_level": level, "bootstrap_ci_at_holm_level": step_ci,
            "significant": bool(p_holm < ALPHA and (step_ci[0] > 0 or step_ci[1] < 0)),
            "ci95": ci95, "ci90": ci90, "bound_b": symmetric_bound(ci90),
        })


def _wider(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]))


# --- per-patient measures (H2, H3, families 2-4) -------------------------------------------


def _manifest(results: Results, label: str) -> dict | None:
    payload = results.payload(label, "forgetting")
    path = ((payload or {}).get("checkpoint_provenance") or {}).get("manifest_path")
    return read_json(path) if path and Path(path).is_file() else None


def per_patient_type_logprobs(results: Results, clean: str, method: str) -> dict[str, dict[str, dict[str, float]]] | None:
    """{patient: {template: {"U", "C", "S"}}} at f100 from the snapshot manifest and the run's references."""

    from unlearning.training.matched_forgetting import per_example_references

    manifest_payload = results.payload(unlearned(clean, method), "forgetting")
    path = ((manifest_payload or {}).get("checkpoint_provenance") or {}).get("manifest_path")
    if not path or not Path(path).is_file():
        return None
    progress = read_json(path).get("forgetting_progress") or {}
    references = per_example_references(Path(path).parent.parent)
    if not progress.get("per_example") or references is None:
        return None
    table: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for role, values in (("U", progress["per_example"]), ("C", references["contaminated"]), ("S", references["clean"])):
        for key, value in values.items():
            patient, template = key.split("::", 1)
            table[patient].setdefault(template, {})[role] = float(value)
    return table


def gap_difference(table, patients: list[str]) -> float:
    """gap_code - gap_diagnosis, each a ratio of means over the given patients."""

    def gap(template):
        c = np.mean([table[p][template]["C"] for p in patients])
        u = np.mean([table[p][template]["U"] for p in patients])
        s = np.mean([table[p][template]["S"] for p in patients])
        return (c - u) / (c - s)

    return float(gap(CODE) - gap(DIAGNOSIS))


def diagnosis_margin(results: Results, label: str) -> dict[str, float] | None:
    """Per patient: own-diagnosis mention minus the mean mention of the other diagnoses (H2b)."""

    payload = results.payload(label, "forgetting")
    if payload is None:
        return None
    rows = [row for row in rescore_forgetting_payload(payload)["records"] if row["template_id"] == DIAGNOSIS_PROBE]
    diagnoses = sorted({row["target"] for row in rows})
    margins = {}
    for row in rows:
        text = row.get("generated_text") or ""
        own = float(exact_field_leakage(text, row["target"]))
        others = [float(exact_field_leakage(text, d)) for d in diagnoses if d != row["target"]]
        margins[str(row["record_id"])] = own - (sum(others) / len(others) if others else 0.0)
    return margins


def patient_kl(results: Results, label: str) -> dict[str, float] | None:
    payload = results.payload(label, "closeness")
    if payload is None:
        return None
    by_patient = defaultdict(list)
    for row in payload.get("records") or []:
        by_patient[str(row["record_id"])].append(float(row["kl_mean"]))
    return {p: float(np.mean(v)) for p, v in by_patient.items()}


def diagnosis_strata(results: Results) -> dict[str, list[str]]:
    payload = results.payload("S0", "forgetting")
    groups = defaultdict(set)
    for row in (payload or {}).get("records") or []:
        if row["template_id"] == DIAGNOSIS_PROBE:
            groups[row["target"]].add(str(row["record_id"]))
    return {d: sorted(p) for d, p in groups.items()}


def patient_test(per_seed_values, strata, statistic, *, direction: str) -> dict[str, Any]:
    """Pooled estimate, hierarchical bootstrap p and per-seed estimates for a per-patient statistic."""

    seeds = sorted(per_seed_values)
    if len(seeds) < 2:
        return {"estimable": False, "seeds": seeds}
    full = {str(s): [p for group in strata[s].values() for p in group] for s in seeds}
    point = statistic(full)
    per_seed = {s: statistic({str(s): full[str(s)]}) for s in seeds}
    boot = hierarchical_bootstrap({str(s): strata[s] for s in seeds}, statistic)
    return {"estimable": True, "seeds": seeds, "mean": point, "per_seed": per_seed, "_boot": boot,
            "p": bootstrap_p(boot, direction=direction), "p_two_sided": bootstrap_p(boot), "direction": direction}


def finish_patient_family(tests: dict[str, dict[str, Any]]) -> None:
    names = [n for n, t in tests.items() if t["estimable"]]
    if not names:
        return
    for name, p_holm in zip(names, holm([tests[n]["p"] for n in names])):
        test = tests[name]
        boot = test.pop("_boot")
        sign = {"greater": 1, "less": -1}.get(test["direction"], np.sign(test["mean"]))
        agree = all(np.sign(v) == sign for v in test["per_seed"].values())
        test.update({"p_holm": p_holm, "ci95": percentile_ci(boot, 0.95), "per_seed_agree": bool(agree),
                     "supported": bool(p_holm < ALPHA and agree)})


# --- the analysis --------------------------------------------------------------------------


def analyse(runs: list[SeedRun], *, allow_unversioned: bool = False) -> dict[str, Any]:
    results = {run.seed: Results(run, allow_unversioned=allow_unversioned) for run in runs}
    levels = set.intersection(*(set(run.levels) for run in runs))
    out: dict[str, Any] = {"seeds": [run.seed for run in runs], "runs": [run.run_id for run in runs],
                           "analysis_seed": ANALYSIS_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES, "families": {}}

    def crossed_family(high, low, reference, point=POINT):
        family = {}
        if not {high, low} <= levels:
            return {f"{m}/{s}": {"estimable": False, "reason": f"{high} or {low} not in this design"} for m in METHODS for s in SUITES}
        for method in METHODS:
            for suite in SUITES:
                per_seed = {seed: c for seed, r in results.items()
                            if (c := question_contrast(r, suite, method, high=high, low=low, reference=reference, point=point)) is not None}
                family[f"{method}/{suite}"] = crossed_test(per_seed)
        return family

    out["families"]["H1"] = crossed_family("S100", "S0", "S")
    finish_crossed_family(out["families"]["H1"], require_all=len(runs))
    for name, args in {"1a": ("S100", "S0", "C"), "1b": ("S25", "G25", "S"), "1c": ("S25", "G25", "C")}.items():
        out["families"][name] = crossed_family(*args)
        finish_crossed_family(out["families"][name])
    out["families"]["5"] = crossed_family("S100", "S0", "S", point="f_all")
    finish_crossed_family(out["families"]["5"], require_all=len(runs))

    # H2 and family 2 (per-patient type gaps and diagnosis mentions)
    strata = {seed: diagnosis_strata(r) for seed, r in results.items()}
    s_levels = sorted(level for level in levels if level.startswith("S"))
    h2, fam2 = {}, {}
    for method in METHODS:
        tables = {seed: {lvl: t for lvl in levels if (t := per_patient_type_logprobs(r, lvl, method)) is not None}
                  for seed, r in results.items()}
        complete = {seed: t for seed, t in tables.items() if all(lvl in t for lvl in s_levels)}
        h2[f"H2a/{method}"] = patient_test(
            complete, strata,
            lambda drawn, t=complete: float(np.mean([np.mean([gap_difference(t[int(s)][lvl], ps) for lvl in s_levels]) for s, ps in drawn.items()])),
            direction="greater")
        margins = {seed: {lvl: m for lvl in s_levels if (m := diagnosis_margin(r, unlearned(lvl, method))) is not None}
                   for seed, r in results.items()}
        complete_m = {seed: m for seed, m in margins.items() if len(m) == len(s_levels)}
        h2[f"H2b/{method}"] = patient_test(
            complete_m, strata,
            lambda drawn, m=complete_m: float(np.mean([np.mean([np.mean([m[int(s)][lvl][p] for p in ps]) for lvl in s_levels]) for s, ps in drawn.items()])),
            direction="greater")
        for high, low in (("S100", "S0"), ("S25", "G25")):
            have = {seed: t for seed, t in tables.items() if high in t and low in t}
            fam2[f"{method}/{high}-{low}"] = patient_test(
                have, strata,
                lambda drawn, t=have, h=high, l=low: float(np.mean([gap_difference(t[int(s)][h], ps) - gap_difference(t[int(s)][l], ps) for s, ps in drawn.items()])),
                direction="two-sided") if {high, low} <= levels else {"estimable": False}
    finish_patient_family(h2)
    finish_patient_family(fam2)
    out["families"]["H2"], out["families"]["2"] = h2, fam2

    # H3 (descriptive) and family 3: KL(S_i || U) - KL(S_i || C_i), per patient
    def kl_difference(r, clean, method):
        u, c = patient_kl(r, unlearned(clean, method)), patient_kl(r, contaminated(clean))
        return None if u is None or c is None else {p: u[p] - c[p] for p in u if p in c}

    h3, fam3 = {}, {}
    for method in METHODS:
        per = {seed: {lvl: d for lvl in levels if (d := kl_difference(r, lvl, method)) is not None} for seed, r in results.items()}
        pooled = {seed: d for seed, d in per.items() if all(lvl in d for lvl in s_levels)}
        h3[method] = patient_test(pooled, strata, lambda drawn, d=pooled: float(np.mean(
            [np.mean([np.mean([d[int(s)][lvl][p] for p in ps]) for lvl in s_levels]) for s, ps in drawn.items()])), direction="two-sided")
        for high, low in (("S100", "S0"), ("S25", "G25")):
            have = {seed: d for seed, d in per.items() if high in d and low in d}
            fam3[f"{method}/{high}-{low}"] = patient_test(have, strata, lambda drawn, d=have, h=high, l=low: float(np.mean(
                [np.mean([d[int(s)][h][p] - d[int(s)][l][p] for p in ps]) for s, ps in drawn.items()])), direction="two-sided") \
                if {high, low} <= levels else {"estimable": False}
    for test in h3.values():
        if test.get("estimable"):
            boot = test.pop("_boot")
            test["ci95"] = percentile_ci(boot, 0.95)
    finish_patient_family(fam3)
    out["families"]["H3"], out["families"]["3"] = h3, fam3

    # family 4: recovery, (U rise - S_i rise) at the higher level minus at the lower
    def recovery_rise(r, label, treatment):
        payload = r.payload(label, f"recovery_{treatment}")
        if payload is None:
            return None
        return {p: v["after"] - v["before"] for p, v in payload["per_patient"].items()}

    fam4 = {}
    for treatment in ("relearn", "benign"):
        for high, low in (("S100", "S0"), ("S25", "G25")):
            for method in METHODS:
                per = {}
                for seed, r in results.items():
                    parts = [recovery_rise(r, unlearned(lvl, method), treatment) for lvl in (high, low)] + \
                            [recovery_rise(r, lvl, treatment) for lvl in (high, low)]
                    if all(part is not None for part in parts):
                        per[seed] = parts
                fam4[f"{treatment}/{method}/{high}-{low}"] = patient_test(per, strata, lambda drawn, d=per: float(np.mean([
                    np.mean([(d[int(s)][0][p] - d[int(s)][2][p]) - (d[int(s)][1][p] - d[int(s)][3][p]) for p in ps if p in d[int(s)][0]])
                    for s, ps in drawn.items()])), direction="two-sided") if {high, low} <= levels else {"estimable": False}
    finish_patient_family(fam4)
    out["families"]["4"] = fam4

    # Sensitivity analyses and estimation-only tables (8.1, 8.2, 8.5, family 6, section 7).
    from unlearning.analysis import stage2_estimation as extra

    level_list = sorted(levels, key=lambda name: (name[0] != "S", int(name[1:])))
    gains = {"S100-S0": extra.specialization_gain(results, "S100", "S0") if {"S100", "S0"} <= levels else {"estimable": False},
             "S25-G25": extra.specialization_gain(results, "S25", "G25") if {"S25", "G25"} <= levels else {"estimable": False}}
    for family, gain_key in (("H1", "S100-S0"), ("1a", "S100-S0"), ("1b", "S25-G25"), ("1c", "S25-G25")):
        gain = gains[gain_key]
        for test in out["families"][family].values():
            if test.get("estimable"):
                test["fraction_of_specialization_gain"] = (
                    test["mean"] / gain["mean"] if gain.get("estimable") and gain.get("defined") else "undefined"
                )
    out["specialization_gain"] = gains
    out["sensitivity_H1"] = extra.h1_sensitivity(results)
    out["covariate_sensitivity"] = extra.covariate_sensitivity(results, level_list)
    out["regression_on_content_dev_loss"] = extra.regression_on_specialization(results, level_list)
    out["estimation"] = extra.estimation_tables(results, level_list)
    out["manipulation_reading"] = extra.manipulation_reading(results, level_list)
    out["unversioned_results_allowed"] = sorted(path for r in results.values() for path in r.unversioned)
    out["code_version"] = code_version()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-seed Stage-2 analysis (after unblinding).")
    parser.add_argument("--configs", nargs="+", default=[str(REPO_ROOT / "configs" / "stage2" / f"stage2_seed{s}.yaml") for s in (43, 44, 45)])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-unversioned", action="store_true", help="dry runs only (e.g. on the pilot)")
    args = parser.parse_args(argv)
    runs = load_runs([Path(p) for p in args.configs])
    result = analyse(runs, allow_unversioned=args.allow_unversioned)
    output = Path(args.output_dir) if args.output_dir else runs[0].raw.parent.parent / "processed" / "stage2"
    target = output / "analysis.json"
    if target.is_file():  # keep the previous version: processed results are never silently replaced
        target.replace(target.with_name(f"analysis__{target.stat().st_mtime_ns}.json"))
    write_json(target, json.loads(json.dumps(result, default=float)))
    print(output / "analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
