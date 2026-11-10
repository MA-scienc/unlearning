"""Registered statistical procedures for Stage 2 (docs/PREREGISTRATION_STAGE2.md, section 8).

Pure functions on arrays, unit-tested on constructed data. Nothing here reads results.

* ``crossed_anova``: the question x seed crossed design of the primary analysis (8.1). The
  variance of the grand mean is (MS_Q + MS_S - MS_E) / (Q x S); a negative variance
  component is set to 0 and the variance recomputed from the remaining terms; Satterthwaite
  degrees of freedom follow the combination used; t-based p and CIs.
* ``crossed_bootstrap``: resamples questions and seeds jointly (8.1).
* ``holm`` and ``holm_levels``: step-down adjusted p and the confidence level of each
  test's Holm step (the bootstrap CI that must also exclude 0).
* ``stratified_patient_resample``: seeds as fixed strata, diagnosis groups resampled within
  a seed, then patients within a group (8.3).
* ``symmetric_bound``: the smallest b with the CI inside [-b, b] (8.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats as _stats

ALPHA = 0.05
BOOTSTRAP_SAMPLES = 10_000
ANALYSIS_SEED = 20261109  # fixed analysis seed (8.7), independent of any config seed


@dataclass(frozen=True)
class CrossedEstimate:
    mean: float
    variance: float
    df: float
    t: float
    p: float
    questions: int
    seeds: int
    components: dict
    truncated: tuple[str, ...]

    def ci(self, level: float) -> tuple[float, float]:
        if self.variance == 0:
            return self.mean, self.mean
        half = _stats.t.ppf(0.5 + level / 2, self.df) * np.sqrt(self.variance)
        return self.mean - half, self.mean + half


def satterthwaite(terms: list[tuple[float, float]]) -> float:
    """df for a sum of scaled mean squares: terms are (coefficient x MS, df of that MS)."""

    numerator = sum(value for value, _ in terms) ** 2
    denominator = sum(value**2 / df for value, df in terms if df > 0)
    return numerator / denominator if denominator > 0 else float("inf")


def crossed_anova(values: np.ndarray) -> CrossedEstimate:
    """Grand mean and its variance for a balanced Q (questions) x S (seeds) array, one value per cell."""

    data = np.asarray(values, dtype=float)
    q, s = data.shape
    if q < 2 or s < 2:
        raise ValueError("the crossed analysis needs at least 2 questions and 2 seeds")
    grand = data.mean()
    by_question, by_seed = data.mean(axis=1), data.mean(axis=0)
    ms_q = s * np.sum((by_question - grand) ** 2) / (q - 1)
    ms_s = q * np.sum((by_seed - grand) ** 2) / (s - 1)
    residual = data - by_question[:, None] - by_seed[None, :] + grand
    ms_e = np.sum(residual**2) / ((q - 1) * (s - 1))
    df_q, df_s, df_e = q - 1, s - 1, (q - 1) * (s - 1)
    sigma_q, sigma_s = (ms_q - ms_e) / s, (ms_s - ms_e) / q
    truncated = tuple(name for name, value in (("question", sigma_q), ("seed", sigma_s)) if value < 0)
    n = q * s
    if not truncated:
        variance = (ms_q + ms_s - ms_e) / n
        df = satterthwaite([(ms_q / n, df_q), (ms_s / n, df_s), (-ms_e / n, df_e)])
    elif truncated == ("question",):
        variance, df = ms_s / n, df_s
    elif truncated == ("seed",):
        variance, df = ms_q / n, df_q
    else:
        variance, df = ms_e / n, df_e
    variance = max(float(variance), 0.0)
    t = grand / np.sqrt(variance) if variance > 0 else (0.0 if grand == 0 else np.inf * np.sign(grand))
    p = float(2 * _stats.t.sf(abs(t), df)) if np.isfinite(t) else 0.0
    return CrossedEstimate(
        mean=float(grand), variance=variance, df=float(df), t=float(t), p=p, questions=q, seeds=s,
        components={"MS_Q": float(ms_q), "MS_S": float(ms_s), "MS_E": float(ms_e),
                    "sigma2_question": float(sigma_q), "sigma2_seed": float(sigma_s), "sigma2_error": float(ms_e)},
        truncated=truncated,
    )


def crossed_bootstrap(values: np.ndarray, *, samples: int = BOOTSTRAP_SAMPLES, seed: int = ANALYSIS_SEED) -> np.ndarray:
    """Bootstrap means resampling questions and seeds jointly (rows and columns with replacement)."""

    data = np.asarray(values, dtype=float)
    q, s = data.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        rows = rng.integers(0, q, size=q)
        columns = rng.integers(0, s, size=s)
        estimates[index] = data[np.ix_(rows, columns)].mean()
    return estimates


def percentile_ci(estimates: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))


def bootstrap_p(estimates: np.ndarray, *, direction: str = "two-sided") -> float:
    """Share of resamples at or beyond 0 against the direction; two-sided doubles the smaller share."""

    below, above = float(np.mean(estimates <= 0)), float(np.mean(estimates >= 0))
    if direction == "greater":
        return below
    if direction == "less":
        return above
    return min(1.0, 2 * min(below, above))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def holm_levels(pvalues: list[float], alpha: float = ALPHA) -> list[float]:
    """Confidence level of each test's Holm step: 1 - alpha / (m - rank)."""

    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    levels = [0.0] * len(pvalues)
    for rank, index in enumerate(order):
        levels[index] = 1 - alpha / (len(pvalues) - rank)
    return levels


def symmetric_bound(interval: tuple[float, float]) -> float:
    return float(max(abs(interval[0]), abs(interval[1])))


def stratified_patient_resample(
    strata: dict[str, dict[str, list[str]]], rng: np.random.Generator
) -> dict[str, list[str]]:
    """Per seed (fixed stratum): resample diagnosis groups, then patients within each drawn group."""

    drawn: dict[str, list[str]] = {}
    for seed, groups in strata.items():
        names = list(groups)
        patients: list[str] = []
        for group in rng.choice(names, size=len(names), replace=True):
            members = groups[group]
            patients += list(rng.choice(members, size=len(members), replace=True))
        drawn[seed] = patients
    return drawn


def hierarchical_bootstrap(
    strata: dict[str, dict[str, list[str]]],
    statistic: Callable[[dict[str, list[str]]], float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = ANALYSIS_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([statistic(stratified_patient_resample(strata, rng)) for _ in range(samples)])
