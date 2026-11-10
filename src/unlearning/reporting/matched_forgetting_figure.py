"""Forgetting-retention curves for matched forgetting (docs/MATCHED_FORGETTING_DESIGN.md, 4).

One panel per method (rows) and retention suite (columns). x = gap closed on the
training probes as recorded in each snapshot manifest (0 = C_i); y = letter-bias-free
accuracy minus S_i's, with paired bootstrap 95% CIs over questions. One line per
specialization level. Points of different levels sit at almost the same x, so they
are dodged by ±0.012 on x to keep the error bars apart (stated in the caption).

Colours: the first three slots of the dataviz reference categorical palette,
validated all-pairs (CVD ΔE ≥ 9.2, normal-vision ΔE ≥ 24). Slot 3 is below 3:1
contrast on white, so identity is also carried by marker shape and the legend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from unlearning.reporting.matched_forgetting import LEVELS, METHODS, RETENTION, _correctness, _metrics, _value, paired_bootstrap

COLOURS = {"0": "#2a78d6", "50": "#eb6834", "100": "#1baf7a"}
MARKERS = {"0": "o", "50": "s", "100": "^"}
DODGE = {"0": -0.012, "50": 0.0, "100": 0.012}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SUITE_TITLES = {"near": "Near domain (medicine, pharmacology)", "far": "Other medical subjects", "general": "General (MMLU-CF)"}


def curve_points(raw: Path, processed: Path, *, tag: str, names: tuple[str, ...], seed: int):
    """``{(method, level, suite): [(x, mean, low, high), ...]}`` from C_i through every snapshot."""

    metrics = _metrics(processed)
    points = {}
    for method in METHODS:
        for level in LEVELS:
            models = [(f"MR_C{level}", 0.0)]
            for name in names:
                model = f"MR_{tag}_C{level}_{method}_{name}"
                gap = _value(metrics, model, "forgetting", "training_probe_gap_closed")
                if gap is not None:
                    models.append((model, gap))
            for suite in RETENTION:
                clean = _correctness(raw, f"S{level}", suite)
                series = []
                for model, x in models:
                    values = _correctness(raw, model, suite)
                    if values is None or clean is None:
                        continue
                    mean, low, high, _ = paired_bootstrap([float(a) - float(b) for a, b in zip(values, clean)], seed=seed)
                    series.append((x, mean, low, high))
                points[(method, level, suite)] = series
    return points


def plot_forgetting_retention(raw: Path, processed: Path, out_dir: Path, *, tag: str, names: tuple[str, ...], seed: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = curve_points(raw, processed, tag=tag, names=names, seed=seed)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    figure, axes = plt.subplots(len(METHODS), len(RETENTION), figsize=(10.5, 6.2), sharex=True, sharey=True)
    for row, method in enumerate(METHODS):
        for column, suite in enumerate(RETENTION):
            ax = axes[row][column]
            ax.axhline(0.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for level in LEVELS:
                series = points.get((method, level, suite)) or []
                if not series:
                    continue
                x = np.array([p[0] for p in series]) + DODGE[level]
                mean = np.array([p[1] for p in series])
                error = np.array([[p[1] - p[2] for p in series], [p[3] - p[1] for p in series]])
                ax.errorbar(
                    x, mean, yerr=error, color=COLOURS[level], marker=MARKERS[level], markersize=6,
                    markeredgecolor="white", markeredgewidth=1.0, linewidth=2.0, elinewidth=1.0, capsize=0,
                    label=f"C{level} (from S{level})", zorder=3,
                )
            if row == 0:
                ax.set_title(SUITE_TITLES[suite], color=INK, fontsize=10)
            if column == 0:
                ax.set_ylabel(f"{method.upper()}\naccuracy change vs S_i", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(LEVELS), frameon=False, bbox_to_anchor=(0.5, 1.0))
    figure.supxlabel("gap closed on training probes (0 = contaminated C_i before unlearning, 1 = clean S_i level)", color=INK, fontsize=9, y=0.045)
    figure.text(
        0.01, 0.008,
        "Letter-bias-free accuracy minus the level's clean model; bars: paired bootstrap 95% CI over questions. "
        "Points dodged ±0.012 on x. One seed, Qwen2.5-0.5B.",
        color=MUTED, fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "forgetting_retention_curves.png"
    for path in (target, target.with_suffix(".pdf")):
        if path.is_file():
            path.replace(path.with_name(f"{path.stem}__{path.stat().st_mtime_ns}{path.suffix}"))
    figure.savefig(target, dpi=200, facecolor="white")
    figure.savefig(target.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return target
