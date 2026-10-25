"""Training-curve figures from the append-only stage logs.

Reads ``runs/<run_id>/<stage>/{metrics,dev_monitor,system}.jsonl`` and writes
one PNG per stage to ``results/figures/<run_id>/training/<stage>.png``: small
multiples (one metric per panel, each with its own single y-axis) of loss,
learning rate, gradient norm, throughput, GPU temperature and GPU memory.
These are operational monitoring figures, not study results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.settings import default_config_path
from unlearning.utils.io import ensure_dir

# Reference palette (light mode), see the dataviz skill's palette.md.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line being written while we read
    return rows


def last_per_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After a resume a step can be logged twice; the later line is authoritative."""

    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "step" in row:
            by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def ema(values: list[float], alpha: float = 0.05) -> list[float]:
    smoothed, current = [], None
    for value in values:
        current = value if current is None else (1 - alpha) * current + alpha * value
        smoothed.append(current)
    return smoothed


def _style_axis(axis, title: str, xlabel: str | None = None) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY, pad=6)
    axis.grid(True, color=GRID, linewidth=1.0, linestyle="-")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    if xlabel:
        axis.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=8)


def plot_stage(stage_dir: Path, output: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = last_per_step(read_jsonl(stage_dir / "metrics.jsonl"))
    if not metrics:
        return None
    dev = last_per_step(read_jsonl(stage_dir / "dev_monitor.jsonl"))
    system = read_jsonl(stage_dir / "system.jsonl")
    steps = [row["step"] for row in metrics]

    def series(key: str) -> tuple[list[int], list[float]]:
        pairs = [(row["step"], row[key]) for row in metrics if isinstance(row.get(key), (int, float))]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    figure, axes = plt.subplots(3, 2, figsize=(11, 8.5), facecolor=SURFACE)
    unit = metrics[-1].get("unit", "")
    stage = stage_dir.name

    loss_axis = axes[0][0]
    loss_steps, losses = series("loss")
    loss_axis.plot(loss_steps, losses, color=SERIES_1, linewidth=1.0, alpha=0.25, label="train, per step")
    loss_axis.plot(loss_steps, ema(losses), color=SERIES_1, linewidth=2.0, label="train, smoothed")
    if dev:
        loss_axis.plot(
            [row["step"] for row in dev],
            [row["dev_loss"] for row in dev],
            color=SERIES_2,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label="held-out dev",
        )
    loss_axis.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper right")
    _style_axis(loss_axis, f"Loss (mean per {unit[:-1] if unit.endswith('s') else unit})")

    lr_steps, lrs = series("lr")
    axes[0][1].plot(lr_steps, lrs, color=SERIES_1, linewidth=2.0)
    _style_axis(axes[0][1], "Learning rate")

    grad_steps, grads = series("grad_norm")
    axes[1][0].plot(grad_steps, grads, color=SERIES_1, linewidth=1.0, alpha=0.25)
    axes[1][0].plot(grad_steps, ema(grads), color=SERIES_1, linewidth=2.0)
    _style_axis(axes[1][0], "Gradient norm (before clipping)")

    tps_steps, tps = series("tokens_per_second")
    axes[1][1].plot(tps_steps, tps, color=SERIES_1, linewidth=1.0, alpha=0.25)
    axes[1][1].plot(tps_steps, ema(tps), color=SERIES_1, linewidth=2.0)
    _style_axis(axes[1][1], "Throughput (input tokens / second)")

    sample_index = list(range(len(system)))
    temperatures = [
        max((gpu.get("temperature_c") or 0) for gpu in sample.get("gpus") or [{}]) for sample in system
    ]
    memory = [
        max((gpu.get("memory_used_mb") or 0) for gpu in sample.get("gpus") or [{}]) / 1024 for sample in system
    ]
    axes[2][0].plot(sample_index, temperatures, color=SERIES_1, linewidth=2.0)
    _style_axis(axes[2][0], "GPU temperature (C)", xlabel="system sample")
    axes[2][1].plot(sample_index, memory, color=SERIES_1, linewidth=2.0)
    _style_axis(axes[2][1], "GPU memory used (GB)", xlabel="system sample")
    for axis in (axes[0][0], axes[0][1], axes[1][0], axes[1][1]):
        axis.set_xlabel("optimizer step", color=TEXT_SECONDARY, fontsize=8)

    figure.suptitle(
        f"{stage}: steps {steps[0]}-{steps[-1]} (operational monitoring)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    ensure_dir(output.parent)
    figure.savefig(output, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return output


def plot_all(runs_dir: Path, figures_dir: Path) -> list[Path]:
    written = []
    if not runs_dir.is_dir():
        return written
    for stage_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if stage_dir.name in {"pipeline"} or not (stage_dir / "metrics.jsonl").is_file():
            continue
        path = plot_stage(stage_dir, figures_dir / "training" / f"{stage_dir.name}.png")
        if path is not None:
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot training curves for every stage of a run.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--stage", default=None, help="one stage folder name, e.g. specialize")
    args = parser.parse_args(argv)
    layout = StorageLayout.for_config(load_config(args.config))
    if args.stage:
        written = [plot_stage(layout.stage_dir(args.stage), layout.figures / "training" / f"{args.stage}.png")]
    else:
        written = plot_all(layout.runs, layout.figures)
    for path in written:
        if path:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
