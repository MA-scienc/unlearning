"""Matched-forgetting unlearning (revision 3): stopping signal, references and calibration rule.

docs/MATCHED_FORGETTING_DESIGN.md. Each matched C_i is unlearned until it has
closed fixed fractions of the gap between C_i and its clean S_i in mean per-token
answer log-probability on training-template probes:

    g(U) = (l(C_i) - l(U)) / (l(C_i) - l(S_i))

S_i only defines where a run stops. It never enters the GA/NPO objective.
This module holds the pieces shared by the calibration (unlearning_calibration.py)
and the unlearning run (unlearn.py). It does not import unlearn.py.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from unlearning.config import PilotConfig
from unlearning.config.unlearning_revisions import MatchedForgetting, UnlearningRevision, threshold_name
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.run_log import RunLog, utc_now
from unlearning.training.contaminate import _load_model, load_clean_checkpoint_reference
from unlearning.training.levels import clean_for_contaminated
from unlearning.training.memorization import MemorizationProbe
from unlearning.utils.hashing import fingerprint_payload
from unlearning.utils.io import append_jsonl, read_json, write_json

REFERENCES_FILE = "forgetting_references.json"
# Per-probe l(S_i) and l(C_i) for trajectories whose references predate keeping per_example
# (written by training/probe_references_backfill.py next to REFERENCES_FILE).
PER_EXAMPLE_REFERENCES_FILE = "forgetting_references_per_example.json"
CALIBRATION_SCHEMA = "unlearning-calibration-v1"
FINAL_CALIBRATION_STATES = ("selected", "no_candidate_reached")
# The pilot levels; clean_parent() maps any level by name (training/levels.py).
CLEAN_FOR_CONTAMINATED = {"C0": "S0", "C50": "S50", "C100": "S100"}


def gap_closed(*, contaminated: float, clean: float, current: float) -> float:
    """Fraction of the C_i -> S_i answer log-prob gap closed: 0 at C_i, 1 at S_i, > 1 past it."""

    gap = contaminated - clean
    if not gap > 0:
        raise ValueError(f"C_i must rate the F answers above S_i (l(C_i)={contaminated}, l(S_i)={clean})")
    return (contaminated - current) / gap


def newly_crossed(value: float, thresholds: Sequence[float], reached: Iterable[str]) -> list[str]:
    """Snapshot names of thresholds this check reaches for the first time."""

    done = set(reached)
    return [threshold_name(t) for t in thresholds if value >= t and threshold_name(t) not in done]


def select_learning_rate(trials: Sequence[dict[str, Any]]) -> float | None:
    """The rule: the smallest candidate whose trial reached the last threshold (gap closed 1.0)."""

    reached = [trial["learning_rate"] for trial in trials if trial.get("final_threshold_reached")]
    return min(reached) if reached else None


# --- references: l(S_i) and l(C_i) -----------------------------------------------------


def clean_parent(layout: StorageLayout, config: PilotConfig, source):
    """The clean S_i that ``source`` (a matched C_i) was contaminated from."""

    clean = load_clean_checkpoint_reference(layout.clean_checkpoint(clean_for_contaminated(source.name)), config=config)
    if clean.checkpoint_id != source.parent_clean_checkpoint_id:
        raise ValueError(f"{source.name} was contaminated from {source.parent_clean_checkpoint_id}, not {clean.checkpoint_id}")
    return clean


def _measure(model, probe: MemorizationProbe, device) -> dict[str, Any]:
    # per_example lets reports show per-record gap distributions (references written
    # before this was kept have a backfilled PER_EXAMPLE_REFERENCES_FILE instead).
    result = probe.answer_logprob(model, device)
    return {"mean": result["mean"], "by_template": result["by_template"], "per_example": result["per_example"]}


def load_or_measure_references(
    path: Path,
    *,
    probe: MemorizationProbe,
    clean,
    source,
    config: PilotConfig,
    auto_model,
    torch,
    device,
    is_main: bool = True,
) -> dict[str, Any]:
    """l(S_i) and l(C_i) on the probe set, measured once and cached next to the run."""

    identity = {
        "probe_fingerprint": probe.fingerprint,
        "probe_templates": list(probe.templates),
        "clean_checkpoint_id": clean.checkpoint_id,
        "contaminated_checkpoint_id": source.checkpoint_id,
        "contaminated_manifest_sha256": source.manifest_sha256,
    }
    if path.is_file():
        stored = read_json(path)
        if {key: stored.get(key) for key in identity} != identity:
            raise RuntimeError(f"forgetting references at {path} belong to other probes or checkpoints")
        return stored
    measured = {}
    for role, loader in (
        ("clean", lambda: _load_model(clean, config, auto_model, torch)),
        ("contaminated", lambda: auto_model.from_pretrained(source.checkpoint_dir / "model", torch_dtype=torch.bfloat16)),
    ):
        model = loader().to(device)
        try:
            measured[role] = _measure(model, probe, device)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    payload = {
        **identity,
        "clean": measured["clean"],
        "contaminated": measured["contaminated"],
        "gap": measured["contaminated"]["mean"] - measured["clean"]["mean"],
        "measured_at": utc_now(),
    }
    payload["fingerprint"] = fingerprint_payload(payload)
    if is_main:
        write_json(path, payload, overwrite=False)
    return payload


def per_example_references(trajectory_dir: Path) -> dict[str, dict[str, float]] | None:
    """``{"clean": {example_id: lp}, "contaminated": {...}}`` for a trajectory, or ``None``.

    Read from the references file when it kept per_example, else from the
    backfilled sidecar, which must name the same probes and checkpoints.
    """

    references_path = Path(trajectory_dir) / REFERENCES_FILE
    if not references_path.is_file():
        return None
    references = read_json(references_path)
    if all("per_example" in references.get(role, {}) for role in ("clean", "contaminated")):
        return {role: references[role]["per_example"] for role in ("clean", "contaminated")}
    sidecar_path = Path(trajectory_dir) / PER_EXAMPLE_REFERENCES_FILE
    if not sidecar_path.is_file():
        return None
    sidecar = read_json(sidecar_path)
    if sidecar.get("references_fingerprint") != references.get("fingerprint"):
        raise ValueError(f"{sidecar_path} was measured for different references")
    return {role: sidecar[role]["per_example"] for role in ("clean", "contaminated")}


# --- the per-check monitor used as TrainingLoop.on_step_end ------------------------------


class ForgettingMonitor:
    """Every ``check_every_steps`` steps: measure l(U), compute g, act on newly crossed thresholds.

    ``on_threshold(names, entry)`` saves snapshots (main run) and returns True when it saved full
    training state. The loop is told to finish once the last threshold is reached. Every process
    computes g identically from identical weights, as the loop requires for ``finish_now``.
    """

    def __init__(
        self,
        *,
        probe: MemorizationProbe,
        references: dict[str, Any],
        spec: MatchedForgetting,
        model_getter: Callable[[], Any],
        device,
        run_log: RunLog,
        history_path: Path | None,
        reached: dict[str, dict[str, Any]] | None = None,
        on_threshold: Callable[[list[str], dict[str, Any]], bool] | None = None,
        is_main: bool = True,
    ):
        self.probe = probe
        self.references = references
        self.spec = spec
        self.model_getter = model_getter
        self.device = device
        self.run_log = run_log
        self.history_path = history_path
        self.reached: dict[str, dict[str, Any]] = dict(reached or {})
        self.on_threshold = on_threshold
        self.is_main = is_main
        self.loop = None
        self.last_entry: dict[str, Any] | None = None

    @property
    def final_reached(self) -> bool:
        return threshold_name(self.spec.thresholds[-1]) in self.reached

    def __call__(self, step: int, metrics: dict[str, Any]) -> bool:
        if step % self.spec.check_every_steps:
            return False
        measured = self.probe.answer_logprob(self.model_getter(), self.device)
        value = gap_closed(
            contaminated=self.references["contaminated"]["mean"],
            clean=self.references["clean"]["mean"],
            current=measured["mean"],
        )
        entry = {
            "step": step,
            "gap_closed": value,
            "answer_logprob": measured["mean"],
            "by_template": measured["by_template"],
            "train_loss": metrics.get("loss"),
            "time": utc_now(),
        }
        self.last_entry = entry
        if self.history_path is not None and self.is_main:
            append_jsonl(self.history_path, entry)
        self.run_log.event("forgetting", f"step {step}: gap closed {value:.3f}", step=step, gap_closed=value)
        self.run_log.update(forgetting=entry)
        names = newly_crossed(value, self.spec.thresholds, self.reached)
        saved = False
        if names:
            for name in names:
                self.reached[name] = {"step": step, "gap_closed": value}
            if self.on_threshold is not None:
                saved = bool(self.on_threshold(names, {**entry, "per_example": measured["per_example"]}))
        if self.final_reached and self.loop is not None:
            self.loop.finish_now = True
        return saved


# --- calibration report ------------------------------------------------------------------


def calibration_dir(layout: StorageLayout, revision: UnlearningRevision) -> Path:
    return layout.stage_dir(f"calibrate_unlearning_{revision.tag}")


def calibration_report_path(layout: StorageLayout, revision: UnlearningRevision) -> Path:
    return calibration_dir(layout, revision) / "calibration_report.json"


def calibration_complete(layout: StorageLayout, revision: UnlearningRevision) -> bool:
    path = calibration_report_path(layout, revision)
    if not path.is_file():
        return False
    report = read_json(path)
    return report.get("status") in FINAL_CALIBRATION_STATES and report.get("revision_fingerprint") == revision.fingerprint


def selected_learning_rate(layout: StorageLayout, revision: UnlearningRevision, method: str) -> tuple[float, str]:
    """(rate, report sha256) chosen for ``method`` by the calibration rule; refuses anything else."""

    from unlearning.utils.hashing import hash_file

    path = calibration_report_path(layout, revision)
    if not path.is_file():
        raise FileNotFoundError(f"run the unlearning calibration first: {path}")
    report = read_json(path)
    if report.get("revision_fingerprint") != revision.fingerprint:
        raise RuntimeError(f"calibration report {path} was made for a different revision file")
    selected = (report.get("methods") or {}).get(method, {}).get("selected_learning_rate")
    if selected is None:
        raise RuntimeError(f"calibration selected no learning rate for {method}: {report.get('status')}")
    return float(selected), hash_file(path)
