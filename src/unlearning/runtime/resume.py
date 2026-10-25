"""Crash-safe resume points.

A resume point is a directory with the same layout as a scientific checkpoint::

    model/               save_pretrained weights (bf16 safetensors)
    training_state.pt    global_step, optimizer, scheduler, RNG states
    resume_meta.json     written last; its presence marks the point complete

Points are written to ``<name>.tmp`` and renamed only after every file is on
disk, so a crash during a save never damages the previous good point. Data
order in every stage is a pure function of the optimizer step, so resuming needs
no data-loader state: restoring weights, optimizer, scheduler and RNG at step k
and continuing with step k + 1 reproduces the uninterrupted computation (up to
GPU kernel non-determinism, which is why resumes are recorded in manifests).
"""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unlearning.runtime.run_log import utc_now
from unlearning.utils.io import ensure_dir, read_json, write_json

TRAINING_STATE_FILE = "training_state.pt"
RESUME_META_FILE = "resume_meta.json"


@dataclass(frozen=True)
class ResumePoint:
    step: int
    directory: Path
    kind: str  # "resume" (operational) or "scientific" (an S/C/U checkpoint with full state)

    @property
    def model_dir(self) -> Path:
        return self.directory / "model"

    @property
    def state_path(self) -> Path:
        return self.directory / TRAINING_STATE_FILE


def capture_rng_state(torch) -> dict[str, Any]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_random_state": random.getstate(),
    }


def restore_rng_state(torch, state: dict[str, Any]) -> None:
    if state.get("python_random_state") is not None:
        random.setstate(_as_python_random_state(state["python_random_state"]))
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    cuda_states = state.get("cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_states)


def _as_python_random_state(raw):
    version, internal, gauss = raw
    return (version, tuple(internal), gauss)


def build_training_state(torch, *, step: int, optimizer, scheduler=None, extra: dict | None = None) -> dict:
    """The payload written as ``training_state.pt`` by every stage."""

    state = {"global_step": int(step), "optimizer": optimizer.state_dict(), **capture_rng_state(torch)}
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    return state


def load_training_state(torch, path: Path) -> dict:
    # These files are produced by this pipeline (optimizer states and Python RNG
    # tuples), so full unpickling is required; never point this at untrusted files.
    return torch.load(path, map_location="cpu", weights_only=False)


def temporary_dir_for(final_dir: Path) -> Path:
    return final_dir.with_name(final_dir.name + ".tmp")


def begin_atomic_dir(final_dir: Path) -> Path:
    """Create a fresh ``<final>.tmp`` directory to write a checkpoint into."""

    temporary = temporary_dir_for(final_dir)
    if temporary.exists():
        shutil.rmtree(temporary)
    ensure_dir(temporary)
    return temporary


def commit_atomic_dir(temporary: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint directory: {final_dir}")
    for attempt in range(20):
        try:
            temporary.rename(final_dir)
            return
        except PermissionError:
            # Windows antivirus/indexers can briefly hold freshly written files.
            if attempt == 19:
                raise
            time.sleep(0.25)


def remove_stale_temporaries(directory: Path) -> list[Path]:
    """Delete ``*.tmp`` checkpoint directories left behind by a crash mid-save."""

    removed = []
    if directory.is_dir():
        for candidate in directory.glob("*.tmp"):
            if candidate.is_dir():
                shutil.rmtree(candidate)
                removed.append(candidate)
    return removed


class ResumeStore:
    """Rolling operational resume points in ``runs/<run_id>/<stage>/resume``."""

    def __init__(self, directory: Path, *, keep_last: int, every_minutes: float, enabled: bool = True):
        self.directory = Path(directory)
        self.keep_last = int(keep_last)
        self.every_seconds = float(every_minutes) * 60.0
        self.enabled = enabled
        self._last_save = time.monotonic()

    def points(self) -> list[ResumePoint]:
        if not self.directory.is_dir():
            return []
        found = []
        for candidate in sorted(self.directory.glob("step*")):
            if candidate.suffix == ".tmp" or not (candidate / RESUME_META_FILE).is_file():
                continue
            meta = read_json(candidate / RESUME_META_FILE)
            found.append(ResumePoint(step=int(meta["step"]), directory=candidate, kind="resume"))
        return sorted(found, key=lambda point: point.step)

    def due(self) -> bool:
        return time.monotonic() - self._last_save >= self.every_seconds

    def mark_saved(self) -> None:
        self._last_save = time.monotonic()

    def save(
        self,
        *,
        step: int,
        accelerator,
        model,
        optimizer,
        scheduler,
        torch,
        meta: dict[str, Any],
    ) -> ResumePoint | None:
        final_dir = self.directory / f"step{step:06d}"
        if accelerator.is_main_process and not final_dir.exists():
            temporary = begin_atomic_dir(final_dir)
            accelerator.unwrap_model(model).save_pretrained(temporary / "model", safe_serialization=True)
            torch.save(
                build_training_state(torch, step=step, optimizer=optimizer, scheduler=scheduler),
                temporary / TRAINING_STATE_FILE,
            )
            write_json(
                temporary / RESUME_META_FILE,
                {"step": int(step), "saved_at": utc_now(), **meta},
            )
            commit_atomic_dir(temporary, final_dir)
            self._prune()
        accelerator.wait_for_everyone()
        self.mark_saved()
        return ResumePoint(step=step, directory=final_dir, kind="resume")

    def _prune(self) -> None:
        points = self.points()
        for stale in points[: max(0, len(points) - self.keep_last)]:
            shutil.rmtree(stale.directory, ignore_errors=True)

    def clear(self) -> None:
        """Remove all operational resume points once the stage has completed."""

        if self.directory.is_dir():
            shutil.rmtree(self.directory, ignore_errors=True)


def latest_point(*candidates: list[ResumePoint]) -> ResumePoint | None:
    merged = [point for group in candidates for point in group]
    if not merged:
        return None
    # Prefer a scientific checkpoint over an operational point at the same step.
    return max(merged, key=lambda point: (point.step, point.kind == "scientific"))
