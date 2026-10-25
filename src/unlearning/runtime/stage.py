"""Stage lifecycle shared by specialization, contamination and unlearning.

``prepare_stage`` decides whether a stage is complete (skip), new (start), or
partially done (resume -- only when ``--resume`` is given). ``--overwrite``
never deletes: it renames existing outputs to ``*__invalidated_<UTC time>`` so an
invalidated run stays inspectable until you remove it yourself.

``ExecutionRecord`` keeps one entry per process run of a stage (a "segment"):
when it started and stopped, which steps it covered, on what hardware and
runtime profile. Final manifests embed it, so a resumed run is always
distinguishable from an uninterrupted one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unlearning.runtime.run_log import command_line, utc_now
from unlearning.utils.env import software_versions
from unlearning.utils.hashing import fingerprint_payload
from unlearning.utils.io import read_json, write_json

STAGE_COMPLETE = "complete"
STAGE_FRESH = "fresh"
STAGE_RESUME = "resume"


class StageAlreadyStarted(RuntimeError):
    pass


def _non_empty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def invalidate(paths: list[Path]) -> list[Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved = []
    for path in paths:
        if path.exists():
            target = path.with_name(f"{path.name}__invalidated_{stamp}")
            path.rename(target)
            moved.append(target)
    return moved


def prepare_stage(
    *,
    final_marker: Path,
    paths: list[Path],
    resume: bool,
    overwrite: bool,
    stage: str,
) -> str:
    complete = final_marker.is_file()
    started = complete or any(_non_empty(path) for path in paths)
    if overwrite and started:
        moved = invalidate(paths)
        for target in moved:
            print(f"[{stage}] invalidated previous outputs -> {target}", flush=True)
        return STAGE_FRESH
    if complete:
        return STAGE_COMPLETE
    if started:
        if resume:
            return STAGE_RESUME
        raise StageAlreadyStarted(
            f"{stage} has partial outputs ({', '.join(str(p) for p in paths if _non_empty(p))}). "
            "Rerun with --resume to continue from the last saved step, or --overwrite to "
            "set the old outputs aside and start over."
        )
    return STAGE_FRESH


class ExecutionRecord:
    def __init__(self, path: Path, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.segments: list[dict[str, Any]] = []
        if self.path.is_file():
            self.segments = list(read_json(self.path).get("segments", []))

    def begin(self, *, start_step: int, resumed_from: dict | None, details: dict[str, Any]) -> None:
        for segment in self.segments:
            if segment.get("state") == "running":
                # The previous process never recorded an exit: killed, crashed or power loss.
                segment["state"] = "crashed"
                segment["note"] = "process ended without recording an exit; runtime unknown"
        self.segments.append(
            {
                "index": len(self.segments),
                "started_at": utc_now(),
                "start_step": int(start_step),
                "resumed_from": resumed_from,
                "state": "running",
                **details,
            }
        )
        self._write()

    def end(self, *, end_step: int, state: str, runtime_seconds: float, **fields: Any) -> None:
        if not self.segments:
            return
        self.segments[-1].update(
            {
                "ended_at": utc_now(),
                "end_step": int(end_step),
                "state": state,
                "runtime_seconds": round(float(runtime_seconds), 3),
                **fields,
            }
        )
        self._write()

    def _write(self) -> None:
        if self.enabled:
            write_json(self.path, self.as_dict(), overwrite=True)

    @property
    def total_runtime_seconds(self) -> float:
        return float(sum(segment.get("runtime_seconds") or 0.0 for segment in self.segments))

    @property
    def resumed(self) -> bool:
        return len(self.segments) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage-execution-v1",
            "segment_count": len(self.segments),
            "resumed": self.resumed,
            "total_runtime_seconds": round(self.total_runtime_seconds, 3),
            "segments": self.segments,
        }


def verify_or_write_run_plan(path: Path, plan: dict, *, fresh: bool, is_main: bool) -> str:
    """Store the computation-defining plan on a fresh start; a resume must match it."""

    fingerprint = fingerprint_payload(plan)
    if fresh or not path.is_file():
        if is_main:
            write_json(path, {**plan, "plan_fingerprint": fingerprint}, overwrite=True)
        return fingerprint
    stored = read_json(path)
    if stored.get("plan_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Refusing to resume: the inputs no longer match the run plan in {path}. "
            "The config or data artifacts changed since this run started."
        )
    return fingerprint


def segment_details(hardware: dict, profile, settings, world_size: int) -> dict:
    """What one process run of a stage ran on (recorded per segment)."""

    devices = (hardware.get("torch") or {}).get("devices") or []
    return {
        "command": command_line(),
        "gpus": [
            {"name": device.get("name"), "memory_mb": device.get("total_memory_mb")}
            for device in devices
        ],
        "processes": world_size,
        "runtime_profile": profile.name,
        "runtime_profile_fingerprint": profile.fingerprint,
        "micro_batch": profile.micro_batch.__dict__,
        "machine_settings": settings.as_dict(),
        "software_versions": software_versions(),
    }
