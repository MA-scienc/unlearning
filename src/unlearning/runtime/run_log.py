"""Live status and append-only logs for one stage run.

Files in the stage directory (``runs/<run_id>/<stage>/``):

``status.json``    current state, overwritten atomically; the dashboard reads it
``metrics.jsonl``  one line per optimizer step (append-only; after a resume a step
                   can appear twice -- readers keep the last line for each step)
``events.jsonl``   started / resumed / checkpoint saved / paused / failed / finished
``system.jsonl``   periodic GPU and host samples (written by ``SystemMonitor``)

Only the main process writes; other processes get a disabled logger.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unlearning.utils.io import append_jsonl, ensure_dir, replace_with_retry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(value: Any) -> Any:
    """Make a value JSON-safe: non-finite floats become None, tensors become floats."""

    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except Exception:
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class RunLog:
    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        stage: str,
        total_steps: int,
        enabled: bool = True,
        console_every_steps: int = 10,
    ) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.run_id = run_id
        self.stage = stage
        self.total_steps = int(total_steps)
        self.console_every_steps = max(1, int(console_every_steps))
        self._lock = threading.Lock()
        self._step_seconds: deque[float] = deque(maxlen=50)
        self._loss_ema: float | None = None
        self._status: dict[str, Any] = {
            "run_id": run_id,
            "stage": stage,
            "state": "starting",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "step": 0,
            "total_steps": self.total_steps,
            "percent": 0.0,
        }
        if self.enabled:
            ensure_dir(self.directory)

    # --- paths ----------------------------------------------------------------
    @property
    def status_path(self) -> Path:
        return self.directory / "status.json"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.jsonl"

    @property
    def events_path(self) -> Path:
        return self.directory / "events.jsonl"

    @property
    def system_path(self) -> Path:
        return self.directory / "system.jsonl"

    # --- writing --------------------------------------------------------------
    def _write_status(self) -> None:
        if not self.enabled:
            return
        self._status["updated_at"] = utc_now()
        temporary = self.status_path.with_name(f".status.json.tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps(_clean(self._status), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            replace_with_retry(temporary, self.status_path)
        except OSError:
            # A dashboard holding the file open must never stop training.
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)
            self._write_status()

    def set_state(self, state: str, message: str | None = None, **fields: Any) -> None:
        self.update(state=state, message=message, **fields)

    def event(self, kind: str, message: str = "", **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"time": utc_now(), "kind": kind, "message": message, **fields}
        append_jsonl(self.events_path, _clean(record))
        print(f"[{self.stage}] {kind}: {message}".rstrip(": "), flush=True)

    def system_sample(self, sample: dict[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(self.system_path, _clean(sample))
        self.update(system=sample)

    def step(self, step: int, metrics: dict[str, Any]) -> None:
        """Record one completed optimizer step and refresh status/ETA."""

        if not self.enabled:
            return
        record = {"step": int(step), "time": utc_now(), **metrics}
        append_jsonl(self.metrics_path, _clean(record))
        seconds = metrics.get("step_seconds")
        if isinstance(seconds, (int, float)) and math.isfinite(seconds):
            self._step_seconds.append(float(seconds))
        loss = metrics.get("loss")
        if isinstance(loss, (int, float)) and math.isfinite(loss):
            self._loss_ema = loss if self._loss_ema is None else 0.95 * self._loss_ema + 0.05 * loss
        remaining = max(0, self.total_steps - int(step))
        mean_seconds = (sum(self._step_seconds) / len(self._step_seconds)) if self._step_seconds else None
        eta = remaining * mean_seconds if mean_seconds is not None else None
        self.update(
            state="running",
            step=int(step),
            percent=round(100.0 * int(step) / self.total_steps, 2) if self.total_steps else None,
            eta_seconds=eta,
            mean_step_seconds=mean_seconds,
            last_metrics=record,
            loss_ema=self._loss_ema,
            message=None,
        )
        if int(step) % self.console_every_steps == 0 or int(step) == self.total_steps:
            self._print_step(step, metrics, eta)

    def _print_step(self, step: int, metrics: dict[str, Any], eta: float | None) -> None:
        parts = [f"[{self.stage}] step {step}/{self.total_steps}"]
        for key, fmt in (
            ("loss", "loss {:.4f}"),
            ("lr", "lr {:.2e}"),
            ("grad_norm", "gnorm {:.3f}"),
            ("tokens_per_second", "{:.0f} tok/s"),
            ("peak_memory_gb", "peak {:.2f} GB"),
        ):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                parts.append(fmt.format(value))
        if eta is not None:
            parts.append(f"ETA {format_duration(eta)}")
        print("  ".join(parts), flush=True)

    def finish(self, state: str, message: str | None = None, **fields: Any) -> None:
        self.set_state(state, message=message, finished_at=utc_now(), **fields)
        self.event(state, message or "")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def monotonic() -> float:
    return time.perf_counter()
