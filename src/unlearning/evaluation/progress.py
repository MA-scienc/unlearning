"""Live progress of one evaluation job for the dashboard.

Writes ``runs/<run_id>/evaluate/status.json`` (overwritten by each job) with the
model, suite, items done, rate and ETA. Display only: it never affects scoring,
and any failure to write progress is ignored.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from unlearning.config import PilotConfig
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.run_log import RunLog, utc_now


class EvaluationProgress:
    def __init__(self, config: PilotConfig, output_path: str | Path, total: int, description: str):
        output = Path(output_path)
        self.total = max(1, int(total))
        self.started = perf_counter()
        self.last_write = 0.0
        self.log: RunLog | None = None
        try:
            layout = StorageLayout.for_config(config)
            self.log = RunLog(
                layout.stage_dir("evaluate"),
                run_id=layout.run_id,
                stage="evaluate",
                total_steps=self.total,
                console_every_steps=10**9,
            )
            self.log.update(
                state="running",
                model=output.parent.name,
                suite=output.stem,
                message=f"model {output.parent.name} | suite {output.stem} | {description}",
                started_at=utc_now(),
                step=0,
                percent=0.0,
                eta_seconds=None,
                mean_step_seconds=None,
                last_metrics=None,
            )
        except Exception:
            self.log = None

    def __call__(self, done: int) -> None:
        if self.log is None:
            return
        now = perf_counter()
        if done < self.total and now - self.last_write < 2.0:
            return
        self.last_write = now
        try:
            seconds_per_item = (now - self.started) / max(1, done)
            self.log.update(
                state="running",
                step=int(done),
                percent=round(100.0 * done / self.total, 2),
                eta_seconds=(self.total - done) * seconds_per_item,
                mean_step_seconds=seconds_per_item,
            )
        except Exception:
            pass

    def finish(self) -> None:
        if self.log is None:
            return
        try:
            self.log.update(state="completed", step=self.total, percent=100.0, eta_seconds=0.0, finished_at=utc_now())
        except Exception:
            pass
