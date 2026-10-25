"""Graceful stop on Ctrl+C, Ctrl+Break, or SIGTERM (cloud preemption).

The first signal only sets a flag; the training loop finishes the current
optimizer step, saves a resume point, and exits with code 130. A second signal
restores the default handler and interrupts immediately.
"""

from __future__ import annotations

import signal
from typing import Any

INTERRUPTED_EXIT_CODE = 130


class GracefulStop:
    def __init__(self) -> None:
        self.requested = False
        self.reason: str | None = None
        self._previous: dict[int, Any] = {}

    def _signals(self) -> list[int]:
        names = ("SIGINT", "SIGTERM", "SIGBREAK")
        return [getattr(signal, name) for name in names if hasattr(signal, name)]

    def _handle(self, signum, _frame) -> None:
        if self.requested:
            self.uninstall()
            raise KeyboardInterrupt
        self.requested = True
        self.reason = signal.Signals(signum).name
        print(
            f"\n[stop] {self.reason} received: finishing the current step, saving a resume "
            "point, then exiting. Press Ctrl+C again to abort immediately.",
            flush=True,
        )

    def install(self) -> "GracefulStop":
        for signum in self._signals():
            try:
                self._previous[signum] = signal.signal(signum, self._handle)
            except (ValueError, OSError):  # not in the main thread
                pass
        return self

    def uninstall(self) -> None:
        for signum, previous in self._previous.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):
                pass
        self._previous.clear()

    def __enter__(self) -> "GracefulStop":
        return self.install()

    def __exit__(self, *_exc) -> None:
        self.uninstall()


class StageInterrupted(Exception):
    """Raised by a stage after it saved a resume point in response to a stop request."""

    def __init__(self, step: int, reason: str | None):
        super().__init__(f"stopped at step {step} ({reason or 'stop requested'})")
        self.step = step
        self.reason = reason
