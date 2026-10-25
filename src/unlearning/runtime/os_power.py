"""Windows power policies for long unattended jobs on laptops.

Observed on the pilot laptop (2026-10-25): with the machine unattended,
clean specialization ran ~2.8x slower (18 s instead of 6.6 s per step) with the
GPU idle 70% of the time, while running at full speed whenever someone was at
the machine. The training loop is limited by how fast one Python thread can
launch GPU work, so Windows moving a background process onto efficiency cores
or lower CPU states starves the GPU.

Each job process therefore, on Windows:

* opts out of execution-speed power throttling (EcoQoS), which keeps its
  threads eligible for performance cores when its window is in the background;
* asks the system (and, by default, the display) to stay awake while it runs.

Nothing here changes what is computed. Both are per-process requests that end
when the process exits; no system power setting is modified. Configure with
``UNLEARNING_DISABLE_POWER_THROTTLING`` and ``UNLEARNING_KEEP_DISPLAY_ON`` (both
default to 1; set 0 to turn off).
"""

from __future__ import annotations

import os

_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_THROTTLE_EXECUTION_SPEED = 0x1
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def _enabled(name: str) -> bool:
    return (os.environ.get(name) or "1").strip() not in {"0", "false", "False", "no"}


def opt_out_of_power_throttling() -> bool:
    """Ask Windows not to throttle this process's execution speed (no-op elsewhere)."""

    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        state = PowerThrottlingState(1, _THROTTLE_EXECUTION_SPEED, 0)
        return bool(
            kernel32.SetProcessInformation(
                kernel32.GetCurrentProcess(),
                _PROCESS_POWER_THROTTLING,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
    except Exception:
        return False


def keep_awake(*, display: bool) -> bool:
    """Keep the system (and optionally the display) awake while this process runs."""

    if os.name != "nt":
        return False
    try:
        import ctypes

        flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | (_ES_DISPLAY_REQUIRED if display else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


def apply_process_power_policies() -> dict[str, bool]:
    """Apply the configured policies to the current process; returns what was applied."""

    applied = {"power_throttling_disabled": False, "keep_awake": False}
    if os.name != "nt":
        return applied
    if _enabled("UNLEARNING_DISABLE_POWER_THROTTLING"):
        applied["power_throttling_disabled"] = opt_out_of_power_throttling()
    applied["keep_awake"] = keep_awake(display=_enabled("UNLEARNING_KEEP_DISPLAY_ON"))
    return applied
