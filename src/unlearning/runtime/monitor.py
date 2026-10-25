"""GPU/host sampling, thermal pausing, and the background system monitor.

Sampling uses NVIDIA's NVML bindings (``nvidia-ml-py``) when available. Pausing
for heat only happens between optimizer steps, so it changes wall-clock time but
never the training computation.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from unlearning.runtime.run_log import RunLog, utc_now

# NVML "clocks event reasons" bits worth surfacing on a dashboard.
_THROTTLE_BITS = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}


class GpuSampler:
    def __init__(self) -> None:
        self._nvml = None
        self.error: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception as exc:  # NVML missing or no NVIDIA driver
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def sample(self) -> list[dict[str, Any]]:
        if self._nvml is None:
            return []
        nvml = self._nvml
        devices = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            entry: dict[str, Any] = {"index": index}
            readers: dict[str, Callable[[], Any]] = {
                "name": lambda: nvml.nvmlDeviceGetName(handle),
                "temperature_c": lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                "power_w": lambda: round(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1),
                "power_limit_w": lambda: round(nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0, 1),
                "utilization_pct": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "memory_used_mb": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used // (1024**2),
                "memory_total_mb": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).total // (1024**2),
                "sm_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
                "throttle_reasons": lambda: _decode_throttle(_clocks_event_reasons(nvml, handle)),
            }
            for key, reader in readers.items():
                try:
                    value = reader()
                    entry[key] = value.decode() if isinstance(value, bytes) else value
                except Exception:
                    entry[key] = None
            devices.append(entry)
        return devices

    def max_temperature(self) -> float | None:
        temperatures = [gpu.get("temperature_c") for gpu in self.sample()]
        temperatures = [float(value) for value in temperatures if value is not None]
        return max(temperatures) if temperatures else None


def _clocks_event_reasons(nvml, handle) -> int:
    reader = getattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
        nvml, "nvmlDeviceGetCurrentClocksThrottleReasons"
    )
    return int(reader(handle))


def _decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in _THROTTLE_BITS.items() if mask & bit]


def host_sample(storage_root: Path) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    try:
        import psutil

        memory = psutil.virtual_memory()
        sample["cpu_pct"] = psutil.cpu_percent(interval=None)
        sample["ram_used_gb"] = round(memory.used / 1024**3, 2)
        sample["ram_total_gb"] = round(memory.total / 1024**3, 2)
    except Exception:
        pass
    try:
        candidate = Path(storage_root)
        while not candidate.exists() and candidate.parent != candidate:
            candidate = candidate.parent
        sample["disk_free_gb"] = round(shutil.disk_usage(candidate).free / 1024**3, 1)
    except OSError:
        pass
    return sample


class SystemMonitor:
    """Background thread writing ``system.jsonl`` every ``interval`` seconds."""

    def __init__(self, run_log: RunLog, *, interval: float, storage_root: Path, sampler: GpuSampler | None = None):
        self.run_log = run_log
        self.interval = float(interval)
        self.storage_root = Path(storage_root)
        self.sampler = sampler or GpuSampler()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        return {"time": utc_now(), "gpus": self.sampler.sample(), **host_sample(self.storage_root)}

    def start(self) -> "SystemMonitor":
        if not self.run_log.enabled:
            return self
        self._thread = threading.Thread(target=self._loop, name="system-monitor", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_log.system_sample(self.snapshot())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


class ThermalGuard:
    """Hold the main process between steps while the GPU is too hot."""

    def __init__(
        self,
        run_log: RunLog,
        sampler: GpuSampler,
        *,
        pause_at_c: float | None,
        resume_at_c: float,
        poll_seconds: float = 15.0,
        max_pause_seconds: float = 30 * 60,
    ) -> None:
        self.run_log = run_log
        self.sampler = sampler
        self.pause_at_c = pause_at_c
        self.resume_at_c = resume_at_c
        self.poll_seconds = poll_seconds
        self.max_pause_seconds = max_pause_seconds
        self.total_paused_seconds = 0.0

    def wait_if_hot(self) -> float:
        if self.pause_at_c is None or not self.sampler.available or not self.run_log.enabled:
            return 0.0
        temperature = self.sampler.max_temperature()
        if temperature is None or temperature < self.pause_at_c:
            return 0.0
        started = time.perf_counter()
        self.run_log.event(
            "thermal_pause",
            f"GPU at {temperature:.0f} C >= {self.pause_at_c:.0f} C; pausing until {self.resume_at_c:.0f} C",
        )
        self.run_log.set_state("paused_thermal", message=f"GPU {temperature:.0f} C")
        while True:
            time.sleep(self.poll_seconds)
            temperature = self.sampler.max_temperature()
            waited = time.perf_counter() - started
            if temperature is None or temperature <= self.resume_at_c:
                break
            if waited >= self.max_pause_seconds:
                self.run_log.event(
                    "thermal_pause_timeout",
                    f"still {temperature:.0f} C after {waited / 60:.0f} min; continuing (check cooling)",
                )
                break
        waited = time.perf_counter() - started
        self.total_paused_seconds += waited
        self.run_log.event("thermal_resume", f"resumed after {waited:.0f} s")
        self.run_log.set_state("running")
        return waited
