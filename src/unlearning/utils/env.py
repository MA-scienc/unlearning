from __future__ import annotations

import platform
import shutil
import subprocess


def _bytes_to_gib(value: int) -> float:
    return round(value / (1024**3), 2)


def _total_ram_gib() -> float | None:
    try:
        import psutil

        return _bytes_to_gib(int(psutil.virtual_memory().total))
    except Exception:
        pass
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        return _bytes_to_gib(int(memory_status.ullTotalPhys))
    except Exception:
        return None


def _nvidia_smi() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    query = (
        "nvidia-smi --query-gpu=name,memory.total,driver_version "
        "--format=csv,noheader,nounits"
    )
    try:
        result = subprocess.run(
            query,
            check=True,
            shell=True,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    devices = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, memory_mb, driver = [part.strip() for part in line.split(",", maxsplit=2)]
        devices.append({"name": name, "memory_mb": int(memory_mb), "driver": driver})
    return {"available": bool(devices), "devices": devices}


def inspect_hardware() -> dict:
    gpu = _nvidia_smi()
    max_gpu_mem_gib = 0.0
    if gpu.get("available"):
        max_gpu_mem_gib = max(device["memory_mb"] for device in gpu["devices"]) / 1024
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "total_ram_gib": _total_ram_gib(),
        "nvidia_smi": gpu,
        "full_parameter_qwen2p5_1p5b_feasibility": {
            "realistically_feasible_here": bool(max_gpu_mem_gib >= 24),
            "reason": (
                "Estimated full-parameter bf16 Qwen2.5-1.5B training with optimizer states "
                "is most realistic on a CUDA GPU with at least 24 GiB VRAM; 8-bit AdamW and "
                "gradient checkpointing reduce memory but do not make CPU/no-GPU training practical."
            ),
        },
    }
