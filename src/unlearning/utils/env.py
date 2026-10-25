"""Hardware and software inspection recorded in every manifest."""

from __future__ import annotations

import platform
import shutil
import subprocess
from importlib import metadata


def _bytes_to_gib(value: int) -> float:
    return round(value / (1024**3), 2)


def _disk_usage_gib(path: str = ".") -> dict:
    usage = shutil.disk_usage(path)
    return {
        "total_gib": _bytes_to_gib(usage.total),
        "used_gib": _bytes_to_gib(usage.used),
        "free_gib": _bytes_to_gib(usage.free),
    }


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


def _torch_info() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "import_error": str(exc)}
    cuda_available = bool(torch.cuda.is_available())
    info = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
    }
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            major, _minor = torch.cuda.get_device_capability(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_mb": int(props.total_memory / (1024**2)),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported())
                    if index == 0
                    else major >= 8,
                }
            )
    return info


def _bitsandbytes_info() -> dict:
    try:
        import bitsandbytes as bnb
    except Exception as exc:
        return {"available": False, "import_error": str(exc)}
    return {"available": True, "version": getattr(bnb, "__version__", None)}


def inspect_hardware(disk_path: str = ".") -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "total_ram_gib": _total_ram_gib(),
        "disk": _disk_usage_gib(disk_path),
        "nvidia_smi": _nvidia_smi(),
        "torch": _torch_info(),
        "bitsandbytes": _bitsandbytes_info(),
    }


_VERSIONED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "numpy",
    "nvidia-ml-py",
    "llm-specialization-unlearning",
)


def software_versions() -> dict:
    """Python, platform, CUDA and pinned library versions for reproducibility records."""

    versions: dict = {"python": platform.python_version(), "platform": platform.platform()}
    for package in _VERSIONED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import torch

        versions["torch_cuda"] = getattr(torch.version, "cuda", None)
        versions["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        versions["torch_cuda"] = None
    return versions
