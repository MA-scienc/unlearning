"""Pre-training hardware checks shared by every training stage.

The scientific recipe fixes bf16 full-parameter training with bitsandbytes
8-bit AdamW, so CUDA, bf16 support and bitsandbytes are always required. The
GPU-memory minimum is machine policy and comes from ``.env``
(``UNLEARNING_MIN_GPU_MEMORY_GB``) instead of being hard-coded per model.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from unlearning.runtime.settings import MachineSettings
from unlearning.utils.env import inspect_hardware


def precheck_gpu_training(settings: MachineSettings, *, purpose: str) -> dict[str, Any]:
    """Inspect hardware and record which training requirements pass."""

    hardware = inspect_hardware(disk_path=str(_existing_parent(settings.storage_root)))
    torch_info = hardware.get("torch", {})
    devices = torch_info.get("devices") or []
    min_memory_mb = int(settings.min_gpu_memory_gb * 1024)
    smallest_device_mb = min((int(device.get("total_memory_mb", 0)) for device in devices), default=0)
    free_disk_gb = float(hardware.get("disk", {}).get("free_gib") or 0.0)
    checks = {
        "torch_importable": bool(torch_info.get("available")),
        "cuda_available": bool(torch_info.get("cuda_available")),
        "bitsandbytes_importable": bool(hardware.get("bitsandbytes", {}).get("available")),
        "bf16_supported_on_all_gpus": bool(devices)
        and all(bool(device.get("bf16_supported")) for device in devices),
        "gpu_memory_meets_env_minimum": bool(devices) and smallest_device_mb >= min_memory_mb,
        "free_disk_meets_env_minimum": free_disk_gb >= settings.min_free_disk_gb,
    }
    hardware["training_precheck"] = {
        "purpose": purpose,
        "passed": all(checks.values()),
        "checks": checks,
        "min_gpu_memory_gb_from_env": settings.min_gpu_memory_gb,
        "smallest_visible_gpu_memory_gb": round(smallest_device_mb / 1024, 2),
        "min_free_disk_gb_from_env": settings.min_free_disk_gb,
        "free_disk_gb": free_disk_gb,
    }
    return hardware


def require_gpu_training(hardware: dict[str, Any]) -> None:
    precheck = hardware.get("training_precheck") or {}
    if precheck.get("passed"):
        return
    failed = [name for name, ok in (precheck.get("checks") or {}).items() if not ok]
    raise RuntimeError(
        f"Hardware precheck failed for {precheck.get('purpose', 'training')}: {', '.join(failed)}. "
        f"GPU memory minimum (UNLEARNING_MIN_GPU_MEMORY_GB) = {precheck.get('min_gpu_memory_gb_from_env')} GB, "
        f"smallest visible GPU = {precheck.get('smallest_visible_gpu_memory_gb')} GB; "
        f"free disk = {precheck.get('free_disk_gb')} GB (minimum "
        f"UNLEARNING_MIN_FREE_DISK_GB = {precheck.get('min_free_disk_gb_from_env')}). "
        "Refusing to load model weights."
    )


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(_existing_parent(path)).free / (1024**3)


def require_free_disk(path: Path, settings: MachineSettings, *, action: str) -> None:
    free = free_disk_gb(path)
    if free < settings.min_free_disk_gb:
        raise RuntimeError(
            f"Only {free:.1f} GB free at {path}; {action} needs at least "
            f"UNLEARNING_MIN_FREE_DISK_GB = {settings.min_free_disk_gb} GB. Free space and rerun "
            "with --resume; the last saved checkpoint is intact."
        )


def _existing_parent(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate
