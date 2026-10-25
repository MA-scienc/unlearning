"""Machine-local settings read from the environment and the repo's ``.env`` file.

These settings say *where* and *on what machine* the experiment runs: storage
location, Hugging Face cache, hardware guard rails, and which experiment and
runtime files to use. They never change the scientific recipe (model, data,
hyperparameters), which lives only in ``configs/pilot/*.yaml`` and is
fingerprinted into run identifiers. Settings are recorded in run manifests so a
run can always be traced back to the machine configuration that produced it.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
FALLBACK_CONFIG_PATH = "configs/pilot/base.yaml"
FALLBACK_RUNTIME_PATH = "configs/runtime/laptop_8gb.yaml"


def load_local_env(path: str | Path | None = None) -> bool:
    """Load ``.env`` from the repo root; variables already set in the shell win."""

    env_path = Path(path) if path is not None else REPO_ROOT / ".env"
    if not env_path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(env_path, override=False)


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a possibly relative path from ``.env`` against the repo root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _float_env(environ: Mapping[str, str], name: str, default: float | None) -> float | None:
    raw = (environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class MachineSettings:
    config_path: Path
    runtime_profile_path: Path
    storage_root: Path
    hf_home: Path
    # Hardware guard rails. A GPU below min_gpu_memory_gb is refused before any
    # weights load; 0 disables the check (an out-of-memory error still stops
    # the run, and the benchmark measures the real requirement).
    min_gpu_memory_gb: float
    # Pause training between optimizer steps while the GPU is at or above this
    # temperature, until it cools to gpu_resume_temp_c. None disables pausing.
    gpu_pause_temp_c: float | None
    gpu_resume_temp_c: float
    # Refuse to start, and stop before a checkpoint save, below this free disk.
    min_free_disk_gb: float

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "MachineSettings":
        env = os.environ if environ is None else environ
        storage_root = resolve_repo_path(env.get("UNLEARNING_STORAGE_ROOT") or ".")
        hf_home_raw = env.get("HF_HOME")
        hf_home = resolve_repo_path(hf_home_raw) if hf_home_raw else storage_root / "hf_cache"
        pause = _float_env(env, "UNLEARNING_GPU_PAUSE_TEMP_C", 90.0)
        if pause is not None and pause <= 0:
            pause = None
        resume_temp = _float_env(env, "UNLEARNING_GPU_RESUME_TEMP_C", 80.0)
        if pause is not None and resume_temp is not None and resume_temp >= pause:
            raise ValueError("UNLEARNING_GPU_RESUME_TEMP_C must be below UNLEARNING_GPU_PAUSE_TEMP_C")
        min_gpu = _float_env(env, "UNLEARNING_MIN_GPU_MEMORY_GB", 0.0)
        min_disk = _float_env(env, "UNLEARNING_MIN_FREE_DISK_GB", 10.0)
        if min_gpu is None or min_gpu < 0:
            raise ValueError("UNLEARNING_MIN_GPU_MEMORY_GB must be >= 0")
        if min_disk is None or min_disk < 0:
            raise ValueError("UNLEARNING_MIN_FREE_DISK_GB must be >= 0")
        return cls(
            config_path=resolve_repo_path(env.get("UNLEARNING_CONFIG") or FALLBACK_CONFIG_PATH),
            runtime_profile_path=resolve_repo_path(
                env.get("UNLEARNING_RUNTIME") or FALLBACK_RUNTIME_PATH
            ),
            storage_root=storage_root,
            hf_home=hf_home,
            min_gpu_memory_gb=float(min_gpu),
            gpu_pause_temp_c=pause,
            gpu_resume_temp_c=float(resume_temp if resume_temp is not None else 80.0),
            min_free_disk_gb=float(min_disk),
        )

    def as_dict(self) -> dict:
        return {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(self).items()}


def apply_hf_cache_default(environ: dict[str, str] | None = None) -> None:
    """Keep Hugging Face downloads inside the storage root unless HF_HOME is set.

    Must run before ``huggingface_hub`` is imported, which is why the package
    ``__init__`` calls it.
    """

    env = os.environ if environ is None else environ
    if not env.get("HF_HOME"):
        storage_root = resolve_repo_path(env.get("UNLEARNING_STORAGE_ROOT") or ".")
        env["HF_HOME"] = str(storage_root / "hf_cache")


def default_config_path() -> str:
    """Experiment config used when --config is omitted ($UNLEARNING_CONFIG)."""

    return str(MachineSettings.from_env().config_path)


def default_runtime_path() -> str:
    """Runtime profile used when --runtime is omitted ($UNLEARNING_RUNTIME)."""

    return str(MachineSettings.from_env().runtime_profile_path)
