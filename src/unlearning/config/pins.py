"""Learning rates pinned from the pilot's calibration reports (Stage 2, section 3).

Stage 2 does not recalibrate: matched contamination and matched-forgetting unlearning
read the rate the pilot selected, from byte-identical copies of the pilot's reports
committed under ``configs/stage2/pinned/``. A pinned report is used only if its sha256
matches the value the caller supplies, so a changed or swapped file is refused.
"""

from __future__ import annotations

from pathlib import Path

from unlearning.runtime.settings import REPO_ROOT
from unlearning.utils.hashing import hash_file

PINNED_DIR = REPO_ROOT / "configs" / "stage2" / "pinned"
# (file, sha256) of the pilot's reports; see configs/stage2/pinned/README.md.
PINNED_CONTAMINATION = (
    PINNED_DIR / "contamination_MR_calibration_report.json",
    "c4b5d773aa439c12c3deb83b16be8d3abed9109ec0d7844f4d883ca17b0b0cf7",
)
PINNED_UNLEARNING = (
    PINNED_DIR / "unlearning_U3_calibration_report.json",
    "8cbcf00cd15d72de4c13da5ebac7fb40e2e84b3835a6b45f0c2dd38af760da83",
)


def pinned_args(pin: tuple[Path, str]) -> list[str]:
    """Command-line arguments that make a stage read a pinned report."""

    return ["--calibration-report", str(pin[0]), "--calibration-report-sha256", pin[1]]


def verified_pinned_report(path: str | Path, expected_sha256: str | None) -> Path:
    """The report path, after checking it exists and has the expected sha256."""

    report = Path(path)
    if not expected_sha256:
        raise ValueError("A pinned calibration report needs its expected sha256 (--calibration-report-sha256)")
    if not report.is_file():
        raise FileNotFoundError(f"Pinned calibration report not found: {report}")
    observed = hash_file(report)
    if observed != expected_sha256.lower():
        raise ValueError(f"Pinned calibration report {report} has sha256 {observed}, expected {expected_sha256}")
    return report
