"""Learning rates pinned from the pilot's calibration reports (Stage 2, section 3).

Stage 2 does not recalibrate: matched contamination and matched-forgetting unlearning
read the rate the pilot selected, from byte-identical copies of the pilot's reports
committed under ``configs/stage2/pinned/``. A pinned report is used only if its sha256
matches the value the caller supplies, so a changed or swapped file is refused.
"""

from __future__ import annotations

from pathlib import Path

from unlearning.utils.hashing import hash_file


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
