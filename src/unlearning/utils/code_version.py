"""The code version (git commit) that produced an artifact or result.

Recorded in every manifest's ``software`` block and in evaluation results so a
measurement can be traced to the exact code. ``git_dirty`` is true when tracked
files had uncommitted changes at the time; untracked files are ignored. On a
machine without git, or outside a checkout, the fields are ``None`` and
``error`` says why. The value is read once per process.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()


@lru_cache(maxsize=1)
def _read_code_version() -> tuple[tuple[str, Any], ...]:
    try:
        commit = _git("rev-parse", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError) as exc:
        return (("git_commit", None), ("git_branch", None), ("git_dirty", None), ("error", type(exc).__name__))
    return (("git_commit", commit), ("git_branch", branch), ("git_dirty", dirty))


def code_version() -> dict[str, Any]:
    """``{"git_commit", "git_branch", "git_dirty"}`` (plus ``error`` if git was unavailable)."""

    return dict(_read_code_version())


# The Stage-2 pre-registration freeze. The record is written in the commit *after* the freeze
# commit (a commit cannot contain its own hash), so a run may come from a later commit as long as
# only documentation differs. "Code" is everything that runs, configures or tests an experiment.
FREEZE_RECORD = _REPO_ROOT / "docs" / "stage2" / "freeze.json"
FROZEN_CODE_PATHS = ("src", "scripts", "configs", "tests", "pyproject.toml", "requirements-lock.txt")


def frozen_code_status(commit: str | None, *, record: Path | None = None) -> str:
    """Whether ``commit`` has exactly the code of the freeze commit: "yes ...", "NO ..." or why it is unknown."""

    import json

    record = FREEZE_RECORD if record is None else record
    if not record.is_file():
        return "freeze record not found"
    freeze = json.loads(record.read_text(encoding="utf-8")).get("freeze_commit")
    if not commit or not freeze:
        return "NO (commit unknown)"
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "diff", "--quiet", freeze, commit, "--", *FROZEN_CODE_PATHS],
            capture_output=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not check ({type(exc).__name__})"
    if result.returncode == 0:
        return f"yes (same code as freeze commit {freeze[:7]})"
    if result.returncode == 1:
        return f"NO: code differs from freeze commit {freeze[:7]}"
    return f"could not check (git exit {result.returncode})"
