from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _assert_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def _temporary_sibling(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp-{os.getpid()}")


def replace_with_retry(source: Path, target: Path, attempts: int = 20, delay: float = 0.05) -> None:
    """``os.replace`` that tolerates a reader briefly holding ``target`` open on Windows."""

    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _atomic_write_lines(target: Path, lines: Iterable[str]) -> None:
    """Write to a temporary sibling, fsync, then rename over ``target``.

    A crash can therefore leave either the previous file or the complete new
    file, never a truncated artifact that a later "already done" check would
    mistake for a finished result.
    """

    temporary = _temporary_sibling(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: str | Path, payload: Any, overwrite: bool = False) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    _assert_can_write(target, overwrite)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    _atomic_write_lines(target, [text])


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], overwrite: bool = False) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    _assert_can_write(target, overwrite)
    _atomic_write_lines(
        target,
        (
            json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
            for record in records
        ),
    )


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record to an append-only log such as ``metrics.jsonl``."""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
