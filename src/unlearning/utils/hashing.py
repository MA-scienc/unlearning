from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_payload(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def fingerprint_records(records: Iterable[dict[str, Any]]) -> str:
    return fingerprint_payload(list(records))


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_directory(path: str | Path) -> str:
    """Fingerprint all regular files below an artifact directory.

    Checkpoint manifests are useful only if the local model and tokenizer files
    they name are tied to the manifest as well.  The fingerprint includes each
    relative path, byte size, and file SHA-256 in a canonical sorted payload,
    so renaming, adding, removing, or changing an artifact is detectable.
    """

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory to fingerprint does not exist: {root}")
    files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"Directory to fingerprint contains no files: {root}")
    inventory = [
        {
            "path": candidate.relative_to(root).as_posix(),
            "size_bytes": candidate.stat().st_size,
            "sha256": hash_file(candidate),
        }
        for candidate in files
    ]
    return fingerprint_payload({"schema_version": "directory-fingerprint-v1", "files": inventory})
