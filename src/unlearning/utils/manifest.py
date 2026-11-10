from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import fingerprint_payload


def build_manifest(
    artifact_type: str,
    schema_version: str,
    payload: dict[str, Any],
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": source_metadata,
        "payload_fingerprint": fingerprint_payload(payload),
        # Outside the payload fingerprint: recording the code version never changes an artifact's identity.
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "code_version": code_version(),
        },
    }
    return manifest
