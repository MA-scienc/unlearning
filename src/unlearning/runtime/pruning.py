"""Remove training_state.pt from finished unlearning snapshots, leaving an audit marker.

Every snapshot stores ``training_state.pt`` (optimizer, scheduler and RNG state,
about 1 GB for Qwen2.5-0.5B). It is read only to resume an interrupted training
run. Once a trajectory has finished (its ``run_manifest.json`` says completed)
the stage is never resumed again, and evaluation reads only ``model/`` and
``tokenizer/``. Pruning removes the file and writes ``training_state.pruned.json``
in its place with the file's SHA-256 (checked against the manifest before
deletion), its size, the time, the reason and the code version.

Effects on what can be checked later:

* the model and tokenizer stay fully verifiable against the manifest;
* the deleted state can no longer be re-hashed, so provenance falls back to the
  hash recorded at pruning time (which matched the manifest);
* that snapshot cannot be used to resume training with the exact optimizer
  state. Re-running the trajectory from its C_i recreates an equivalent one: same
  data order and settings; GPU kernels are not deterministic, so weights may differ
  slightly (the benchmark's resume round-trip matched the loss exactly with a
  parameter-checksum difference of 0.014).

Only ``unlearning_snapshot`` artifacts are pruned. Clean S_i and contaminated
C_i checkpoints are inputs to later stages and keep their training state.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unlearning.runtime.resume import TRAINING_STATE_FILE
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import hash_file
from unlearning.utils.io import read_json

PRUNED_MARKER_FILE = "training_state.pruned.json"
MARKER_SCHEMA = "pruned-training-state-v1"
PRUNABLE_ARTIFACT_TYPE = "unlearning_snapshot"
DEFAULT_REASON = (
    "storage: the trajectory is complete, so this optimizer/scheduler/RNG state is never "
    "read again (it is only used to resume an interrupted training run)"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def read_pruned_marker(checkpoint_root: Path) -> dict[str, Any] | None:
    """The validated pruning marker of a checkpoint, or ``None`` if it has none."""

    path = Path(checkpoint_root) / PRUNED_MARKER_FILE
    if not path.is_file():
        return None
    marker = read_json(path)
    if not isinstance(marker, dict) or marker.get("schema_version") != MARKER_SCHEMA:
        raise ValueError(f"{path} is not a {MARKER_SCHEMA} marker")
    if marker.get("file") != TRAINING_STATE_FILE or not _SHA256.match(str(marker.get("sha256", ""))):
        raise ValueError(f"{path} does not record a training-state SHA-256")
    return marker


def training_state_fingerprint(checkpoint_root: Path, recorded: str | None) -> tuple[str, bool]:
    """``(sha256, pruned)`` for a checkpoint's training state.

    The file is hashed when present. When it was pruned, the marker's hash is
    returned, and it must equal ``recorded`` (the manifest's fingerprint).
    """

    root = Path(checkpoint_root)
    state = root / TRAINING_STATE_FILE
    if state.is_file():
        return hash_file(state), False
    marker = read_pruned_marker(root)
    if marker is None:
        raise FileNotFoundError(f"Checkpoint training state was not found: {state}")
    if recorded is not None and marker["sha256"] != recorded:
        raise ValueError(f"{root / PRUNED_MARKER_FILE} records a hash that differs from the manifest")
    return marker["sha256"], True


@dataclass(frozen=True)
class PruneCandidate:
    snapshot_dir: Path
    checkpoint_id: str
    recorded_sha256: str
    size_bytes: int


def _trajectory_complete(trajectory: Path) -> bool:
    run_manifest = trajectory / "run_manifest.json"
    return run_manifest.is_file() and read_json(run_manifest).get("status") == "completed"


def find_prunable(
    stage_root: Path, *, trajectory_prefix: str = ""
) -> tuple[list[PruneCandidate], list[tuple[Path, str]]]:
    """Snapshots under ``stage_root/<trajectory>/<snapshot>`` that may be pruned, and the skipped ones.

    ``trajectory_prefix`` limits the search to one experiment's trajectories.
    """

    candidates: list[PruneCandidate] = []
    skipped: list[tuple[Path, str]] = []
    if not Path(stage_root).is_dir():
        return candidates, skipped
    trajectories = (p for p in Path(stage_root).iterdir() if p.is_dir() and p.name.startswith(trajectory_prefix))
    for trajectory in sorted(trajectories):
        complete = _trajectory_complete(trajectory)
        for manifest_path in sorted(trajectory.glob("*/manifest.json")):
            snapshot = manifest_path.parent
            state = snapshot / TRAINING_STATE_FILE
            if not state.is_file():
                continue
            if not complete:
                skipped.append((snapshot, "trajectory has no completed run_manifest.json"))
                continue
            manifest = read_json(manifest_path)
            if (manifest.get("manifest") or {}).get("artifact_type") != PRUNABLE_ARTIFACT_TYPE:
                skipped.append((snapshot, "not an unlearning snapshot"))
                continue
            recorded = (manifest.get("artifact_fingerprints") or {}).get("training_state_file")
            if not isinstance(recorded, str) or not _SHA256.match(recorded):
                skipped.append((snapshot, "manifest records no training-state fingerprint"))
                continue
            candidates.append(
                PruneCandidate(
                    snapshot_dir=snapshot,
                    checkpoint_id=str(manifest.get("checkpoint_id")),
                    recorded_sha256=recorded,
                    size_bytes=state.stat().st_size,
                )
            )
    return candidates, skipped


def prune_training_state(candidate: PruneCandidate, *, reason: str = DEFAULT_REASON) -> dict[str, Any]:
    """Verify the file against its manifest, write the marker, then delete the file."""

    import json

    snapshot = candidate.snapshot_dir
    state = snapshot / TRAINING_STATE_FILE
    observed = hash_file(state)
    if observed != candidate.recorded_sha256:
        raise ValueError(f"{state} does not match its manifest fingerprint; it was not pruned")
    record = {
        "schema_version": MARKER_SCHEMA,
        "file": TRAINING_STATE_FILE,
        "sha256": observed,
        "size_bytes": candidate.size_bytes,
        "checkpoint_id": candidate.checkpoint_id,
        "manifest_file_sha256": hash_file(snapshot / "manifest.json"),
        "pruned_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "code_version": code_version(),
        "recreate_by": "re-running this unlearning trajectory from its contaminated checkpoint",
    }
    marker = snapshot / PRUNED_MARKER_FILE
    temporary = marker.with_name(marker.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    state.unlink()
    return {**record, "snapshot_dir": str(snapshot)}


LEDGER_FILE = "pruned_training_states.jsonl"


def prune_completed_trajectories(
    stage_root: Path, ledger: Path, *, trajectory_prefix: str = "", reason: str = DEFAULT_REASON, log=print
) -> list[dict[str, Any]]:
    """Prune every eligible snapshot and append one ledger line per deleted file."""

    import json

    candidates, _ = find_prunable(stage_root, trajectory_prefix=trajectory_prefix)
    pruned = []
    ledger.parent.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates, start=1):
        record = prune_training_state(candidate, reason=reason)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        pruned.append(record)
        log(f"[{index}/{len(candidates)}] pruned {candidate.size_bytes / 2**30:.2f} GiB  {candidate.snapshot_dir}")
    return pruned


def main(argv: list[str] | None = None) -> int:
    import argparse

    from unlearning.config import load_config
    from unlearning.runtime.layout import StorageLayout
    from unlearning.runtime.settings import default_config_path

    parser = argparse.ArgumentParser(
        description="Delete training_state.pt from finished unlearning snapshots (dry run unless --apply)."
    )
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--apply", action="store_true", help="delete the files; without it only list them")
    parser.add_argument("--reason", default=DEFAULT_REASON)
    args = parser.parse_args(argv)
    layout = StorageLayout.for_config(load_config(args.config))
    prefix = f"{layout.run_id}__"
    candidates, skipped = find_prunable(layout.unlearned_root, trajectory_prefix=prefix)
    total = sum(candidate.size_bytes for candidate in candidates)
    print(f"{len(candidates)} prunable training states, {total / 2**30:.1f} GiB, under {layout.unlearned_root}")
    for snapshot, why in skipped:
        print(f"  skipped {snapshot}: {why}")
    if not args.apply:
        for candidate in candidates:
            print(f"  would prune {candidate.size_bytes / 2**30:.2f} GiB  {candidate.snapshot_dir}")
        print("Dry run: nothing deleted. Re-run with --apply to delete.")
        return 0
    ledger = layout.runs / "maintenance" / LEDGER_FILE
    pruned = prune_completed_trajectories(layout.unlearned_root, ledger, trajectory_prefix=prefix, reason=args.reason)
    print(f"Pruned {len(pruned)} files ({sum(r['size_bytes'] for r in pruned) / 2**30:.1f} GiB). Ledger: {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
