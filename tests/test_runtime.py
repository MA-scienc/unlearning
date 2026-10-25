from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unlearning.runtime.profile import (
    RuntimeProfile,
    RuntimeProfileError,
    load_runtime_profile,
    micro_steps_per_rank,
)
from unlearning.runtime.resume import (
    RESUME_META_FILE,
    ResumePoint,
    ResumeStore,
    begin_atomic_dir,
    commit_atomic_dir,
    latest_point,
    remove_stale_temporaries,
)
from unlearning.runtime.settings import REPO_ROOT, MachineSettings
from unlearning.runtime.stage import (
    STAGE_COMPLETE,
    STAGE_FRESH,
    STAGE_RESUME,
    ExecutionRecord,
    StageAlreadyStarted,
    prepare_stage,
)


def profile_dict(**overrides) -> dict:
    raw = {
        "name": "test",
        "launcher": "python",
        "accelerate_config": None,
        "micro_batch": {"specialization": 2, "contamination": 4, "unlearning": 4, "npo_reference": 8},
        "checkpointing": {"resume_every_minutes": 30, "keep_last": 2},
        "logging": {"console_every_steps": 10, "system_sample_seconds": 30},
        "monitoring": {"specialization_dev_every_steps": 200, "specialization_dev_sequences": 32},
    }
    raw.update(overrides)
    return raw


class MachineSettingsTests(unittest.TestCase):
    def test_defaults_keep_everything_inside_the_repo(self):
        settings = MachineSettings.from_env({})
        self.assertEqual(settings.storage_root, REPO_ROOT)
        self.assertEqual(settings.hf_home, REPO_ROOT / "hf_cache")
        self.assertEqual(settings.min_gpu_memory_gb, 0.0)

    def test_gpu_memory_minimum_and_thermal_limits_come_from_env(self):
        settings = MachineSettings.from_env(
            {
                "UNLEARNING_MIN_GPU_MEMORY_GB": "7.5",
                "UNLEARNING_GPU_PAUSE_TEMP_C": "87",
                "UNLEARNING_GPU_RESUME_TEMP_C": "78",
                "UNLEARNING_STORAGE_ROOT": "D:/storage",
            }
        )
        self.assertEqual(settings.min_gpu_memory_gb, 7.5)
        self.assertEqual(settings.gpu_pause_temp_c, 87.0)
        self.assertEqual(settings.storage_root, Path("D:/storage"))
        self.assertEqual(settings.hf_home, Path("D:/storage") / "hf_cache")

    def test_zero_pause_temperature_disables_thermal_pausing(self):
        self.assertIsNone(MachineSettings.from_env({"UNLEARNING_GPU_PAUSE_TEMP_C": "0"}).gpu_pause_temp_c)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            MachineSettings.from_env({"UNLEARNING_MIN_GPU_MEMORY_GB": "lots"})
        with self.assertRaises(ValueError):
            MachineSettings.from_env({"UNLEARNING_GPU_PAUSE_TEMP_C": "80", "UNLEARNING_GPU_RESUME_TEMP_C": "85"})


class RuntimeProfileTests(unittest.TestCase):
    def test_repository_profiles_load(self):
        for name in ("laptop_8gb", "cloud_1gpu_80gb", "cloud_ddp_4gpu"):
            profile = load_runtime_profile(REPO_ROOT / "configs" / "runtime" / f"{name}.yaml")
            self.assertEqual(profile.name, name)

    def test_accelerate_launcher_requires_a_config(self):
        with self.assertRaises(RuntimeProfileError):
            RuntimeProfile.from_dict(profile_dict(launcher="accelerate"))

    def test_micro_batches_must_split_the_step_evenly_across_processes(self):
        self.assertEqual(micro_steps_per_rank(effective_batch=32, micro_batch=4, world_size=4, stage="s"), 2)
        with self.assertRaises(RuntimeProfileError):
            micro_steps_per_rank(effective_batch=16, micro_batch=3, world_size=1, stage="c")
        with self.assertRaises(RuntimeProfileError):
            micro_steps_per_rank(effective_batch=16, micro_batch=8, world_size=4, stage="u")

    def test_fingerprint_ignores_file_location(self):
        a = RuntimeProfile.from_dict(profile_dict(), source_path="a.yaml")
        b = RuntimeProfile.from_dict(profile_dict(), source_path="b.yaml")
        self.assertEqual(a.fingerprint, b.fingerprint)


class StageLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "out"
        self.logs = self.root / "logs"
        self.marker = self.output / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, **kwargs):
        options = {"resume": False, "overwrite": False}
        options.update(kwargs)
        return prepare_stage(final_marker=self.marker, paths=[self.output, self.logs], stage="t", **options)

    def test_fresh_partial_resume_and_complete(self):
        self.assertEqual(self.prepare(), STAGE_FRESH)
        (self.logs / "resume").mkdir(parents=True)
        with self.assertRaises(StageAlreadyStarted):
            self.prepare()
        self.assertEqual(self.prepare(resume=True), STAGE_RESUME)
        self.output.mkdir()
        self.marker.write_text("{}", encoding="utf-8")
        self.assertEqual(self.prepare(), STAGE_COMPLETE)
        self.assertEqual(self.prepare(resume=True), STAGE_COMPLETE)

    def test_overwrite_sets_old_outputs_aside_instead_of_deleting(self):
        self.output.mkdir()
        self.marker.write_text("{}", encoding="utf-8")
        self.assertEqual(self.prepare(overwrite=True), STAGE_FRESH)
        self.assertFalse(self.output.exists())
        archived = list(self.root.glob("out__invalidated_*"))
        self.assertEqual(len(archived), 1)
        self.assertTrue((archived[0] / "manifest.json").is_file())

    def test_execution_record_accumulates_segments(self):
        record = ExecutionRecord(self.root / "execution.json")
        record.begin(start_step=0, resumed_from=None, details={})
        record.end(end_step=10, state="interrupted", runtime_seconds=5.0)
        again = ExecutionRecord(self.root / "execution.json")
        again.begin(start_step=10, resumed_from={"step": 10}, details={})
        again.end(end_step=20, state="completed", runtime_seconds=7.0)
        payload = json.loads((self.root / "execution.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["resumed"])
        self.assertEqual(payload["segment_count"], 2)
        self.assertEqual(payload["total_runtime_seconds"], 12.0)

    def test_a_segment_that_never_ended_is_marked_crashed(self):
        record = ExecutionRecord(self.root / "execution.json")
        record.begin(start_step=0, resumed_from=None, details={})  # then the process is killed
        again = ExecutionRecord(self.root / "execution.json")
        again.begin(start_step=0, resumed_from=None, details={})
        self.assertEqual([segment["state"] for segment in again.segments], ["crashed", "running"])


class ResumeStoreTests(unittest.TestCase):
    def test_incomplete_saves_are_ignored_and_cleaned(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            complete = directory / "step000010"
            complete.mkdir()
            (complete / RESUME_META_FILE).write_text('{"step": 10}', encoding="utf-8")
            (directory / "step000020").mkdir()  # no meta file: crashed mid-save
            (directory / "step000030.tmp").mkdir()
            store = ResumeStore(directory, keep_last=2, every_minutes=1)
            self.assertEqual([point.step for point in store.points()], [10])
            removed = remove_stale_temporaries(directory)
            self.assertEqual([path.name for path in removed], ["step000030.tmp"])

    def test_atomic_directory_commit_refuses_to_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "S50"
            staging = begin_atomic_dir(final)
            (staging / "file").write_text("x", encoding="utf-8")
            commit_atomic_dir(staging, final)
            self.assertTrue((final / "file").is_file())
            with self.assertRaises(FileExistsError):
                commit_atomic_dir(begin_atomic_dir(final), final)

    def test_latest_point_prefers_newest_then_scientific(self):
        resume = ResumePoint(step=100, directory=Path("r"), kind="resume")
        science = ResumePoint(step=100, directory=Path("s"), kind="scientific")
        older = ResumePoint(step=50, directory=Path("o"), kind="scientific")
        self.assertIs(latest_point([resume, older], [science]), science)
        self.assertIsNone(latest_point([], []))


if __name__ == "__main__":
    unittest.main()
