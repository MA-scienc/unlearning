"""Golden identity of the seed-42 pilot (Stage 1). Stage-2 code must never change it.

The values below were read from the pilot's own artifacts on 2026-11-10 (calibration
reports, trajectory plans, the finished 568-job pipeline) and regenerated from code and
config with identical results. If any of them changes, the pipeline would treat finished
pilot jobs as missing, demand recalibration, or break the provenance of pilot results.
A deliberate change needs a dated amendment, never an edit of these constants.
"""

from __future__ import annotations

import tempfile
import unittest

from unlearning.config import load_config
from unlearning.config.arms import load_matched_memorization_arm
from unlearning.config.unlearning_revisions import load_unlearning_revision
from unlearning.data.synthetic_forget import generate_forget_dataset
from unlearning.experiments.naming import experiment_run_id
from unlearning.experiments.pipeline import PilotPipeline
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.settings import REPO_ROOT, MachineSettings
from unlearning.training.matched_contamination import DEFAULT_ARM_PATH
from unlearning.training.unlearn import _method_config, deterministic_batch_schedule, validate_forget_examples
from unlearning.utils.hashing import fingerprint_payload, fingerprint_records

CONFIG = REPO_ROOT / "configs" / "pilot" / "qwen2.5-0.5b_laptop.yaml"
RUNTIME = REPO_ROOT / "configs" / "runtime" / "laptop_8gb.yaml"

PILOT_RUN_ID = (
    "pilot-medical-s0-s50-s100-qwen05b__qwen-qwen2-5-0-5b__seed42__synthetic-patient-f-v2__cfg2396728f114a"
)
ARM_FINGERPRINT = "27499cbc376059fbde6954d01b635199c4bf161717b422cb5263c6a56dbace7b"
REVISION_FINGERPRINTS = {
    "revision2.yaml": "239942fd56e52c3a39004ced2616cf142a8529301846b13467a6e8bfdf04128d",
    "revision3.yaml": "272637f8a0f0d210059c68fe192ece88c0e23139dfc9d5b486bebcc54cbe8d1b",
}
F_CONTAMINATION_FINGERPRINT = "edb06902d2a276bbe81fbf5fe3c979a2f5c563cff1de95bb9260323ce5211cb1"
F_CANONICAL_ORDER_FINGERPRINT = "20edaace5f01ac6cd5b875265b550bb44014cf035e4c95de29d5f24fd979eebf"
BATCH_SCHEDULE_FINGERPRINTS = {
    400: "4d6bb31e95e1a0baebc7d48fa0f34ff32e864b8b443ec782ab906ad966806696",  # revisions 1 and 2
    2000: "68e263f2a76856b4c41ec4f4c3ef11fd6b010518244e1a9f38b06e358a9fbff5",  # revision 3 (cap)
}
PILOT_JOB_COUNT = 568
PILOT_JOB_LIST_FINGERPRINT = "c5e1b6f48342bdcd34396bd2acfec3b7c3146d9df3a0ed1ed2c1d83400e09c18"
U3_C50_GA_TRAJECTORY = f"{PILOT_RUN_ID}__unlearned-u3__s050__matched-memorization-replay__ga"


class PilotIdentityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = MachineSettings.from_env({"UNLEARNING_STORAGE_ROOT": self.temporary.name})

    def tearDown(self):
        self.temporary.cleanup()

    def test_run_id(self):
        self.assertEqual(experiment_run_id(self.config), PILOT_RUN_ID)

    def test_arm_and_revision_fingerprints(self):
        self.assertEqual(load_matched_memorization_arm(DEFAULT_ARM_PATH).fingerprint, ARM_FINGERPRINT)
        for name, fingerprint in REVISION_FINGERPRINTS.items():
            revision = load_unlearning_revision(REPO_ROOT / "configs" / "pilot" / "unlearning" / name)
            self.assertEqual(revision.fingerprint, fingerprint, name)

    def test_forget_set_and_unlearning_batch_schedules(self):
        dataset = generate_forget_dataset(self.config.forget)
        self.assertEqual(fingerprint_records(dataset.contamination_examples), F_CONTAMINATION_FINGERPRINT)
        canonical = validate_forget_examples(dataset.contamination_examples)
        self.assertEqual(fingerprint_payload(canonical), F_CANONICAL_ORDER_FINGERPRINT)
        ids = tuple(example["example_id"] for example in canonical)
        batch = _method_config(self.config, "ga").effective_batch_size
        for steps, fingerprint in BATCH_SCHEDULE_FINGERPRINTS.items():
            schedule = deterministic_batch_schedule(example_ids=ids, training_steps=steps, batch_size=batch)
            self.assertEqual(fingerprint_payload(schedule), fingerprint, steps)

    def test_pipeline_job_list(self):
        jobs = PilotPipeline(config_path=CONFIG, runtime_path=RUNTIME, settings=self.settings).jobs()
        ids = [job.id for job in jobs]
        self.assertEqual(len(ids), PILOT_JOB_COUNT)
        self.assertEqual(fingerprint_payload(ids), PILOT_JOB_LIST_FINGERPRINT)

    def test_storage_names(self):
        layout = StorageLayout.for_config(self.config, self.settings)
        arm = load_matched_memorization_arm(DEFAULT_ARM_PATH)
        trajectory = layout.unlearning_trajectory(self.config, "C50", "ga", condition=arm.name, revision_tag="U3")
        self.assertEqual(trajectory.name, U3_C50_GA_TRAJECTORY)
        self.assertEqual(layout.clean_checkpoint("S50").name, "S50")
        self.assertEqual(layout.clean_checkpoint("S50").parent.name, PILOT_RUN_ID)


if __name__ == "__main__":
    unittest.main()
