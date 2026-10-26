from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from unlearning.experiments.pipeline import GATES, PilotPipeline, approve_gate, run_pipeline
from unlearning.runtime.settings import REPO_ROOT, MachineSettings

CONFIG = REPO_ROOT / "configs" / "pilot" / "qwen2.5-0.5b_laptop.yaml"
LAPTOP = REPO_ROOT / "configs" / "runtime" / "laptop_8gb.yaml"
DDP = REPO_ROOT / "configs" / "runtime" / "cloud_ddp_4gpu.yaml"


class PipelinePlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = MachineSettings.from_env({"UNLEARNING_STORAGE_ROOT": self.temporary.name})

    def tearDown(self):
        self.temporary.cleanup()

    def pipeline(self, runtime: Path = LAPTOP) -> PilotPipeline:
        return PilotPipeline(config_path=CONFIG, runtime_path=runtime, settings=self.settings)

    def test_plan_covers_every_stage_model_and_suite_in_order(self):
        jobs = self.pipeline().jobs()
        ids = [job.id for job in jobs]
        self.assertEqual(len(ids), len(set(ids)))
        # 3 clean x 6 suites + 3 fixed C x 5 + 3 matched C x 5 + 30 unlearned snapshots x 5
        self.assertEqual(sum(job.id.startswith("eval__") for job in jobs), 18 + 15 + 15 + 150)
        order = ["preflight", "gate0_data", "benchmark", "gate1_hardware", "specialize",
                 "gate2_specialization", "contaminate_C0", "calibrate_MM", "gate_calibration",
                 "contaminate_MM_C0", "eval__MM_C0__forgetting", "gate3_memorization",
                 "unlearn_MM_C100_npo", "eval__MM_U_C100_npo_step0400__near", "aggregate"]
        self.assertFalse(any(job.id.startswith("unlearn_C") for job in jobs))  # fixed arm is not unlearned
        self.assertEqual([ids.index(name) for name in order], sorted(ids.index(name) for name in order))
        self.assertEqual({job.gate for job in jobs if job.kind == "gate"}, set(GATES))

    def test_only_download_jobs_are_online(self):
        online = [job.id for job in self.pipeline().jobs() if not job.offline]
        self.assertEqual(online, ["preflight", "benchmark"])

    def test_training_jobs_always_resume_and_use_accelerate_when_configured(self):
        laptop = {job.id: job for job in self.pipeline().jobs()}
        self.assertIn("--resume", laptop["specialize"].command)
        self.assertNotIn("accelerate.commands.launch", laptop["specialize"].command)
        ddp = {job.id: job for job in self.pipeline(DDP).jobs()}
        self.assertIn("accelerate.commands.launch", ddp["unlearn_MM_C0_ga"].command)
        self.assertIn("accelerate.commands.launch", ddp["contaminate_MM_C50"].command)
        self.assertNotIn("accelerate.commands.launch", ddp["calibrate_MM"].command)  # single process
        self.assertNotIn("accelerate.commands.launch", ddp["eval__C0__near"].command)

    def test_run_stops_at_an_unapproved_gate_and_approval_requires_its_report(self):
        pipeline = self.pipeline()
        self.assertEqual(approve_gate(pipeline, "gate0_data", None), 2)  # not reached yet
        pipeline.layout.preflight_report.parent.mkdir(parents=True)
        pipeline.layout.preflight_report.write_text("{}", encoding="utf-8")
        self.assertEqual(run_pipeline(pipeline, dry_run=True), 0)
        self.assertTrue(pipeline.gate_report_path("gate0_data").is_file())
        self.assertNotIn("gate0_data", pipeline.approvals())
        self.assertEqual(approve_gate(pipeline, "gate0_data", "checked"), 0)
        self.assertEqual(pipeline.approvals()["gate0_data"]["note"], "checked")


if __name__ == "__main__":
    unittest.main()
