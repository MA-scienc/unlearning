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
        # 3 clean x 6 suites + 3 fixed C x 5 + 3 matched (replay) C x 5 + 2 unlearning revisions x 30 snapshots x 5
        self.assertEqual(sum(job.id.startswith("eval__") for job in jobs), 18 + 15 + 15 + 150 + 150)
        order = ["preflight", "gate0_data", "benchmark", "gate1_hardware", "specialize",
                 "gate2_specialization", "contaminate_C0", "prepare_replay_MR", "calibrate_MR",
                 "gate_calibration", "contaminate_MR_C0", "eval__MR_C0__forgetting", "gate3_memorization",
                 "unlearn_MR_C100_npo", "eval__MR_U_C100_npo_step0400__near",
                 "unlearn_MR_U2_C0_ga", "eval__MR_U2_C100_npo_step0400__near", "aggregate"]
        self.assertFalse(any(job.id.startswith("unlearn_C") for job in jobs))  # fixed arm is not unlearned
        self.assertEqual([ids.index(name) for name in order], sorted(ids.index(name) for name in order))
        self.assertEqual({job.gate for job in jobs if job.kind == "gate"}, set(GATES))

    def test_calibration_and_replay_from_other_arm_settings_are_not_done(self):
        from unlearning.data.replay import replay_dir
        from unlearning.training.matched_contamination import calibration_report_path
        from unlearning.utils.io import write_json

        pipeline = self.pipeline()
        arm, replay = pipeline.arm, pipeline.arm.replay
        jobs = {job.id: job for job in pipeline.jobs()}
        report = calibration_report_path(pipeline.layout, arm)
        write_json(report, {"status": "no_candidate_qualified", "arm_fingerprint": "other"})
        self.assertFalse(jobs[f"calibrate_{arm.tag}"].done({}))
        write_json(report, {"status": "no_candidate_qualified", "arm_fingerprint": arm.fingerprint}, overwrite=True)
        self.assertTrue(jobs[f"calibrate_{arm.tag}"].done({}))

        manifest = {
            "dataset_id": replay.dataset_id, "dataset_config": replay.dataset_config, "revision": replay.revision,
            "split": replay.split, "sequence_length": replay.sequence_length, "count": replay.prepared_sequences // 2,
            "shuffle_seed": replay.shuffle_seed, "tokenizer_id": pipeline.config.model.tokenizer_id,
            "tokenizer_revision": pipeline.config.model.revision,
        }
        path = replay_dir(pipeline.layout.generated, arm) / "manifest.json"
        write_json(path, manifest)
        self.assertFalse(jobs[f"prepare_replay_{arm.tag}"].done({}))
        write_json(path, {**manifest, "count": replay.prepared_sequences}, overwrite=True)
        self.assertTrue(jobs[f"prepare_replay_{arm.tag}"].done({}))

    def test_unlearning_revision_adds_its_own_jobs_and_leaves_revision_1_unchanged(self):
        jobs = {job.id: job for job in self.pipeline().jobs()}
        base, revised = jobs["unlearn_MR_C50_npo"], jobs["unlearn_MR_U2_C50_npo"]
        self.assertNotIn("--unlearning-revision", base.command)
        flag = revised.command.index("--unlearning-revision")
        self.assertEqual(Path(revised.command[flag + 1]).name, "revision2.yaml")
        self.assertEqual((base.group, revised.group), ("unlearning", "unlearning_U2"))
        self.assertIn("__unlearned__s050__", str(base.outputs[0]))
        self.assertIn("__unlearned-u2__s050__", str(revised.outputs[0]))
        self.assertEqual(jobs["eval__MR_U2_C50_npo_step0400__near"].group, "unlearned_eval_U2")
        self.assertIn("__unlearned-u2__", jobs["eval__MR_U2_C50_npo_step0400__near"].command[-5])

        with tempfile.TemporaryDirectory() as empty:
            only_revision_1 = PilotPipeline(
                config_path=CONFIG, runtime_path=LAPTOP, settings=self.settings, unlearning_revisions_dir=Path(empty)
            ).jobs()
        ids = [job.id for job in only_revision_1]
        self.assertIn("unlearn_MR_C0_ga", ids)
        self.assertFalse(any("U2" in job_id for job_id in ids))

    def test_only_download_jobs_are_online(self):
        online = [job.id for job in self.pipeline().jobs() if not job.offline]
        self.assertEqual(online, ["preflight", "benchmark", "prepare_replay_MR"])

    def test_training_jobs_always_resume_and_use_accelerate_when_configured(self):
        laptop = {job.id: job for job in self.pipeline().jobs()}
        self.assertIn("--resume", laptop["specialize"].command)
        self.assertNotIn("accelerate.commands.launch", laptop["specialize"].command)
        ddp = {job.id: job for job in self.pipeline(DDP).jobs()}
        self.assertIn("accelerate.commands.launch", ddp["unlearn_MR_C0_ga"].command)
        self.assertIn("accelerate.commands.launch", ddp["contaminate_MR_C50"].command)
        self.assertNotIn("accelerate.commands.launch", ddp["calibrate_MR"].command)  # single process
        self.assertNotIn("accelerate.commands.launch", ddp["eval__C0__near"].command)

    def test_retention_table_judges_by_bias_removed_accuracy_and_shows_raw(self):
        from unlearning.experiments.pipeline import _retention_table
        from unlearning.utils.io import write_json

        pipeline = self.pipeline()
        records = [{"option_logprobs": [-2.0 + (a == j) + 3.0 * (j == 2) for j in range(3)], "answer_index": a} for a in (0, 1, 2) * 4]
        write_json(pipeline.result_path("S0", "external_medical"), {"accuracy": 1 / 3, "count": 12, "records": records})
        lines = _retention_table(pipeline, ["S0", "S50"], ["external_medical"])
        self.assertIn("judges retention", lines[0])
        self.assertEqual(lines[-2], "| S0 | 1.000 [1.00, 1.00] (raw 0.333, n=12) |")
        self.assertEqual(lines[-1], "| S50 | missing |")

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
