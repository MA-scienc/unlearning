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
        # 3 clean x 6 suites + 3 fixed C x 5 + 3 matched (replay) C x 5 + 2 fixed-budget revisions x 30
        # snapshots x 5 + matched forgetting (6 runs x 5 snapshots x 5 suites, closeness for 3 C_i + 30)
        self.assertEqual(sum(job.id.startswith("eval__") for job in jobs), 18 + 15 + 15 + 150 + 150 + 150 + 33)
        order = ["preflight", "gate0_data", "benchmark", "gate1_hardware", "specialize",
                 "gate2_specialization", "contaminate_C0", "prepare_replay_MR", "calibrate_MR",
                 "gate_calibration", "contaminate_MR_C0", "eval__MR_C0__forgetting", "gate3_memorization",
                 "unlearn_MR_C100_npo", "eval__MR_U_C100_npo_step0400__near",
                 "unlearn_MR_U2_C0_ga", "eval__MR_U2_C100_npo_step0400__near",
                 "calibrate_unlearning_U3", "gate_unlearning_calibration", "unlearn_MR_U3_C0_ga",
                 "eval__MR_U3_C100_npo_f100__near", "eval__MR_U3_C100_npo_final__closeness", "aggregate"]
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

    def test_matched_forgetting_revision_calibrates_gates_and_skips_unreached_thresholds(self):
        from unlearning.utils.io import write_json

        pipeline = self.pipeline()
        jobs = {job.id: job for job in pipeline.jobs()}
        calibrate = jobs["calibrate_unlearning_U3"]
        self.assertIn("calibrate_unlearning.py", " ".join(calibrate.command))
        self.assertEqual(Path(calibrate.command[calibrate.command.index("--unlearning-revision") + 1]).name, "revision3.yaml")
        self.assertNotIn("accelerate.commands.launch", calibrate.command)
        self.assertEqual(jobs["gate_unlearning_calibration"].gate, "gate_unlearning_calibration")
        self.assertIn("__unlearned-u3__s000__", str(jobs["unlearn_MR_U3_C0_ga"].outputs[0]))
        closeness = jobs["eval__MR_U3_C50_npo_f050__closeness"].command
        self.assertIn("evaluate_closeness.py", " ".join(closeness))
        self.assertTrue(closeness[closeness.index("--clean-manifest") + 1].endswith(str(Path("S50") / "manifest.json")))
        self.assertIn("eval__MR_C100__closeness", jobs)

        # Before the run: not done. After it: f100 not reached -> its evaluations are skipped, final runs.
        f100, final = jobs["eval__MR_U3_C0_ga_f100__near"], jobs["eval__MR_U3_C0_ga_final__near"]
        self.assertFalse(f100.done({}))
        run_manifest = jobs["unlearn_MR_U3_C0_ga"].outputs[0]
        write_json(run_manifest, {"snapshot_paths": {"f025": "x", "f050": "x", "f075": "x", "final": "x"}})
        self.assertTrue(f100.done({}))
        self.assertTrue(jobs["eval__MR_U3_C0_ga_f100__closeness"].done({}))
        self.assertFalse(final.done({}))

    def test_unlearning_calibration_gate_report(self):
        from unlearning.experiments.pipeline import write_gate_report
        from unlearning.training.matched_forgetting import calibration_report_path
        from unlearning.utils.io import write_json

        pipeline = self.pipeline()
        revision = pipeline.unlearning_revisions[-1]
        write_json(
            calibration_report_path(pipeline.layout, revision),
            {
                "status": "selected",
                "selection_rule": "smallest candidate reaching 1.0",
                "references": {"clean": {"mean": -4.2}, "contaminated": {"mean": -0.06}, "gap": 4.14},
                "methods": {
                    "ga": {
                        "status": "selected",
                        "selected_learning_rate": 5e-6,
                        "trials": [
                            {"learning_rate": 2e-6, "reached": {"f025": {"step": 300}}, "steps_run": 1000,
                             "last_check": {"gap_closed": 0.4}, "diverged": False},
                            {"learning_rate": 5e-6, "reached": {name: {"step": 40 * i} for i, name in enumerate(("f025", "f050", "f075", "f100"), 1)},
                             "steps_run": 160, "last_check": {"gap_closed": 1.02}, "diverged": False},
                        ],
                    }
                },
            },
        )
        text = write_gate_report(pipeline, "gate_unlearning_calibration").read_text(encoding="utf-8")
        self.assertIn("## GA: selected, selected learning rate 5e-06", text)
        self.assertIn("| 2e-06 | 300 | - | - | - | 1000 | 0.400 | False |", text)
        self.assertIn("| 5e-06 | 40 | 80 | 120 | 160 | 160 | 1.020 | False |", text)
        self.assertIn("gap 4.1400", text)

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
