import re
import tempfile
import unittest

from unlearning.config.pins import PINNED_CONTAMINATION, PINNED_UNLEARNING
from unlearning.evaluation.aggregate import main as aggregate_main
from unlearning.experiments.pipeline import PilotPipeline
from unlearning.experiments.stage2_pipeline import STAGE2_GATES, Stage2Pipeline, is_stage2, write_stage2_gate_report
from unlearning.runtime.settings import REPO_ROOT, MachineSettings

STAGE2 = REPO_ROOT / "configs" / "stage2" / "stage2_seed43.yaml"
PILOT = REPO_ROOT / "configs" / "pilot" / "qwen2.5-0.5b_laptop.yaml"
RUNTIME = REPO_ROOT / "configs" / "runtime" / "laptop_8gb.yaml"
LEVELS = ("C0", "C5", "C25", "C100", "CG25")  # four S levels and the format-matched control


class Stage2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = MachineSettings.from_env({"UNLEARNING_STORAGE_ROOT": self.temporary.name})
        self.pipeline = Stage2Pipeline(config_path=STAGE2, runtime_path=RUNTIME, settings=self.settings)

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_stage2_jobs(self):
        ids = [job.id for job in self.pipeline.jobs()]
        self.assertEqual(len(ids), len(set(ids)))
        forbidden = ("benchmark", "calibrate", "aggregate", "gate0", "gate1", "gate2", "gate3", "gate_")
        self.assertFalse([job for job in ids if job.startswith(forbidden)])
        self.assertFalse([job for job in ids if re.fullmatch(r"contaminate_C\d+", job)])  # no fixed-exposure arm
        self.assertEqual([job for job in ids if job.startswith("unlearn_")], [
            f"unlearn_MR_U3_{level}_{method}" for level in LEVELS for method in ("ga", "npo")
        ])
        gates = [job.id for job in self.pipeline.jobs() if job.kind == "gate"]
        self.assertEqual(gates, list(STAGE2_GATES))
        # order: data gate before training, procedural gate between contamination and unlearning
        self.assertLess(ids.index("stage2_data"), ids.index("specialize"))
        self.assertLess(ids.index("specialize"), ids.index("specialize_control"))  # G25 starts from S0's reference
        self.assertLess(ids.index("prepare_control_corpus"), ids.index("specialize_control"))
        self.assertIn("eval__G25__near", ids)
        self.assertEqual([job for job in ids if job.endswith("__manipulation")],
                         [f"eval__{name}__manipulation" for name in ("S0", "S5", "S25", "S100", "G25")])
        self.assertIn("eval__MR_U3_CG25_npo_f100__closeness", ids)
        self.assertLess(ids.index("contaminate_MR_C100"), ids.index("stage2_procedure"))
        self.assertLess(ids.index("stage2_procedure"), ids.index("unlearn_MR_U3_C0_ga"))

    def test_training_reads_the_pinned_rates(self):
        jobs = {job.id: job for job in self.pipeline.jobs()}
        for level in LEVELS:
            command = jobs[f"contaminate_MR_{level}"].command
            self.assertIn(PINNED_CONTAMINATION[1], command)
            self.assertIn(str(PINNED_CONTAMINATION[0]), command)
            for method in ("ga", "npo"):
                command = jobs[f"unlearn_MR_U3_{level}_{method}"].command
                self.assertIn(PINNED_UNLEARNING[1], command)
                self.assertTrue(any(part.endswith("revision3.yaml") for part in command))
                self.assertIn("--continue-to-all-types", command)

    def test_evaluation_plan(self):
        ids = [job.id for job in self.pipeline.jobs()]
        evals = [job for job in ids if job.startswith("eval__")]
        clean = [job for job in evals if re.match(r"eval__[SG]\d+__(?!real_facts|manipulation)", job)]
        contaminated = [job for job in evals if re.match(r"eval__MR_CG?\d+__(?!closeness|real_facts)", job)]
        full_u = [job for job in evals if re.match(r"eval__MR_U3_.*_(f050|f100|f_all|final)__(?!closeness|real_facts)", job)]
        retention_u = [job for job in evals if re.match(r"eval__MR_U3_.*_(f025|f075)__", job)]
        closeness = [job for job in evals if job.endswith("__closeness")]
        nonf = [job for job in evals if job.endswith("__closeness_nonf")]
        self.assertEqual(len(nonf), 5 + 10 * 4)
        self.assertEqual(len(clean), 5 * 6)  # every suite, MedMCQA full dev included
        self.assertEqual(len(contaminated), 5 * 6)  # C_i also get MedMCQA full dev
        self.assertEqual(len(full_u), 10 * 4 * 6)  # f050, f100, f_all, final
        self.assertEqual(len(retention_u), 10 * 2 * 4)  # near, far, general, PubMedQA only
        self.assertFalse([job for job in retention_u if job.endswith(("__forgetting", "__domain_full"))])
        self.assertEqual(len(closeness), 5 + 10 * 4)
        self.assertIn("eval__MR_C25__domain_full", ids)
        recovery = [job for job in ids if job.startswith("recover__")]
        self.assertEqual(len(recovery), (5 + 5 + 10) * 2)  # S_i, C_i, U at f100; relearn and benign
        self.assertGreater(ids.index(recovery[0]), max(ids.index(job) for job in evals))  # after every "before"
        real_facts = [job for job in evals if job.endswith("__real_facts")]
        self.assertEqual(len(real_facts), 5 + 5 + 10 * 4)  # S_i and G25, C_i, full U points

    def test_each_run_is_pruned_and_its_retention_points_evaluated_before_the_next_run(self):
        ids = [job.id for job in self.pipeline.jobs()]
        runs = [job for job in ids if job.startswith("unlearn_")]
        for current, following in zip(runs, runs[1:] + [None]):
            level, method = current.split("_")[-2:]
            chain = [current, f"prune_state__{level}_{method}",
                     f"eval__MR_U3_{level}_{method}_f025__near", f"prune_weights__MR_U3_{level}_{method}_f025",
                     f"prune_weights__MR_U3_{level}_{method}_f075"]
            positions = [ids.index(job) for job in chain]
            self.assertEqual(positions, sorted(positions), current)
            if following is not None:
                self.assertLess(positions[-1], ids.index(following))
        self.assertNotIn("eval__MR_U3_C0_ga_f025__forgetting", ids)  # retention only at f025/f075

    def test_gate_reports_are_procedural(self):
        for gate in STAGE2_GATES:
            text = write_stage2_gate_report(self.pipeline, gate).read_text(encoding="utf-8")
            self.assertIn("procedure only", text)
            for outcome in ("leakage", "accuracy", "KL", "retention loss", "gap closed"):
                self.assertNotIn(outcome, text)

    def test_routing_and_blinding(self):
        self.assertTrue(is_stage2(STAGE2))
        self.assertFalse(is_stage2(PILOT))
        with self.assertRaises(ValueError):
            Stage2Pipeline(config_path=PILOT, runtime_path=RUNTIME, settings=self.settings)
        self.assertEqual(PilotPipeline.gates.keys().isdisjoint(STAGE2_GATES), True)
        with self.assertRaises(SystemExit) as stopped:
            aggregate_main(["--config", str(STAGE2)])
        self.assertIn("blinded", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
