"""Matched-forgetting unlearning (revision 3): stopping signal, rule, monitor, closeness, aggregation."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_engine import tiny_model
from unlearning.config.schema import ConfigError
from unlearning.config.unlearning_revisions import UnlearningRevision, load_unlearning_revision
from unlearning.evaluation.aggregate import heldout_gap_rows, is_degenerate, parse_label, rows_for_file
from unlearning.evaluation.closeness import answer_start_divergence, summarize
from unlearning.runtime.settings import REPO_ROOT
from unlearning.training.matched_forgetting import ForgettingMonitor, gap_closed, newly_crossed, select_learning_rate
from unlearning.training.memorization import answer_token_logprob_means
from unlearning.utils.io import write_json

REVISIONS = REPO_ROOT / "configs" / "pilot" / "unlearning"
MATCHED = {
    "revision": 3,
    "tag": "U3",
    "mode": "matched_forgetting",
    "matched_forgetting": {
        "learning_rate_candidates": [2e-6, 5e-6, 1e-5, 2e-5],
        "thresholds": [0.25, 0.5, 0.75, 1.0],
        "check_every_steps": 2,
        "max_steps": 1000,
        "probe_templates": ["condition_lookup", "medication_lookup", "private_code_lookup"],
    },
}


class StoppingRuleTests(unittest.TestCase):
    def test_gap_closed_runs_from_contaminated_to_clean(self):
        self.assertEqual(gap_closed(contaminated=-0.1, clean=-4.1, current=-0.1), 0.0)
        self.assertEqual(gap_closed(contaminated=-0.1, clean=-4.1, current=-4.1), 1.0)
        self.assertAlmostEqual(gap_closed(contaminated=-0.1, clean=-4.1, current=-2.1), 0.5)
        self.assertGreater(gap_closed(contaminated=-0.1, clean=-4.1, current=-20.0), 1.0)
        with self.assertRaisesRegex(ValueError, "above S_i"):
            gap_closed(contaminated=-4.0, clean=-1.0, current=-2.0)

    def test_thresholds_are_reported_once(self):
        thresholds = (0.25, 0.5, 0.75, 1.0)
        self.assertEqual(newly_crossed(0.1, thresholds, []), [])
        self.assertEqual(newly_crossed(0.6, thresholds, []), ["f025", "f050"])
        self.assertEqual(newly_crossed(0.6, thresholds, ["f025"]), ["f050"])
        self.assertEqual(newly_crossed(1.3, thresholds, ["f025", "f050"]), ["f075", "f100"])

    def test_rule_selects_the_smallest_rate_that_closed_the_whole_gap(self):
        trials = [
            {"learning_rate": 2e-6, "final_threshold_reached": False},
            {"learning_rate": 1e-5, "final_threshold_reached": True},
            {"learning_rate": 5e-6, "final_threshold_reached": True},
        ]
        self.assertEqual(select_learning_rate(trials), 5e-6)
        self.assertIsNone(select_learning_rate(trials[:1]))

    def test_answer_logprob_kernel_matches_a_direct_computation(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 6, 11)
        labels = torch.full((3, 6), -100)
        labels[0, 3:5] = torch.tensor([2, 7])
        labels[1, 1:6] = torch.tensor([1, 2, 3, 4, 5])
        labels[2, 5] = 9
        means = answer_token_logprob_means(logits, labels)
        log_probs = logits.log_softmax(dim=-1)
        for row in range(3):
            positions = [p for p in range(1, 6) if labels[row, p] != -100]
            expected = sum(float(log_probs[row, p - 1, labels[row, p]]) for p in positions) / len(positions)
            self.assertAlmostEqual(float(means[row]), expected, places=5)


class _FakeProbe:
    def __init__(self, means):
        self.means = list(means)

    def answer_logprob(self, model, device):
        mean = self.means.pop(0)
        return {"mean": mean, "by_template": {"t": mean}, "per_example": {"x": mean}, "probe_count": 1}


class _Log:
    def event(self, *args, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class _Loop:
    finish_now = False


class ForgettingMonitorTests(unittest.TestCase):
    def monitor(self, means, saved, reached=None):
        revision = UnlearningRevision.from_dict(MATCHED)
        monitor = ForgettingMonitor(
            probe=_FakeProbe(means),
            references={"contaminated": {"mean": 0.0}, "clean": {"mean": -4.0}},
            spec=revision.matched,
            model_getter=lambda: None,
            device="cpu",
            run_log=_Log(),
            history_path=None,
            reached=reached,
            on_threshold=lambda names, entry: saved.append((entry["step"], tuple(names), entry["per_example"])) or True,
        )
        monitor.loop = _Loop()
        return monitor

    def test_checks_every_n_steps_saves_each_threshold_once_and_finishes_at_one(self):
        saved: list = []
        # g at checks (steps 2, 4, 6, 8): 0.1, 0.55, 0.8, 1.05
        monitor = self.monitor([-0.4, -2.2, -3.2, -4.2], saved)
        results = [monitor(step, {"loss": 0.0}) for step in range(1, 9)]
        self.assertEqual(results, [False, False, False, True, False, True, False, True])
        self.assertEqual([(step, names) for step, names, _ in saved], [(4, ("f025", "f050")), (6, ("f075",)), (8, ("f100",))])
        self.assertTrue(monitor.loop.finish_now)
        self.assertEqual(monitor.reached["f050"]["step"], 4)
        self.assertEqual(saved[0][2], {"x": -2.2})  # per-example values travel with the snapshot

    def test_resumed_thresholds_are_not_saved_again(self):
        saved: list = []
        monitor = self.monitor([-3.2], saved, reached={"f025": {"step": 4}, "f050": {"step": 4}})
        monitor(2, {})
        self.assertEqual([names for _, names, _ in saved], [("f075",)])
        self.assertFalse(monitor.loop.finish_now)


class MatchedRevisionTests(unittest.TestCase):
    def test_repository_revision_3_and_unchanged_revision_2_fingerprint(self):
        revision = load_unlearning_revision(REVISIONS / "revision3.yaml")
        self.assertTrue(revision.is_matched)
        self.assertEqual(revision.matched.snapshot_names, ("f025", "f050", "f075", "f100"))
        self.assertEqual(revision.matched.learning_rate_candidates, (2e-6, 5e-6, 1e-5, 2e-5))
        # Recorded in every revision-2 snapshot before modes existed; must never change.
        self.assertEqual(
            load_unlearning_revision(REVISIONS / "revision2.yaml").fingerprint,
            "239942fd56e52c3a39004ced2616cf142a8529301846b13467a6e8bfdf04128d",
        )

    def test_matched_method_config_needs_the_calibrated_rate(self):
        from tests.test_config import minimal_config
        from unlearning.config.schema import PilotConfig

        config = PilotConfig.from_dict(minimal_config())
        revision = UnlearningRevision.from_dict(MATCHED)
        with self.assertRaisesRegex(ValueError, "calibrated"):
            revision.method_config(config, "ga")
        method = revision.method_config(config, "npo", learning_rate=5e-6)
        self.assertEqual((method.learning_rate, method.steps), (5e-6, 1000))
        self.assertEqual(method.beta, config.unlearning.methods["npo"].beta)

    def test_rejects_invalid_matched_settings(self):
        def variant(**changes):
            return {**MATCHED, "matched_forgetting": {**MATCHED["matched_forgetting"], **changes}}

        cases = [
            (variant(thresholds=[0.25, 0.5]), "last threshold must be 1.0"),
            (variant(thresholds=[0.5, 0.25, 1.0]), "ascending"),
            (variant(learning_rate_candidates=[1e-5, 2e-6]), "ascending"),
            (variant(check_every_steps=3), "multiple"),
            (variant(probe_templates=["eval_direct_code"]), "unknown probe templates"),
            ({**MATCHED, "learning_rate": {"ga": 1e-5, "npo": 1e-5}}, "may set only"),
            ({**variant(), "matched_forgetting": {**MATCHED["matched_forgetting"], "steps": 5}}, "unknown keys"),
        ]
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ConfigError, message):
                UnlearningRevision.from_dict(raw)


class ClosenessTests(unittest.TestCase):
    def test_divergence_is_zero_for_the_clean_model_itself_and_positive_otherwise(self):
        clean, other = tiny_model(seed=0), tiny_model(seed=1)
        prompt, continuation = [3, 4, 5, 6], [7, 8, 9]
        same = answer_start_divergence(clean, clean, prompt, continuation, torch=torch, device="cpu")
        self.assertEqual((same["kl_mean"], same["kl_first"], same["positions"]), (0.0, 0.0, 3))
        different = answer_start_divergence(clean, other, prompt, continuation, torch=torch, device="cpu")
        self.assertGreater(different["kl_mean"], 0.0)
        # An empty continuation (immediate end of sequence) still scores the first answer position.
        self.assertEqual(answer_start_divergence(clean, other, prompt, [], torch=torch, device="cpu")["positions"], 1)

    def test_summary_aggregates_templates_within_a_record_first(self):
        rows = [
            {"record_id": "A", "kl_mean": 1.0, "kl_first": 2.0},
            {"record_id": "A", "kl_mean": 3.0, "kl_first": 2.0},
            {"record_id": "B", "kl_mean": 10.0, "kl_first": 0.0},
        ]
        summary = summarize(rows, seed=1)
        self.assertEqual(summary["kl_to_clean_answer_start"], 6.0)  # (2 + 10) / 2, not 14 / 3
        self.assertAlmostEqual(summary["prompt_mean_kl_to_clean_answer_start"], 14 / 3)
        low, high = summary["kl_to_clean_answer_start_ci95"]
        self.assertLessEqual(low, 6.0)
        self.assertGreaterEqual(high, 6.0)


class AggregationTests(unittest.TestCase):
    def test_threshold_labels(self):
        label = parse_label("MR_U3_C50_npo_f050")
        self.assertEqual(
            (label["unlearning_revision"], label["specialization_level"], label["forgetting_threshold"], label["unlearning_step"]),
            (3, 50, 0.5, None),
        )
        self.assertEqual(parse_label("MR_U3_C0_ga_final")["forgetting_threshold"], "final")
        self.assertEqual(parse_label("MR_U2_C0_ga_step0050")["unlearning_step"], 50)

    def test_degenerate_outputs(self):
        for text in ("stable stable stable stable stable stable stable", "chron" * 30, "\\D" * 40, "',...\n" * 20):
            self.assertTrue(is_degenerate(text), text[:20])
        for text in ("lisinopril", " venous thromboembolism.", "The verification code for the synthetic patient PX-1 is PC-2."):
            self.assertFalse(is_degenerate(text), text)

    def test_heldout_gap_closed_uses_the_level_s_own_clean_and_contaminated_models(self):
        def row(model, value):
            return {"model": model, "suite": "forgetting", "metric": "target_logprob", "value": value, **parse_label(model)}

        rows = [row("S50", -4.0), row("MR_C50", -1.0), row("MR_U3_C50_npo_f100", -4.0), row("MR_U3_C50_ga_f050", -2.5)]
        derived = {r["model"]: r["value"] for r in heldout_gap_rows(rows)}
        self.assertEqual(derived, {"MR_U3_C50_npo_f100": 1.0, "MR_U3_C50_ga_f050": 0.5})

    def test_snapshot_progress_and_closeness_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "snapshot_manifest.json"
            manifest.write_text(json.dumps({"snapshot_step": 42, "forgetting_progress": {"gap_closed": 0.53}}), encoding="utf-8")
            forgetting = root / "MR_U3_C0_ga_f050" / "forgetting.json"
            forgetting.parent.mkdir()
            forgetting.write_text(
                json.dumps(
                    {
                        "records": [{"generated_text": "stable " * 20}, {"generated_text": "lisinopril"}],
                        "summary": {},
                        "checkpoint_provenance": {"manifest_path": str(manifest)},
                    }
                ),
                encoding="utf-8",
            )
            closeness = root / "MR_U3_C0_ga_f050" / "closeness.json"
            closeness.write_text(
                json.dumps({"summary": {"kl_to_clean_answer_start": 0.7, "kl_to_clean_answer_start_ci95": [0.5, 0.9], "kl_to_clean_first_token": 1.2, "record_count": 120}}),
                encoding="utf-8",
            )
            from unittest.mock import patch

            with patch("unlearning.evaluation.aggregate.rescore_forgetting_payload", return_value={"summary": {}}):
                rows = {r["metric"]: r for r in rows_for_file("run", forgetting, seed=1)}
            close = {r["metric"]: r for r in rows_for_file("run", closeness, seed=1)}
        self.assertEqual(rows["degenerate_output_rate"]["value"], 0.5)
        self.assertEqual((rows["training_probe_gap_closed"]["value"], rows["snapshot_step"]["value"]), (0.53, 42))
        self.assertEqual((close["kl_to_clean_answer_start"]["ci_low"], close["kl_to_clean_answer_start"]["value"]), (0.5, 0.7))
        self.assertTrue(math.isclose(close["kl_to_clean_first_token"]["value"], 1.2))


class SummaryTests(unittest.TestCase):
    def test_summary_tables_from_processed_and_raw_results(self):
        import csv

        from unlearning.reporting.matched_forgetting import build_summary

        with tempfile.TemporaryDirectory() as directory:
            raw, processed = Path(directory) / "raw", Path(directory) / "processed"
            processed.mkdir()
            rows = [
                {"model": "MR_U3_C0_ga_f100", "suite": "forgetting", "metric": "snapshot_step", "value": 888},
                {"model": "MR_U3_C0_ga_f100", "suite": "forgetting", "metric": "training_probe_gap_closed", "value": 1.009},
                {"model": "MR_U3_C0_ga_f100", "suite": "forgetting", "metric": "target_exact_leakage_rate", "value": 0.153},
                {"model": "MR_U3_C0_ga_f100", "suite": "near", "metric": "bias_removed_accuracy", "value": 0.5},
                {"model": "S0", "suite": "near", "metric": "bias_removed_accuracy", "value": 0.75},
            ]
            with (processed / "metrics_long.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model", "suite", "metric", "value"])
                writer.writeheader()
                writer.writerows(rows)
            # Same 4 questions: S0 right on all, U right on 2 (no position preference, so bias removal is neutral).
            def records(correct):
                return [{"option_logprobs": [-1.0 if j == (a if ok else (a + 1) % 4) else -3.0 for j in range(4)], "answer_index": a}
                        for a, ok in zip(range(4), correct)]
            for model, correct in (("S0", [True] * 4), ("MR_U3_C0_ga_f100", [True, False, True, False])):
                (raw / model).mkdir(parents=True)
                (raw / model / "near.json").write_text(json.dumps({"records": records(correct)}), encoding="utf-8")
            text = build_summary(raw, processed, "run", tag="U3", names=("f025", "f050", "f075", "f100"), seed=1)
        self.assertIn("| GA C0 | not reached | not reached | not reached | 888 (1.009) |", text)
        self.assertIn("| C0 | GA f100 | n/a | 0.153 |", text)
        self.assertIn("| GA | near | -0.500 [", text)  # 2 of 4 questions lost vs S0
        self.assertIn("n/a", text)  # levels without results are shown, not dropped


class _TwoTypeProbe:
    """Per check: (code, diagnosis) mean log-probs; the overall mean is their average."""

    def __init__(self, checks):
        self.checks = list(checks)

    def answer_logprob(self, model, device):
        code, diagnosis = self.checks.pop(0)
        return {"mean": (code + diagnosis) / 2, "by_template": {"code": code, "diagnosis": diagnosis},
                "per_example": {"x": code}, "probe_count": 2}


class AllTypesStoppingTests(unittest.TestCase):
    """Stage 2's f_all: continue after f100 until every fact type closed its own gap."""

    REFERENCES = {
        "contaminated": {"mean": 0.0, "by_template": {"code": 0.0, "diagnosis": 0.0}},
        "clean": {"mean": -4.0, "by_template": {"code": -4.0, "diagnosis": -4.0}},
    }

    def monitor(self, checks, saved, all_types):
        monitor = ForgettingMonitor(
            probe=_TwoTypeProbe(checks), references=self.REFERENCES,
            spec=UnlearningRevision.from_dict(MATCHED).matched, model_getter=lambda: None, device="cpu",
            run_log=_Log(), history_path=None, all_types=all_types,
            on_threshold=lambda names, entry: saved.append((entry["step"], tuple(names))) or True,
        )
        monitor.loop = _Loop()
        return monitor

    # checks at steps 2, 4, 6: code races ahead (overall g = 1 at step 4) while the diagnosis lags
    CHECKS = [(-4.0, -0.4), (-7.0, -1.0), (-9.0, -4.2)]

    def test_without_all_types_the_run_stops_at_f100(self):
        saved: list = []
        monitor = self.monitor(self.CHECKS, saved, all_types=False)
        for step in (1, 2, 3, 4):
            monitor(step, {})
        self.assertTrue(monitor.done and monitor.loop.finish_now)
        self.assertNotIn("f_all", [name for _, names in saved for name in names])

    def test_all_types_continues_until_the_lagging_type_closes_its_gap(self):
        saved: list = []
        monitor = self.monitor(self.CHECKS, saved, all_types=True)
        for step in (1, 2, 3, 4):
            monitor(step, {})
        self.assertTrue(monitor.final_reached)  # g = (0 - (-4)) / 4 = 1.0 at step 4
        self.assertFalse(monitor.done or monitor.loop.finish_now)  # diagnosis gap only 0.25
        self.assertAlmostEqual(monitor.last_entry["gap_closed_by_type"]["diagnosis"], 0.25)
        monitor(5, {})
        monitor(6, {})
        self.assertEqual(saved[-1], (6, ("f_all",)))
        self.assertTrue(monitor.done and monitor.loop.finish_now)
        self.assertGreaterEqual(min(monitor.reached["f_all"]["gap_closed_by_type"].values()), 1.0)


class ReportingStatisticsTests(unittest.TestCase):
    def test_holm_adjusts_step_down_and_keeps_input_order(self):
        from unlearning.reporting.matched_forgetting import holm

        adjusted = holm([0.01, 0.04, 0.03])
        for value, expected in zip(adjusted, [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(holm([0.9, 0.8]), [1.0, 1.0])  # 2 x 0.8 is capped at 1; later steps never go lower

    def test_paired_bootstrap_p_value(self):
        from unlearning.reporting.matched_forgetting import paired_bootstrap

        mean, low, high, p = paired_bootstrap([1.0, 0.0, 1.0, 1.0] * 30, seed=3, samples=2000)
        self.assertAlmostEqual(mean, 0.75)
        self.assertGreater(low, 0)
        self.assertEqual(p, 0.0)  # no resample crosses zero
        _, low, high, p = paired_bootstrap([1.0, -1.0] * 50, seed=3, samples=2000)
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        self.assertGreater(p, 0.5)


class ForgettingCompositionTests(unittest.TestCase):
    def _trajectory(self, root: Path, *, sidecar_fingerprint: str = "refs") -> Path:
        from unlearning.training.matched_forgetting import PER_EXAMPLE_REFERENCES_FILE, REFERENCES_FILE

        trajectory = root / "trajectory"
        (trajectory / "f100").mkdir(parents=True)
        # Two patients, two fact types. C ~ 0, S = -4. U forgets codes past S, keeps diagnoses.
        clean = {"PX-1::condition_lookup": -4.0, "PX-1::private_code_lookup": -4.0,
                 "PX-2::condition_lookup": -4.0, "PX-2::private_code_lookup": -4.0}
        contaminated = {key: 0.0 for key in clean}
        unlearned = {"PX-1::condition_lookup": -0.4, "PX-1::private_code_lookup": -8.0,
                     "PX-2::condition_lookup": -0.0, "PX-2::private_code_lookup": -12.0}
        def means(values):
            out = {}
            for template in ("condition_lookup", "private_code_lookup"):
                selected = [v for k, v in values.items() if k.endswith(template)]
                out[template] = sum(selected) / len(selected)
            return out
        write_json(trajectory / REFERENCES_FILE, {
            "fingerprint": "refs",
            "clean": {"mean": -4.0, "by_template": means(clean)},
            "contaminated": {"mean": 0.0, "by_template": means(contaminated)},
        })
        write_json(trajectory / PER_EXAMPLE_REFERENCES_FILE, {
            "references_fingerprint": sidecar_fingerprint,
            "clean": {"per_example": clean},
            "contaminated": {"per_example": contaminated},
        })
        write_json(trajectory / "f100" / "manifest.json", {
            "forgetting_progress": {"by_template": means(unlearned), "per_example": unlearned, "gap_closed": 1.0},
        })
        return trajectory

    def test_gap_per_fact_type_and_per_patient(self):
        from unlearning.reporting.matched_forgetting import forgetting_composition

        with tempfile.TemporaryDirectory() as directory:
            trajectory = self._trajectory(Path(directory))
            composition = forgetting_composition(trajectory / "f100" / "manifest.json")
        self.assertAlmostEqual(composition["by_type"]["condition_lookup"], 0.05)  # 0.2 of a 4-nat gap
        self.assertAlmostEqual(composition["by_type"]["private_code_lookup"], 2.5)  # overshoots S_i
        self.assertEqual(composition["patients_at_clean"], {"condition_lookup": 0.0, "private_code_lookup": 1.0})
        self.assertEqual(composition["patients"], 2)
        # Patient gaps on the mean of their probes: PX-1 (0.4 + 8) / 2 / 4 = 1.05, PX-2 12 / 2 / 4 = 1.5.
        self.assertAlmostEqual(composition["patient_gap_quantiles"][1], 1.275)

    def test_backfilled_references_must_belong_to_the_run(self):
        from unlearning.training.matched_forgetting import per_example_references

        with tempfile.TemporaryDirectory() as directory:
            trajectory = self._trajectory(Path(directory), sidecar_fingerprint="other")
            with self.assertRaisesRegex(ValueError, "different references"):
                per_example_references(trajectory)


if __name__ == "__main__":
    unittest.main()
