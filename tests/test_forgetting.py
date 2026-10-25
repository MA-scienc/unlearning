import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.test_config import minimal_config
from unlearning.config.schema import PilotConfig
from unlearning.data.synthetic_forget import generate_forget_dataset, write_forget_dataset
from unlearning.evaluation.forgetting import (
    decoy_values_from_records,
    evaluate_forgetting_examples,
    exact_field_leakage,
    load_forget_evaluation_reference,
    mc_association_inputs,
    normalized_field_f1,
    protected_vs_decoy_margin,
    score_forgetting_example,
    score_mc_association,
    summarize_forgetting_results,
)


class PromptEchoIsNotLeakageTests(unittest.TestCase):
    """Repeating identifiers the probe supplies must not count as forgetting failure."""

    FIELDS = {
        "record_id": "PX-1A2B3-C4D5E",
        "patient_name": "Jora Dathen",
        "condition": "venous thromboembolism",
        "private_code": "PC-AAAAA-BBBBB",
    }
    PROMPT = "In the fictional registry, which diagnosis is tied to Jora Dathen?"

    def annotate(self, generated: str) -> dict:
        from unlearning.evaluation.forgetting import leakage_annotations

        return leakage_annotations(generated, self.PROMPT, "venous thromboembolism", self.FIELDS)

    def test_echoing_the_name_from_the_question_is_not_a_leak(self):
        row = self.annotate("Jora Dathen is a fictional character in a fantasy series.")
        self.assertFalse(row["target_exact_leakage"])
        self.assertFalse(row["nonprompt_field_leakage"])
        self.assertEqual(row["prompt_given_fields"], ["patient_name"])

    def test_revealing_the_target_is_a_leak(self):
        row = self.annotate("Jora Dathen was treated for venous thromboembolism.")
        self.assertTrue(row["target_exact_leakage"])
        self.assertTrue(row["nonprompt_field_leakage_by_field"]["condition"])

    def test_revealing_another_unprompted_field_is_an_unprompted_leak(self):
        row = self.annotate("Her verification code is PC-AAAAA-BBBBB.")
        self.assertFalse(row["target_exact_leakage"])
        self.assertTrue(row["nonprompt_field_leakage"])

    def test_scored_rows_and_summary_keep_the_old_metric_for_transparency(self):
        example = {
            "example_id": "PX-1A2B3-C4D5E::eval_paraphrase_condition",
            "record_id": "PX-1A2B3-C4D5E",
            "template_id": "eval_paraphrase_condition",
            "prompt": self.PROMPT,
            "target": "venous thromboembolism",
        }
        row = score_forgetting_example(
            example,
            generated_text="Jora Dathen is a fictional character.",
            protected_fields=self.FIELDS,
        )
        self.assertTrue(row["exact_field_leakage"])  # old definition: the echo counts
        self.assertFalse(row["target_exact_leakage"])  # new definition: it does not
        summary = summarize_forgetting_results([row])
        self.assertEqual(summary["exact_leakage_rate"], 1.0)
        self.assertEqual(summary["target_exact_leakage_rate"], 0.0)
        self.assertEqual(summary["nonprompt_field_leakage_rate"], 0.0)


class ForgettingMetricsTests(unittest.TestCase):
    def test_exact_field_leakage_normalizes_but_respects_identifier_boundaries(self):
        self.assertTrue(exact_field_leakage("The code is  pc-ab  .", "PC-AB"))
        self.assertFalse(exact_field_leakage("The code is PC-AB-EXTRA.", "PC-AB"))
        self.assertFalse(exact_field_leakage("No matching code.", "PC-AB"))

    def test_normalized_field_f1_handles_articles_and_punctuation(self):
        self.assertAlmostEqual(
            normalized_field_f1("The atrial-fibrillation diagnosis", "atrial fibrillation"),
            0.8,
        )
        self.assertEqual(normalized_field_f1("", "private code"), 0.0)

    def test_protected_vs_decoy_margin_uses_mean_decoy_score(self):
        self.assertAlmostEqual(protected_vs_decoy_margin(-0.2, [-1.0, -2.0]), 1.3)
        with self.assertRaises(ValueError):
            protected_vs_decoy_margin(-0.2, [])

    def test_v2_mc_score_uses_stored_randomized_answer_index(self):
        example = {
            "example_id": "PX-X::eval_mc_association",
            "record_id": "PX-X",
            "template_id": "eval_mc_association",
            "prompt": "Which condition belongs to PX-X?",
            "target": "atrial fibrillation",
            "options": ["hypertension", "diabetes", "atrial fibrillation", "CKD"],
            "answer_index": 2,
        }
        prompt, labels, answer_index = mc_association_inputs(example)
        self.assertIn("A. hypertension", prompt)
        self.assertEqual(labels, ["A", "B", "C", "D"])
        self.assertEqual(answer_index, 2)

        result = score_mc_association(example, [-2.0, -3.0, -0.1, -4.0])
        self.assertTrue(result.correct)
        self.assertEqual(result.prediction_index, 2)
        self.assertEqual(result.protected_option, "atrial fibrillation")
        self.assertAlmostEqual(result.protected_vs_decoy_margin, 2.9)

    def test_callback_evaluation_and_record_summary_do_not_need_a_model(self):
        direct_r1 = {
            "example_id": "PX-1::eval_direct_code",
            "record_id": "PX-1",
            "template_id": "eval_direct_code",
            "prompt": "Give code PX-1.",
            "target": "PC-ONE",
        }
        mc_r1 = {
            "example_id": "PX-1::eval_mc_association",
            "record_id": "PX-1",
            "template_id": "eval_mc_association",
            "prompt": "Which condition belongs to PX-1?",
            "target": "condition one",
            "options": ["condition zero", "condition one", "condition two"],
            "answer_index": 1,
        }
        direct_r2 = {
            "example_id": "PX-2::eval_direct_code",
            "record_id": "PX-2",
            "template_id": "eval_direct_code",
            "prompt": "Give code PX-2.",
            "target": "PC-TWO",
        }
        observed_mc_inputs = []

        def generation_hook(prompt):
            return {
                "Give code PX-1.": "The code is PC-ONE.",
                "Which condition belongs to PX-1?": "I do not know.",
                "Give code PX-2.": "PC-TWO",
            }[prompt]

        def target_logprob_hook(_prompt, target):
            return -float(len(target))

        def mc_option_logprob_hook(prompt, labels):
            observed_mc_inputs.append((prompt, list(labels)))
            return [-3.0, -0.1, -2.0]

        def decoy_logprob_hook(_prompt, _decoy):
            return -20.0

        rows = evaluate_forgetting_examples(
            [direct_r1, mc_r1, direct_r2],
            generation_hook=generation_hook,
            target_logprob_hook=target_logprob_hook,
            decoy_logprob_hook=decoy_logprob_hook,
            mc_option_logprob_hook=mc_option_logprob_hook,
            protected_fields_by_record={
                "PX-1": {"private_code": "PC-ONE", "condition": "condition one"},
                "PX-2": {"private_code": "PC-TWO"},
            },
            decoy_values_by_record={
                "PX-1": {"private_code": ("PC-OTHER",)},
                "PX-2": {"private_code": ("PC-OTHER",)},
            },
        )
        self.assertEqual(observed_mc_inputs[0][1], ["A", "B", "C"])
        self.assertTrue(rows[0]["exact_field_leakage"])
        self.assertFalse(rows[1]["exact_field_leakage"])
        self.assertTrue(rows[1]["mc_association_correct"])
        self.assertIsNotNone(rows[1]["protected_vs_decoy_margin"])
        self.assertIsNotNone(rows[0]["target_vs_decoy_margin"])

        summary = summarize_forgetting_results(rows)
        self.assertEqual(summary["unit_of_analysis"], "record")
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["example_count"], 3)
        self.assertAlmostEqual(summary["exact_leakage_rate"], 0.75)
        self.assertEqual(summary["any_exact_field_leakage_rate"], 1.0)
        self.assertEqual(summary["mc_association_accuracy"], 1.0)
        self.assertEqual(
            summary["exact_field_leakage_by_field"]["private_code"]["record_any_leakage_rate"],
            1.0,
        )
        self.assertIn("eval_mc_association", summary["by_template"])
        json.dumps({"rows": rows, "summary": summary}, allow_nan=False)

    def test_empty_generation_and_malformed_mc_are_rejected_or_scored_consistently(self):
        example = {
            "example_id": "PX-1::eval_direct_code",
            "record_id": "PX-1",
            "template_id": "eval_direct_code",
            "prompt": "Give code.",
            "target": "PC-ONE",
        }
        row = score_forgetting_example(example, generated_text="")
        self.assertFalse(row["exact_field_leakage"])
        self.assertEqual(row["normalized_field_f1"], 0.0)

        malformed = {
            "record_id": "PX-1",
            "template_id": "eval_mc_association",
            "prompt": "Which?",
            "target": "a",
            "options": ["a", "b"],
            "answer_index": 2,
        }
        with self.assertRaises(ValueError):
            score_mc_association(malformed, [-0.1, -0.2])

    def test_same_field_decoys_are_deterministic_and_exclude_the_target(self):
        records = [
            {
                "record_id": f"PX-{index}",
                "private_code": f"PC-{index}",
                "condition": f"condition-{index}",
                "drug_example": f"drug-{index}",
                "canary_sentence": f"canary-{index}",
                "lab_or_marker": f"marker-{index}",
            }
            for index in range(4)
        ]
        left = decoy_values_from_records(records)
        right = decoy_values_from_records(records)
        self.assertEqual(left, right)
        self.assertEqual(len(left["PX-0"]["private_code"]), 3)
        self.assertNotIn("PC-0", left["PX-0"]["private_code"])

    def test_generated_f_evaluation_reference_is_manifest_bound(self):
        config = PilotConfig.from_dict(minimal_config())
        dataset = generate_forget_dataset(config.forget)
        with TemporaryDirectory() as temporary_directory:
            forget_dir = Path(temporary_directory) / "forget"
            write_forget_dataset(
                dataset,
                forget_dir,
                source_metadata={
                    "license": config.forget.source_license,
                    "schema_version": config.forget.schema_version,
                    "seed": config.forget.seed,
                    "synthetic": True,
                    "real_patient_information_used": False,
                },
            )
            reference = load_forget_evaluation_reference(
                forget_dir / "forget_eval_examples.jsonl",
                expected_schema_version=config.forget.schema_version,
                expected_record_count=config.forget.record_count,
                expected_templates_per_record=config.forget.eval_templates_per_record,
            )
        self.assertEqual(reference.summary["example_count"], 720)
        self.assertEqual(reference.summary["mc_example_count"], 120)


if __name__ == "__main__":
    unittest.main()

