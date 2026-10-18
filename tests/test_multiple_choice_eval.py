import unittest

from unlearning.evaluation.multiple_choice import (
    bootstrap_accuracy_ci,
    multiple_choice_labels,
    score_from_option_logprobs,
    summarize_mc_results,
)
from unlearning.evaluation.clean_checkpoints import task_evaluation_inputs


class MultipleChoiceEvalTests(unittest.TestCase):
    def test_multiple_choice_labels_are_independent_of_option_wording(self):
        self.assertEqual(multiple_choice_labels(4), ["A", "B", "C", "D"])
        self.assertEqual(multiple_choice_labels(2), ["A", "B"])
        with self.assertRaises(ValueError):
            multiple_choice_labels(1)

    def test_medmcqa_scores_labels_not_full_option_wording(self):
        record = {
            "question": "Which option is correct?",
            "options": ["short", "a deliberately much longer answer option", "third", "fourth"],
            "answer_index": 1,
        }
        _prompt, score_options, answer_index, normalize = task_evaluation_inputs(record, "medmcqa")
        self.assertEqual(score_options, ["A", "B", "C", "D"])
        self.assertEqual(answer_index, 1)
        self.assertFalse(normalize)

    def test_option_logprob_scoring(self):
        result = score_from_option_logprobs(
            record_id="q1",
            answer_index=1,
            option_logprobs=[-3.0, -0.2, -1.0, -4.0],
        )
        self.assertTrue(result.correct)
        self.assertEqual(result.prediction_index, 1)
        self.assertAlmostEqual(result.nll, 0.2)

    def test_summary_and_ci_are_deterministic(self):
        results = [
            score_from_option_logprobs(record_id="a", answer_index=0, option_logprobs=[0.0, -1.0]),
            score_from_option_logprobs(record_id="b", answer_index=1, option_logprobs=[0.0, -1.0]),
        ]
        summary = summarize_mc_results(results)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(
            bootstrap_accuracy_ci([r.correct for r in results], seed=7, samples=20),
            bootstrap_accuracy_ci([r.correct for r in results], seed=7, samples=20),
        )


if __name__ == "__main__":
    unittest.main()
