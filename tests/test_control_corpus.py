import unittest

from unlearning.data.control_corpus import build_control_records, convert_item, format_control_text
from unlearning.data.medmcqa import format_specialization_text

ITEM = {
    "question": "A train covers 60 km in 1.5 hours. What is its speed?",
    "options": ["A)30", "B)40", "C)45", "D)50", "E)60"],
    "rationale": "Speed = 60/1.5 = 40 km/h.\nThe answer is B.",
    "correct": "B",
}


class ControlCorpusTests(unittest.TestCase):
    def test_one_distractor_is_dropped_and_letters_follow(self):
        for index in range(50):
            record, _ = convert_item(index, ITEM)
            self.assertEqual(len(record["options"]), 4)
            self.assertNotEqual(record["dropped_option"], "B")
            self.assertEqual(record["answer_text"], "40")
            self.assertIn(f"The answer is {record['answer_label']}.", record["explanation"])
        self.assertEqual(convert_item(7, ITEM), convert_item(7, ITEM))  # deterministic per item

    def test_template_matches_the_specialization_template_without_medical(self):
        record, _ = convert_item(3, ITEM)
        control = format_control_text(record)
        medical = format_specialization_text({**record, "subject_name": "Mathematics", "topic_name": "Word problems"})
        self.assertEqual("Medical " + control, medical)

    def test_exclusions(self):
        medical = {**ITEM, "question": "Insulin dose is 10 units per hour; how much in 3 hours?"}
        records, counts = build_control_records(
            [ITEM, medical, {**ITEM, "question": "What is 2+2?"}, {**ITEM, "options": ["A)1"]}],
            keywords=["insulin"], evaluation_questions=["what is 2 + 2?"],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual((counts["medical_keyword"], counts["evaluation_overlap"], counts["malformed"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
