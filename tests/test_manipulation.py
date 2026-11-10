import unittest

from unlearning.evaluation.manipulation import template_segments, wikitext_paragraphs

RECORD = {"subject_name": "Medicine", "topic_name": None, "question": "Drug of choice?", "options": ["a", "b", "c", "d"],
          "answer_label": "B", "answer_text": "b", "explanation": None}


class ManipulationTests(unittest.TestCase):
    def test_segments_reproduce_both_templates_and_mask_constants(self):
        medical = template_segments(RECORD, control=False)
        content = "".join(text for text, scaffold in medical if not scaffold)
        self.assertEqual(content, "MedicineDrug of choice?abcdBb")  # fillers and labels are scaffold
        control = template_segments({**RECORD, "explanation": "2+2=4"}, control=True)
        self.assertTrue("".join(text for text, _ in control).startswith("multiple-choice question\n"))
        self.assertIn(("2+2=4", False), control)

    def test_wikitext_selection(self):
        rows = [" = Heading = ", "", "short", "x" * 250, " = = Sub = = ", "y" * 199]
        self.assertEqual(wikitext_paragraphs(rows), ["x" * 250])


if __name__ == "__main__":
    unittest.main()
