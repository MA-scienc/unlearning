import unittest

from unlearning.data.safeguards import DatasetSeparationError, assert_no_forget_terms_in_texts


class SafeguardTests(unittest.TestCase):
    def test_catches_forget_term_in_specialization_text(self):
        records = [{"record_id": "PX-000001", "patient_name": "Arel Vossel"}]
        with self.assertRaises(DatasetSeparationError):
            assert_no_forget_terms_in_texts(records, ["Question mentions PX-000001 by mistake"])

    def test_allows_unrelated_medical_terms(self):
        records = [{"record_id": "PX-000001", "patient_name": "Arel Vossel"}]
        assert_no_forget_terms_in_texts(records, ["Atrial fibrillation and apixaban are concepts."])


if __name__ == "__main__":
    unittest.main()
