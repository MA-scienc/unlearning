import unittest

from unlearning.data.packing import compute_step_counts, tokenize_and_pack_texts


class ToyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 + 1 for char in text]


class PackingTests(unittest.TestCase):
    def test_packing_is_deterministic(self):
        texts = ["abc", "def"]
        left = tokenize_and_pack_texts(texts, ToyTokenizer(), context_length=4)
        right = tokenize_and_pack_texts(texts, ToyTokenizer(), context_length=4)
        self.assertEqual(left, right)
        self.assertEqual(len(left.sequences), 2)

    def test_step_counts(self):
        counts = compute_step_counts(
            packed_sequence_count=100,
            epochs=3,
            effective_batch_sequences=32,
            drop_last_batches=False,
        )
        self.assertEqual(counts["s100_steps"], 10)
        self.assertEqual(counts["s50_steps_floor"], 5)


if __name__ == "__main__":
    unittest.main()
