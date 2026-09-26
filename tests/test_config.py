import unittest

from unlearning.config.schema import ConfigError, PilotConfig


def minimal_config():
    return {
        "experiment_id": "pilot",
        "seed": 42,
        "model": {
            "model_id": "Qwen/Qwen2.5-1.5B",
            "tokenizer_id": "Qwen/Qwen2.5-1.5B",
            "revision": "8faed761d45a263340a0528343f099c05c9a4323",
            "dtype": "bf16",
            "training_mode": "full_parameter",
            "gradient_checkpointing": True,
            "optimizer": "adamw_8bit",
        },
        "paths": {
            "data_dir": "data",
            "processed_dir": "data/processed",
            "generated_dir": "data/generated",
            "manifest_dir": "data/manifests",
            "result_dir": "results",
            "checkpoint_dir": "checkpoints",
        },
        "specialization": {
            "dataset_id": "openlifescienceai/medmcqa",
            "dataset_revision": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
            "source_url": "https://example.test/medmcqa",
            "source_license": "apache-2.0",
            "schema_version": "medmcqa-clm-v1",
            "train_split": "train",
            "dev_split": "validation",
            "test_split": "test",
            "context_length": 1024,
            "epochs": 3,
            "effective_batch_sequences": 32,
            "micro_batch_sequences": 1,
            "learning_rate": 0.00002,
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
            "max_grad_norm": 1.0,
            "drop_last_batches": False,
            "checkpoint_fractions": [0.0, 0.5, 1.0],
        },
        "forget": {
            "schema_version": "synthetic-patient-f-v1",
            "seed": 314159,
            "record_count": 120,
            "contamination_templates_per_record": 8,
            "eval_templates_per_record": 6,
            "mc_decoy_count": 3,
            "source_license": "CC0-1.0",
        },
        "contamination": {
            "condition": "fixed_exposure",
            "learning_rate": 0.00001,
            "epochs": 10,
            "effective_batch_size": 16,
            "sequence_length": 512,
            "warmup_ratio": 0.03,
            "scheduler": "linear",
            "optimizer": "adamw_8bit",
        },
        "unlearning": {
            "snapshot_steps": [0, 50, 100, 200, 400],
            "retain_loss_weight": 0.0,
            "methods": {
                "ga": {
                    "learning_rate": 0.000001,
                    "steps": 400,
                    "effective_batch_size": 16,
                    "sequence_length": 512,
                    "gradient_clip_norm": 1.0,
                },
                "npo": {
                    "learning_rate": 0.000001,
                    "steps": 400,
                    "effective_batch_size": 16,
                    "sequence_length": 512,
                    "gradient_clip_norm": 1.0,
                    "beta": 0.1,
                    "reference_model": "contaminated_checkpoint",
                },
            },
        },
        "evaluation": {
            "pubmedqa_dataset_id": "qiaojin/PubMedQA",
            "pubmedqa_revision": "9001f2853fb87cab8d220904e0de81ac6973b318",
            "pubmedqa_subset": "pqa_labeled",
            "pubmedqa_split": "train",
            "pubmedqa_source_url": "https://example.test/pubmedqa",
            "pubmedqa_license": "mit",
            "mmlu_cf_dataset_id": "microsoft/MMLU-CF",
            "mmlu_cf_revision": "c25b89a968a2062e422dd96ddf0fe3387507f0dd",
            "mmlu_cf_split": "validation",
            "mmlu_cf_source_url": "https://example.test/mmlu-cf",
            "mmlu_cf_license": "CDLA-2.0",
            "mmlu_cf_subset_size": 1000,
            "mmlu_cf_seed": 271828,
            "near_subjects": ["Medicine", "Pharmacology"],
            "near_topic_keywords": ["cardi", "diabetes"],
            "far_subjects": ["Skin", "Psychiatry"],
        },
    }


class ConfigTests(unittest.TestCase):
    def test_minimal_config_validates(self):
        config = PilotConfig.from_dict(minimal_config())
        self.assertEqual(config.unlearning.snapshot_steps, (0, 50, 100, 200, 400))

    def test_rejects_matched_memorization_for_pilot(self):
        raw = minimal_config()
        raw["contamination"]["condition"] = "matched_memorization"
        with self.assertRaises(ConfigError):
            PilotConfig.from_dict(raw)

    def test_rejects_posthoc_snapshot_policy(self):
        raw = minimal_config()
        raw["unlearning"]["snapshot_steps"] = [400]
        with self.assertRaises(ConfigError):
            PilotConfig.from_dict(raw)

    def test_rejects_duplicate_snapshot_steps(self):
        raw = minimal_config()
        raw["unlearning"]["snapshot_steps"] = [0, 50, 50, 400]
        with self.assertRaises(ConfigError):
            PilotConfig.from_dict(raw)

    def test_rejects_wrong_eval_template_count(self):
        raw = minimal_config()
        raw["forget"]["eval_templates_per_record"] = 5
        with self.assertRaises(ConfigError):
            PilotConfig.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
