"""Engine exactness and resume tests on a tiny random Qwen2 model (CPU only).

These verify the two properties the experiment relies on:

1. Splitting an optimizer step into micro-batches (or across processes) yields
   the same gradient as the original full-batch objective, for every stage.
2. Interrupting, saving a resume point and resuming reproduces the parameters
   of an uninterrupted run.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from unlearning.runtime.resume import ResumeStore
from unlearning.runtime.run_log import RunLog
from unlearning.runtime.settings import MachineSettings
from unlearning.runtime.signals import GracefulStop
from unlearning.training.contaminate import (
    IGNORE_INDEX,
    ContaminationObjective,
    FixedExposurePlan,
    collate_completion_batch,
)
from unlearning.training.engine import TrainingLoop, apply_resume_state, create_accelerator
from unlearning.training.specialize_clean import SpecializationObjective
from unlearning.training.unlearn import (
    CompletionOnlySequence,
    UnlearningObjective,
    collate_completion_only_batch,
    completion_logprob_sums,
    ga_completion_only_loss,
    npo_completion_only_loss,
)

VOCAB = 61


def tiny_model(seed: int = 0) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def engine_gradient(model, objective, step: int, world: int = 1) -> list[torch.Tensor]:
    """Replicates the engine: rank r runs micro-batches r, r+W, ... scaled by W / N,
    then DDP averages the per-rank gradients."""

    micro_batches = objective.micro_batches(step)
    normalizer = objective.normalizer(step, micro_batches)
    per_rank = []
    for rank in range(world):
        model.zero_grad(set_to_none=True)
        for micro_batch in micro_batches[rank::world]:
            loss_sum, _stats = objective.loss_sum(model, dict(micro_batch.tensors))
            (loss_sum * (world / normalizer)).backward()
        per_rank.append([parameter.grad.detach().clone() for parameter in model.parameters()])
    return [sum(grads) / world for grads in zip(*per_rank)]


def reference_gradient(model, loss_fn) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss_fn().backward()
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def assert_same_gradients(case: unittest.TestCase, left, right) -> None:
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def completion_example(index: int, prompt_len: int, completion_len: int) -> dict:
    generator = torch.Generator().manual_seed(1000 + index)
    length = prompt_len + completion_len
    input_ids = torch.randint(1, VOCAB, (length,), generator=generator).tolist()
    labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
    return {"example_id": f"E{index:03d}", "input_ids": input_ids, "labels": labels}


class MicroBatchExactnessTests(unittest.TestCase):
    def test_specialization_micro_batching_matches_full_batch_token_mean(self):
        model = tiny_model()
        packed = torch.randint(1, VOCAB, (24, 12), generator=torch.Generator().manual_seed(3))
        full = SpecializationObjective(packed, effective_batch_sequences=8, micro_batch_sequences=8)
        batch = full.micro_batches(2)[0].tensors["input_ids"]
        expected = reference_gradient(model, lambda: model(input_ids=batch, labels=batch).loss)
        for micro in (1, 2, 4):
            objective = SpecializationObjective(packed, effective_batch_sequences=8, micro_batch_sequences=micro)
            assert_same_gradients(self, engine_gradient(model, objective, step=2), expected)
        two_processes = SpecializationObjective(packed, effective_batch_sequences=8, micro_batch_sequences=2)
        assert_same_gradients(self, engine_gradient(model, two_processes, step=2, world=2), expected)

    def test_contamination_micro_batching_matches_full_batch_token_mean(self):
        model = tiny_model()
        # Unequal completion lengths make naive mean-of-means differ from the true mean.
        examples = [completion_example(index, 3 + index % 3, 2 + (index * 5) % 7) for index in range(8)]
        plan = FixedExposurePlan(
            example_count=8,
            epochs=2,
            effective_batch_size=4,
            steps_per_epoch=2,
            optimizer_steps=4,
            total_example_exposures=16,
        )
        step = 4  # second epoch, second batch -> examples 4..7
        full = collate_completion_batch(examples[4:8], pad_token_id=0)
        tensors = {key: torch.tensor(full[key]) for key in ("input_ids", "attention_mask", "labels")}
        expected = reference_gradient(model, lambda: model(**tensors).loss)
        for micro in (1, 2, 4):
            objective = ContaminationObjective(
                torch, examples, plan=plan, micro_batch_size=micro, pad_token_id=0
            )
            assert_same_gradients(self, engine_gradient(model, objective, step=step), expected)

    def _sequences(self) -> list[CompletionOnlySequence]:
        sequences = []
        for index in range(6):
            example = completion_example(index, 2 + index % 2, 1 + (index * 3) % 5)
            sequences.append(
                CompletionOnlySequence(
                    example_id=example["example_id"],
                    record_id=f"R{index}",
                    template_id="t",
                    input_ids=tuple(example["input_ids"]),
                    labels=tuple(example["labels"]),
                    prompt_token_count=0,
                    completion_token_count=0,
                )
            )
        return sequences

    def test_ga_micro_batching_matches_full_batch_objective(self):
        model = tiny_model()
        sequences = self._sequences()
        step = 2  # batch of 4 wraps around the 6 canonical examples
        selected = [sequences[index] for index in (4, 5, 0, 1)]
        full = collate_completion_only_batch(selected, pad_token_id=0)

        def full_loss():
            logits = model(input_ids=full["input_ids"], attention_mask=full["attention_mask"]).logits
            return ga_completion_only_loss(logits, full["labels"])

        expected = reference_gradient(model, full_loss)
        for micro in (1, 2, 4):
            objective = UnlearningObjective(
                torch, sequences, method="ga", batch_size=4, micro_batch_size=micro, pad_token_id=0
            )
            assert_same_gradients(self, engine_gradient(model, objective, step=step), expected)

    def test_npo_micro_batching_matches_full_batch_objective(self):
        model = tiny_model()
        sequences = self._sequences()
        reference_model = tiny_model(seed=7)
        everything = collate_completion_only_batch(sequences, pad_token_id=0)
        with torch.no_grad():
            reference_sums, _ = completion_logprob_sums(
                reference_model(
                    input_ids=everything["input_ids"], attention_mask=everything["attention_mask"]
                ).logits,
                everything["labels"],
            )
        reference = {
            sequence.example_id: float(value) for sequence, value in zip(sequences, reference_sums)
        }
        selected = [sequences[index] for index in (4, 5, 0, 1)]
        full = collate_completion_only_batch(selected, pad_token_id=0)

        def full_loss():
            logits = model(input_ids=full["input_ids"], attention_mask=full["attention_mask"]).logits
            return npo_completion_only_loss(
                logits,
                full["labels"],
                [reference[example_id] for example_id in full["example_ids"]],
                beta=0.1,
            )

        expected = reference_gradient(model, full_loss)
        for micro in (1, 2, 4):
            objective = UnlearningObjective(
                torch,
                sequences,
                method="npo",
                batch_size=4,
                micro_batch_size=micro,
                pad_token_id=0,
                reference_values=reference,
                beta=0.1,
            )
            assert_same_gradients(self, engine_gradient(model, objective, step=2), expected)


class ResumeEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = MachineSettings.from_env(
            {"UNLEARNING_STORAGE_ROOT": str(self.root), "UNLEARNING_MIN_FREE_DISK_GB": "0"}
        )
        self.packed = torch.randint(1, VOCAB, (16, 10), generator=torch.Generator().manual_seed(11))
        self.accelerator = create_accelerator(cpu=True)

    def tearDown(self):
        self.temporary.cleanup()

    def _components(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step))
        return optimizer, scheduler

    def _loop(self, model, optimizer, scheduler, name: str, stop: GracefulStop | None = None):
        return TrainingLoop(
            accelerator=self.accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=SpecializationObjective(
                self.packed, effective_batch_sequences=4, micro_batch_sequences=2
            ),
            total_steps=6,
            max_grad_norm=1.0,
            run_log=RunLog(self.root / name, run_id="test", stage=name, total_steps=6, console_every_steps=100),
            resume_store=ResumeStore(self.root / name / "resume", keep_last=2, every_minutes=60),
            stop=stop or GracefulStop(),
            settings=self.settings,
        )

    def test_stop_saves_resume_point_and_resume_matches_uninterrupted_run(self):
        straight = tiny_model()
        optimizer, scheduler = self._components(straight)
        result = self._loop(straight, optimizer, scheduler, "straight").run(start_step=0)
        self.assertTrue(result.completed)

        interrupted = tiny_model()
        optimizer, scheduler = self._components(interrupted)
        stop = GracefulStop()
        loop = self._loop(interrupted, optimizer, scheduler, "interrupted", stop=stop)
        loop.on_step_end = lambda step, metrics: setattr(stop, "requested", step == 3) or False
        first = loop.run(start_step=0)
        self.assertFalse(first.completed)
        self.assertEqual(first.last_step, 3)
        points = loop.resume_store.points()
        self.assertEqual([point.step for point in points], [3])

        resumed = Qwen2ForCausalLM.from_pretrained(points[0].model_dir, torch_dtype=torch.float32)
        resumed.config.use_cache = False
        optimizer, scheduler = self._components(resumed)
        apply_resume_state(torch, points[0], optimizer=optimizer, scheduler=scheduler)
        second = self._loop(resumed, optimizer, scheduler, "interrupted").run(start_step=3)
        self.assertTrue(second.completed)
        self.assertEqual(second.last_step, 6)

        for a, b in zip(straight.parameters(), resumed.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)

    def test_metrics_and_status_are_written(self):
        model = tiny_model()
        optimizer, scheduler = self._components(model)
        loop = self._loop(model, optimizer, scheduler, "logged")
        loop.run(start_step=0)
        lines = (self.root / "logged" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 6)
        self.assertIn('"loss"', lines[0])
        self.assertTrue((self.root / "logged" / "status.json").is_file())


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(VOCAB)}, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0", eos_token="t1", pad_token="t1")


class ScientificCheckpointTests(unittest.TestCase):
    def test_s50_save_is_atomic_verifiable_downstream_and_resumable(self):
        from tests.test_config import minimal_config
        from unlearning.config.schema import PilotConfig
        from unlearning.evaluation.provenance import resolve_snapshot_reference
        from unlearning.experiments.naming import experiment_run_id
        from unlearning.runtime.resume import ResumePoint
        from unlearning.training.checkpoints import CheckpointIdentity
        from unlearning.training.contaminate import load_clean_checkpoint_reference
        from unlearning.training.specialize_clean import _save_model_checkpoint

        config = PilotConfig.from_dict(minimal_config())
        run_id = experiment_run_id(config)
        accelerator = create_accelerator(cpu=True)
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        model(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])).loss.backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / run_id
            arguments = dict(
                accelerator=accelerator,
                model=model,
                tokenizer=tiny_tokenizer(),
                optimizer=optimizer,
                scheduler=scheduler,
                torch=torch,
                config=config,
                run_dir=run_dir,
                run_id=run_id,
                name="S50",
                global_step=1645,
                parent_checkpoint_id=CheckpointIdentity(run_id, "S0", 0).checkpoint_id,
                data_metadata={"train_record_fingerprint": "test"},
                hardware={"gpu": "cpu-test"},
                execution={"segments": []},
            )
            checkpoint_id = _save_model_checkpoint(**arguments)
            self.assertFalse((run_dir / "S50.tmp").exists())

            reference = load_clean_checkpoint_reference(run_dir / "S50", config=config)
            self.assertEqual(reference.contaminated_name, "C50")
            self.assertEqual(reference.checkpoint_id, checkpoint_id)
            snapshot = resolve_snapshot_reference(str(run_dir / "S50"), run_dir / "S50" / "manifest.json", config)
            self.assertEqual(snapshot.kind, "local_checkpoint")

            point = ResumePoint(step=1645, directory=run_dir / "S50", kind="scientific")
            reloaded = Qwen2ForCausalLM.from_pretrained(point.model_dir, torch_dtype=torch.float32)
            optimizer2 = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
            scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda step: 1.0)
            apply_resume_state(torch, point, optimizer=optimizer2, scheduler=scheduler2)
            self.assertEqual(scheduler2.last_epoch, scheduler.last_epoch)
            for a, b in zip(model.parameters(), reloaded.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

            with self.assertRaises(FileExistsError):
                _save_model_checkpoint(**arguments)


if __name__ == "__main__":
    unittest.main()
