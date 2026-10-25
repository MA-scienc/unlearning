"""Shared optimizer loop for every training stage, built on Hugging Face Accelerate.

The same code runs on one laptop GPU (``python script.py``) and on several GPUs
(``accelerate launch script.py``).

Exactness of micro-batching and multi-GPU execution
---------------------------------------------------
Each optimizer step's examples are fixed by the step number alone (the stage's
``Objective.micro_batches``). The stage's loss is defined over the *whole*
step: a token mean (specialization, contamination, GA) or an example mean (NPO).
The engine therefore asks the objective for a **sum** over each micro-batch and
divides by the whole step's token/example count (``Objective.normalizer``),
which is known before any forward pass. Splitting a step into micro-batches, or
across processes, then leaves the objective and its gradient unchanged; only
floating-point summation order differs. (Averaging per-micro-batch means would
silently re-weight examples by length.)

With ``W`` processes, DDP averages gradients across processes, so each process
scales its local sum by ``W / normalizer``; the averaged gradient equals the
gradient of ``total_sum / normalizer``.

Interruptions
-------------
After every optimizer step the loop checks (agreed across processes) whether a
stop was requested (Ctrl+C / SIGTERM) or a timed resume point is due, saves a
resume point when needed, and on a stop request exits cleanly so that
``--resume`` continues at the next step.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from unlearning.runtime.hardware import require_free_disk
from unlearning.runtime.monitor import ThermalGuard
from unlearning.runtime.resume import ResumePoint, ResumeStore, load_training_state, restore_rng_state
from unlearning.runtime.run_log import RunLog
from unlearning.runtime.settings import MachineSettings
from unlearning.runtime.signals import GracefulStop

IGNORE_INDEX = -100

# A stat is reported as sum(values) / sum(counts) over the whole optimizer step.
StatSums = dict[str, tuple[Any, Any]]


@dataclass(frozen=True)
class MicroBatch:
    tensors: dict[str, Any]
    example_ids: tuple[str, ...] = ()

    @property
    def input_tokens(self) -> int:
        mask = self.tensors.get("attention_mask")
        if mask is not None:
            return int(mask.sum())
        return int(self.tensors["input_ids"].numel())


class Objective(Protocol):
    name: str
    unit: str

    def micro_batches(self, step: int) -> list[MicroBatch]:
        """Every micro-batch of optimizer step ``step`` (1-indexed), in a fixed order."""

    def normalizer(self, step: int, micro_batches: Sequence[MicroBatch]) -> float:
        """Total units (tokens or examples) of the whole step; the loss divisor."""

    def loss_sum(self, model, batch: dict[str, Any]) -> tuple[Any, StatSums]:
        """Sum-reduced loss over one micro-batch plus optional (sum, count) stats."""


def supervised_token_count(labels, ignore_index: int = IGNORE_INDEX):
    """Tokens that contribute to a causal-LM loss after the one-token shift."""

    return labels[:, 1:].ne(ignore_index).sum()


def split_into_micro_batches(items: Sequence[Any], micro_batch_size: int) -> list[Sequence[Any]]:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if len(items) % micro_batch_size != 0:
        raise ValueError(
            f"A step of {len(items)} items cannot be split into micro-batches of {micro_batch_size}"
        )
    return [items[start : start + micro_batch_size] for start in range(0, len(items), micro_batch_size)]


def create_accelerator(*, cpu: bool = False):
    """One process on a laptop; one process per GPU under ``accelerate launch``.

    Mixed precision is left to the stages (bf16 weights plus explicit bf16
    autocast) so the numerics match the frozen recipe on every launcher.
    """

    from accelerate import Accelerator

    return Accelerator(cpu=cpu, mixed_precision="no")


def autocast_context(torch, device):
    if getattr(device, "type", str(device)) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def enable_memory_saving(model, *, gradient_checkpointing: bool) -> None:
    if gradient_checkpointing:
        # Non-reentrant checkpointing is required for DDP and is numerically identical.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False


def gradient_norm(torch, parameters) -> float:
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(grad.float(), 2) for grad in grads])
    return float(torch.linalg.vector_norm(norms, 2))


def apply_resume_state(torch, point: ResumePoint, *, optimizer, scheduler) -> dict[str, Any]:
    """Load optimizer/scheduler/RNG from a resume point (weights load separately)."""

    state = load_training_state(torch, point.state_path)
    if int(state["global_step"]) != point.step:
        raise RuntimeError(
            f"Resume point {point.directory} records step {state['global_step']}, expected {point.step}"
        )
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if "scheduler" not in state:
            raise RuntimeError(f"Resume point {point.directory} has no scheduler state")
        scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(torch, state)
    return state


@dataclass
class LoopResult:
    last_step: int
    completed: bool
    stop_reason: str | None
    last_metrics: dict[str, Any] | None
    runtime_seconds: float
    resume_points_saved: int


@dataclass
class TrainingLoop:
    accelerator: Any
    model: Any
    optimizer: Any
    scheduler: Any
    objective: Objective
    total_steps: int
    max_grad_norm: float | None
    run_log: RunLog
    resume_store: ResumeStore
    stop: GracefulStop
    settings: MachineSettings
    thermal: ThermalGuard | None = None
    # Called after each completed step; returns True when it saved a checkpoint
    # holding full training state (a scientific checkpoint that can serve as a
    # resume point), which resets the timed resume interval.
    on_step_end: Callable[[int, dict[str, Any]], bool] | None = None
    resume_meta: Callable[[int, dict[str, Any]], dict[str, Any]] | None = None
    completed_step: int = field(default=0, init=False)

    def run(self, *, start_step: int) -> LoopResult:
        import torch

        accelerator = self.accelerator
        world = accelerator.num_processes
        rank = accelerator.process_index
        device = accelerator.device
        started = time.perf_counter()
        last_metrics: dict[str, Any] | None = None
        saved_points = 0
        step = start_step
        self.completed_step = start_step
        for step in range(start_step + 1, self.total_steps + 1):
            if self.thermal is not None:
                self.thermal.wait_if_hot()
            metrics = self._optimizer_step(torch, step=step, world=world, rank=rank, device=device)
            self.completed_step = step
            last_metrics = metrics
            self.run_log.step(step, metrics)

            saved_full_state = bool(self.on_step_end(step, metrics)) if self.on_step_end else False
            stop_now, save_due = self._agreed_flags(torch, device, world)
            if step == self.total_steps:
                break
            if saved_full_state:
                self.resume_store.mark_saved()
            elif stop_now or save_due:
                self._save_resume_point(torch, step, metrics, reason="stop" if stop_now else "interval")
                saved_points += 1
            if stop_now:
                return LoopResult(
                    last_step=step,
                    completed=False,
                    stop_reason=self.stop.reason or "stop requested",
                    last_metrics=last_metrics,
                    runtime_seconds=time.perf_counter() - started,
                    resume_points_saved=saved_points,
                )
        return LoopResult(
            last_step=step,
            completed=step == self.total_steps,
            stop_reason=None,
            last_metrics=last_metrics,
            runtime_seconds=time.perf_counter() - started,
            resume_points_saved=saved_points,
        )

    def _optimizer_step(self, torch, *, step: int, world: int, rank: int, device) -> dict[str, Any]:
        step_started = time.perf_counter()
        all_micro_batches = self.objective.micro_batches(step)
        if not all_micro_batches or len(all_micro_batches) % world != 0:
            raise RuntimeError(
                f"Step {step} has {len(all_micro_batches)} micro-batches, which cannot be shared "
                f"equally by {world} processes"
            )
        mine = all_micro_batches[rank::world]
        normalizer = float(self.objective.normalizer(step, all_micro_batches))
        if not normalizer > 0:
            raise RuntimeError(f"Step {step} has a non-positive loss normalizer {normalizer}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        self.optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        stat_sums: dict[str, list] = {}
        input_tokens = 0
        for index, micro_batch in enumerate(mine):
            synchronize = index == len(mine) - 1
            sync_context = nullcontext() if synchronize else self.accelerator.no_sync(self.model)
            with sync_context:
                batch = {key: value.to(device, non_blocking=True) for key, value in micro_batch.tensors.items()}
                with autocast_context(torch, device):
                    loss_sum, stats = self.objective.loss_sum(self.model, batch)
                detached = loss_sum.detach().float()
                if not bool(torch.isfinite(detached)):
                    raise RuntimeError(
                        f"Non-finite {self.objective.name} loss at optimizer step {step}: {float(detached)}"
                    )
                (loss_sum * (world / normalizer)).backward()
            loss_total += detached
            input_tokens += micro_batch.input_tokens
            for name, (value_sum, count) in stats.items():
                accumulated = stat_sums.setdefault(name, [0.0, 0.0])
                accumulated[0] += float(value_sum)
                accumulated[1] += float(count)

        parameters = list(self.model.parameters())
        if self.max_grad_norm is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm))
        else:
            grad_norm = gradient_norm(torch, parameters)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        stat_names = sorted(stat_sums)
        packed = torch.tensor(
            [float(loss_total), float(input_tokens)]
            + [stat_sums[name][0] for name in stat_names]
            + [stat_sums[name][1] for name in stat_names],
            device=device,
            dtype=torch.float64 if device.type != "mps" else torch.float32,
        )
        if world > 1:
            packed = self.accelerator.reduce(packed, reduction="sum")
        values = packed.tolist()
        step_seconds = time.perf_counter() - step_started
        metrics: dict[str, Any] = {
            "loss": values[0] / normalizer,
            "lr": learning_rate,
            "grad_norm": grad_norm,
            "grad_clipped": bool(self.max_grad_norm is not None and grad_norm > self.max_grad_norm),
            "loss_units": normalizer,
            "unit": self.objective.unit,
            "input_tokens": int(values[1]),
            "step_seconds": step_seconds,
            "tokens_per_second": values[1] / step_seconds if step_seconds > 0 else None,
            "processes": world,
            "micro_batches": len(all_micro_batches),
        }
        for offset, name in enumerate(stat_names):
            total = values[2 + offset]
            count = values[2 + len(stat_names) + offset]
            metrics[name] = total / count if count else None
        if device.type == "cuda":
            metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
            metrics["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
        return metrics

    def _agreed_flags(self, torch, device, world: int) -> tuple[bool, bool]:
        flags = torch.tensor(
            [1.0 if self.stop.requested else 0.0, 1.0 if self.resume_store.due() else 0.0],
            device=device,
        )
        if world > 1:
            import torch.distributed as dist

            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        return bool(flags[0] > 0), bool(flags[1] > 0)

    def _save_resume_point(self, torch, step: int, metrics: dict[str, Any], *, reason: str) -> None:
        if self.accelerator.is_main_process:
            require_free_disk(self.resume_store.directory, self.settings, action="saving a resume point")
        self.run_log.set_state("saving", message=f"resume point at step {step}")
        saved_at = time.perf_counter()
        extra = self.resume_meta(step, metrics) if self.resume_meta else {}
        point = self.resume_store.save(
            step=step,
            accelerator=self.accelerator,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            torch=torch,
            meta={"reason": reason, "last_loss": metrics.get("loss"), **extra},
        )
        self.run_log.event(
            "resume_point_saved",
            f"step {step} ({reason}) in {time.perf_counter() - saved_at:.0f}s",
            step=step,
            path=str(point.directory),
        )
        self.run_log.update(last_resume_point={"step": step, "path": str(point.directory)})
