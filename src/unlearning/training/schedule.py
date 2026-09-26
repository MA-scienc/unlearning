from __future__ import annotations

from dataclasses import dataclass

from unlearning.config import PilotConfig


@dataclass(frozen=True)
class SpecializationSchedule:
    provisional_s100_steps: int
    s50_steps: int
    s100_steps: int
    step_adjustment: int
    effective_batch_sequences: int
    micro_batch_sequences: int
    gradient_accumulation_steps: int

    @property
    def checkpoint_steps(self) -> tuple[int, int]:
        return (self.s50_steps, self.s100_steps)


def freeze_even_s100(provisional_s100_steps: int) -> tuple[int, int]:
    if provisional_s100_steps <= 0:
        raise ValueError("provisional_s100_steps must be positive")
    frozen = provisional_s100_steps if provisional_s100_steps % 2 == 0 else provisional_s100_steps - 1
    if frozen <= 0:
        raise ValueError("frozen S100 step count must be positive")
    return frozen, provisional_s100_steps - frozen


def build_schedule(config: PilotConfig, provisional_s100_steps: int) -> SpecializationSchedule:
    frozen_s100, adjustment = freeze_even_s100(provisional_s100_steps)
    effective = config.specialization.effective_batch_sequences
    micro = config.specialization.micro_batch_sequences
    if effective % micro != 0:
        raise ValueError("effective batch size must be divisible by micro batch size")
    return SpecializationSchedule(
        provisional_s100_steps=provisional_s100_steps,
        s50_steps=frozen_s100 // 2,
        s100_steps=frozen_s100,
        step_adjustment=adjustment,
        effective_batch_sequences=effective,
        micro_batch_sequences=micro,
        gradient_accumulation_steps=effective // micro,
    )


def expected_pilot_schedule(config: PilotConfig) -> SpecializationSchedule:
    return build_schedule(config, provisional_s100_steps=3291)
