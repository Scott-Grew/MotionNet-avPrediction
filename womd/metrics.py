"""Training-time monitor of minADE and minFDE.

It steers runs; reported numbers come only from Waymo's scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
from torch import Tensor

from womd import contract
from womd.loader import SceneBatch, TargetArrays
from womd.loss import LossTerms
from womd.model import QUERY_COUNT, ModePredictions
from womd.pruning import prune_modes

# A progress line is printed once per this many batches.
LOG_EVERY_BATCHES = 20


class TrainingStep(NamedTuple):
    """One training step's losses and the raw predictions the monitor reads."""

    losses: LossTerms
    predicted: ModePredictions


class StepOutcome(NamedTuple):
    """What the optimiser did with one step's gradients."""

    clipped: bool
    skipped: bool  # the scaler found non-finite gradients


class MonitorResults(NamedTuple):
    """The training monitor over the batches it saw. Distances are in metres
    over the 80 future steps, and each value is nan when nothing was counted.
    """

    min_ade: float
    min_fde: float
    mean_kept_modes: float
    backfill_rate: float  # share of samples that kept fewer than 6 modes


class LossValues(NamedTuple):
    """The total loss and its two likelihood terms, as floats."""

    total: float
    regression: float
    classification: float

    def plus(self, other: LossValues) -> LossValues:
        return LossValues(
            *(mine + theirs for mine, theirs in zip(self, other, strict=True))
        )

    def over(self, count: int) -> LossValues:
        return LossValues(*(value / max(count, 1) for value in self))


@dataclass
class PhaseSeconds:
    """Wall-clock seconds spent waiting for data, stepping and monitoring."""

    data_wait: float = 0.0
    step: float = 0.0
    monitor: float = 0.0

    def elapsed(self) -> float:
        return sum(getattr(self, phase.name) for phase in fields(self))


class EpochSummary(NamedTuple):
    """One finished epoch, as the epoch report and checkpoint read it."""

    losses: LossValues  # mean per batch
    monitor: MonitorResults
    seconds: PhaseSeconds


def mean_or_nan(total: Tensor | float, count: Tensor | int) -> float:
    """A running sum over its count, or nan when nothing was counted."""
    return float(total / count) if count else float("nan")


def mode_distances(trajectories: Tensor, targets: TargetArrays) -> Tensor:
    """Average per-step distance to the ground truth over valid future steps,
    separately for each mode.
    """
    step_distances = (
        trajectories - targets.future_positions.unsqueeze(1)
    ).norm(dim=-1)
    validity = targets.future_mask.unsqueeze(1).to(step_distances.dtype)
    valid_step_count = validity.sum(dim=-1).clamp_min(1.0)
    return (step_distances * validity).sum(dim=-1) / valid_step_count


class MetricAccumulator:
    """Accumulates minADE, minFDE and mode-pruning counts across batches as a
    training-time monitor; not the reported score.
    """

    def __init__(self) -> None:
        self.ade_sum = 0.0
        self.ade_count = 0
        self.fde_sum = 0.0
        self.fde_count = 0
        self.kept_mode_sum = 0
        self.backfilled_count = 0
        self.sample_count = 0

    def update(self, predicted: ModePredictions, targets: TargetArrays) -> None:
        """Prunes to the kept modes, then folds this batch's minADE, minFDE and
        mode-pruning counts into the running totals.
        """
        pruned = prune_modes(
            predicted.trajectories, predicted.confidence_logits
        )
        distances = (
            pruned.trajectories - targets.future_positions.unsqueeze(1)
        ).norm(dim=-1)
        average_distances = mode_distances(pruned.trajectories, targets)

        has_valid_step = targets.future_mask.any(dim=-1)
        best_average = average_distances.min(dim=-1).values
        self.ade_sum += (best_average * has_valid_step).sum()
        self.ade_count += has_valid_step.sum()

        final_valid = targets.future_mask[:, -1]
        best_final = distances[:, :, -1].min(dim=-1).values
        self.fde_sum += (best_final * final_valid).sum()
        self.fde_count += final_valid.sum()

        self.kept_mode_sum += pruned.kept_count.sum()
        self.backfilled_count += (
            pruned.kept_count < contract.NUM_PREDICTED_MODES
        ).sum()
        self.sample_count += predicted.confidence_logits.shape[0]

    def results(self) -> MonitorResults:
        return MonitorResults(
            min_ade=mean_or_nan(self.ade_sum, self.ade_count),
            min_fde=mean_or_nan(self.fde_sum, self.fde_count),
            mean_kept_modes=mean_or_nan(self.kept_mode_sum, self.sample_count),
            backfill_rate=mean_or_nan(self.backfilled_count, self.sample_count),
        )


class EpochProgress:
    """Running totals for one epoch, the losses, the training monitor,
    optimiser health and timing, overall and per logging window.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.accumulator = MetricAccumulator()
        self.window_accumulator = MetricAccumulator()
        self.loss_sums = LossValues(0.0, 0.0, 0.0)
        self.window_loss_sums = LossValues(0.0, 0.0, 0.0)
        self.seconds = PhaseSeconds()
        self.batch_count = 0
        self.sample_count = 0
        self.non_finite_count = 0
        self.skipped_count = 0
        self.clipped_count = 0
        self.window_winner_counts = torch.zeros(
            QUERY_COUNT, dtype=torch.long, device=device
        )

    def record_step(self, step: TrainingStep, outcome: StepOutcome) -> bool:
        """Folds one step's losses and optimiser health into the totals, and
        returns whether the total loss was finite.
        """
        losses = LossValues(*(float(term.detach()) for term in step.losses))
        self.loss_sums = self.loss_sums.plus(losses)
        self.window_loss_sums = self.window_loss_sums.plus(losses)
        total_is_finite = math.isfinite(losses.total)
        self.non_finite_count += int(not total_is_finite)
        self.clipped_count += int(outcome.clipped)
        self.skipped_count += int(outcome.skipped)
        self.batch_count += 1
        return total_is_finite

    def record_monitor(self, step: TrainingStep, batch: SceneBatch) -> None:
        """Every batch counts which mode came closest, to spot modes that never
        win; the pruned minADE and minFDE run on one batch in
        LOG_EVERY_BATCHES, because pruning is the monitor's slow part.
        """
        self.sample_count += batch.targets.agent_history.shape[0]
        predicted = ModePredictions(
            *(tensor.detach().float() for tensor in step.predicted)
        )
        with torch.no_grad():
            winners = mode_distances(
                predicted.trajectories, batch.targets
            ).argmin(dim=1)
            self.window_winner_counts.scatter_add_(
                0, winners, torch.ones_like(winners)
            )
            if self.batch_count % LOG_EVERY_BATCHES != 0:
                return
            self.accumulator.update(predicted, batch.targets)
            self.window_accumulator.update(predicted, batch.targets)

    def window_scalars(self, learning_rate: float) -> dict[str, float]:
        """The table reported every LOG_EVERY_BATCHES, the losses, the training
        monitor, optimiser health and where the time went.
        """
        monitor = self.accumulator.results()
        window_monitor = self.window_accumulator.results()
        never_win_count = int((self.window_winner_counts == 0).sum())
        peak_gigabytes = 0.0
        if self.device.type == "cuda":
            peak_gigabytes = torch.cuda.max_memory_allocated() / 1e9
        averages = self.loss_sums.over(self.batch_count)
        seconds = self.seconds
        elapsed = seconds.elapsed()
        return {
            "loss/total": averages.total,
            "loss/regression": averages.regression,
            "loss/classification": averages.classification,
            "loss_window/total": self.window_loss_sums.total
            / LOG_EVERY_BATCHES,
            "monitor/ade_80step": monitor.min_ade,
            "monitor/fde_80step": monitor.min_fde,
            "monitor_window/ade_80step": window_monitor.min_ade,
            "monitor_window/fde_80step": window_monitor.min_fde,
            "monitor_window/kept_modes": window_monitor.mean_kept_modes,
            "monitor_window/backfill_rate": window_monitor.backfill_rate,
            "monitor_window/never_win_anchors": never_win_count,
            "health/non_finite_losses": self.non_finite_count,
            "health/skipped_steps": self.skipped_count,
            "health/clipped_steps": self.clipped_count,
            "optimisation/learning_rate": learning_rate,
            "throughput/samples_per_second": self.sample_count / elapsed,
            "time_share/data_wait": seconds.data_wait / elapsed,
            "time_share/step": seconds.step / elapsed,
            "time_share/monitor": seconds.monitor / elapsed,
            "memory/peak_gigabytes": peak_gigabytes,
        }

    def start_new_window(self) -> None:
        self.window_accumulator = MetricAccumulator()
        self.window_loss_sums = LossValues(0.0, 0.0, 0.0)
        self.window_winner_counts.zero_()

    def summary(self) -> EpochSummary:
        return EpochSummary(
            losses=self.loss_sums.over(self.batch_count),
            monitor=self.accumulator.results(),
            seconds=self.seconds,
        )


def epoch_scalars(summary: EpochSummary) -> dict[str, float]:
    """The epoch report's table, keyed by the names its charts use."""
    losses, monitor, seconds = summary
    return {
        "epoch/loss_total": losses.total,
        "epoch/loss_regression": losses.regression,
        "epoch/loss_classification": losses.classification,
        "epoch/ade_80step": monitor.min_ade,
        "epoch/fde_80step": monitor.min_fde,
        "epoch/kept_modes": monitor.mean_kept_modes,
        "epoch/backfill_rate": monitor.backfill_rate,
        "epoch/data_wait_seconds": seconds.data_wait,
        "epoch/step_seconds": seconds.step,
        "epoch/monitor_seconds": seconds.monitor,
    }
