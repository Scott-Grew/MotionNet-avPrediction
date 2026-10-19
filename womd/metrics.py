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
from womd.loss import LossTerms, mean_over_valid_steps
from womd.model import QUERY_COUNT, ModePredictions
from womd.pruning import prune_modes

# A progress line is printed once per this many batches.
LOG_EVERY_BATCHES = 20

Scalars = dict[str, float]


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
        pairs = zip(self, other, strict=True)
        return LossValues(*(mine + theirs for mine, theirs in pairs))

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
    """Each mode's mean distance to the logged future over its valid steps."""
    logged = targets.future_positions.unsqueeze(1)
    step_distances = (trajectories - logged).norm(dim=-1)
    return mean_over_valid_steps(step_distances, targets.future_mask)


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
        logits = predicted.confidence_logits
        pruned = prune_modes(predicted.trajectories, logits)
        logged = targets.future_positions.unsqueeze(1)
        distances = (pruned.trajectories - logged).norm(dim=-1)
        average_distances = mode_distances(pruned.trajectories, targets)

        has_valid_step = targets.future_mask.any(dim=-1)
        best_average = average_distances.min(dim=-1).values
        self.ade_sum += (best_average * has_valid_step).sum()
        self.ade_count += has_valid_step.sum()

        final_valid = targets.future_mask[:, -1]
        best_final = distances[:, :, -1].min(dim=-1).values
        self.fde_sum += (best_final * final_valid).sum()
        self.fde_count += final_valid.sum()

        backfilled = pruned.kept_count < contract.NUM_PREDICTED_MODES
        self.kept_mode_sum += pruned.kept_count.sum()
        self.backfilled_count += backfilled.sum()
        self.sample_count += logits.shape[0]

    def results(self) -> MonitorResults:
        return MonitorResults(
            min_ade=mean_or_nan(self.ade_sum, self.ade_count),
            min_fde=mean_or_nan(self.fde_sum, self.fde_count),
            mean_kept_modes=mean_or_nan(self.kept_mode_sum, self.sample_count),
            backfill_rate=mean_or_nan(self.backfilled_count, self.sample_count),
        )


class Tally:
    """Losses summed per batch and the pruned monitor, over one stretch of
    batches; the epoch keeps one and each logging window another.
    """

    def __init__(self) -> None:
        self.batch_count = 0
        self.loss_sums = LossValues(0.0, 0.0, 0.0)
        self.monitor = MetricAccumulator()

    def add(self, losses: LossValues) -> None:
        self.batch_count += 1
        self.loss_sums = self.loss_sums.plus(losses)

    def mean_losses(self) -> LossValues:
        return self.loss_sums.over(self.batch_count)


class EpochProgress:
    """Running totals for one epoch: losses and the monitor for the epoch and
    for the current logging window, optimiser health, and timing.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.epoch = Tally()
        self.window = Tally()
        self.seconds = PhaseSeconds()
        self.sample_count = 0
        self.non_finite_count = 0
        self.skipped_count = 0
        self.clipped_count = 0
        self.winner_counts = torch.zeros(
            QUERY_COUNT, dtype=torch.long, device=device
        )

    @property
    def batch_count(self) -> int:
        return self.epoch.batch_count

    def record_step(self, step: TrainingStep, outcome: StepOutcome) -> bool:
        """Folds one step's losses and optimiser health into the totals, and
        returns whether the total loss was finite.
        """
        losses = LossValues(*(float(term.detach()) for term in step.losses))
        self.epoch.add(losses)
        self.window.add(losses)
        total_is_finite = math.isfinite(losses.total)
        self.non_finite_count += int(not total_is_finite)
        self.clipped_count += int(outcome.clipped)
        self.skipped_count += int(outcome.skipped)
        return total_is_finite

    def record_monitor(self, step: TrainingStep, batch: SceneBatch) -> None:
        """Every batch counts which mode came closest, to spot modes that never
        win; the pruned minADE and minFDE run on one batch in
        LOG_EVERY_BATCHES, because pruning is the monitor's slow part.
        """
        targets = batch.targets
        self.sample_count += targets.agent_history.shape[0]
        detached = (tensor.detach().float() for tensor in step.predicted)
        predicted = ModePredictions(*detached)
        with torch.no_grad():
            distances = mode_distances(predicted.trajectories, targets)
            winners = distances.argmin(dim=1)
            self.winner_counts.scatter_add_(
                0, winners, torch.ones_like(winners)
            )
            if self.batch_count % LOG_EVERY_BATCHES != 0:
                return
            self.epoch.monitor.update(predicted, targets)
            self.window.monitor.update(predicted, targets)

    def peak_gigabytes(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated() / 1e9

    def window_scalars(self, learning_rate: float) -> Scalars:
        """The table reported every LOG_EVERY_BATCHES: the losses, the training
        monitor, optimiser health and where the time went.
        """
        losses = self.epoch.mean_losses()
        monitor = self.epoch.monitor.results()
        window = self.window.monitor.results()
        never_win_count = int((self.winner_counts == 0).sum())
        seconds = self.seconds
        elapsed = seconds.elapsed()
        return {
            "loss/total": losses.total,
            "loss/regression": losses.regression,
            "loss/classification": losses.classification,
            "loss_window/total": self.window.mean_losses().total,
            "monitor/ade_80step": monitor.min_ade,
            "monitor/fde_80step": monitor.min_fde,
            "monitor_window/ade_80step": window.min_ade,
            "monitor_window/fde_80step": window.min_fde,
            "monitor_window/kept_modes": window.mean_kept_modes,
            "monitor_window/backfill_rate": window.backfill_rate,
            "monitor_window/never_win_anchors": never_win_count,
            "health/non_finite_losses": self.non_finite_count,
            "health/skipped_steps": self.skipped_count,
            "health/clipped_steps": self.clipped_count,
            "optimisation/learning_rate": learning_rate,
            "throughput/samples_per_second": self.sample_count / elapsed,
            "time_share/data_wait": seconds.data_wait / elapsed,
            "time_share/step": seconds.step / elapsed,
            "time_share/monitor": seconds.monitor / elapsed,
            "memory/peak_gigabytes": self.peak_gigabytes(),
        }

    def start_new_window(self) -> None:
        self.window = Tally()
        self.winner_counts.zero_()

    def summary(self) -> EpochSummary:
        losses = self.epoch.mean_losses()
        monitor = self.epoch.monitor.results()
        return EpochSummary(losses, monitor, self.seconds)


def epoch_scalars(summary: EpochSummary) -> Scalars:
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
