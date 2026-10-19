"""Training-time monitor: losses, minADE and minFDE on the kept modes, how many
anchors never win, and optimiser health.

It steers runs; reported numbers come only from Waymo's scorer.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple

import torch
from torch import Tensor

from womd.loader import TargetArrays
from womd.loss import LossTerms, mean_over_valid_steps
from womd.model import QUERY_COUNT, ModePredictions
from womd.pruning import prune_modes

# A log line is printed once per this many batches, and the pruned
# minADE and minFDE run on that batch only, because pruning is slow.
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


class EpochSummary(NamedTuple):
    """One finished epoch: mean losses per batch, the monitor's minADE and
    minFDE in metres, and the wall-clock seconds it took.
    """

    losses: LossValues
    min_ade: float
    min_fde: float
    seconds: float


def mean_or_nan(total: Tensor | float, count: Tensor | int) -> float:
    """A running sum over its count, or nan when nothing was counted."""
    return float(total / count) if count else float("nan")


def mode_distances(trajectories: Tensor, targets: TargetArrays) -> Tensor:
    """Each mode's mean distance to the logged future over its valid steps."""
    logged = targets.future_positions.unsqueeze(1)
    step_distances = (trajectories - logged).norm(dim=-1)
    return mean_over_valid_steps(step_distances, targets.future_mask)


class MetricAccumulator:
    """minADE and minFDE over the 6 kept modes, summed across the batches it
    is given; a monitor, not the reported score.
    """

    def __init__(self) -> None:
        self.ade_sum = 0.0
        self.ade_count = 0
        self.fde_sum = 0.0
        self.fde_count = 0

    def update(self, predicted: ModePredictions, targets: TargetArrays) -> None:
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

    def min_ade(self) -> float:
        return mean_or_nan(self.ade_sum, self.ade_count)

    def min_fde(self) -> float:
        return mean_or_nan(self.fde_sum, self.fde_count)


class EpochProgress:
    """Running totals for one epoch: losses, optimiser health, samples seen,
    the monitor, and which anchors' modes came closest since the last log
    line, to spot anchors that never win.
    """

    def __init__(self, device: torch.device) -> None:
        self.start = time.perf_counter()
        self.batch_count = 0
        self.sample_count = 0
        self.loss_sums = LossValues(0.0, 0.0, 0.0)
        self.monitor = MetricAccumulator()
        self.non_finite_count = 0
        self.skipped_count = 0
        self.clipped_count = 0
        self.winner_counts = torch.zeros(
            QUERY_COUNT, dtype=torch.long, device=device
        )

    def record(self, step: TrainingStep, outcome: StepOutcome) -> bool:
        """Folds one step's losses and optimiser health into the totals and
        returns whether its total loss was finite.
        """
        losses = LossValues(*(float(term.detach()) for term in step.losses))
        self.loss_sums = self.loss_sums.plus(losses)
        self.batch_count += 1
        loss_is_finite = math.isfinite(losses.total)
        self.non_finite_count += int(not loss_is_finite)
        self.clipped_count += int(outcome.clipped)
        self.skipped_count += int(outcome.skipped)
        return loss_is_finite

    def monitor_batch(self, step: TrainingStep, targets: TargetArrays) -> None:
        """Counts the batch's samples and closest modes, and on every
        LOG_EVERY_BATCHES-th batch adds it to the pruned minADE and minFDE.
        """
        self.sample_count += targets.agent_history.shape[0]
        detached = (tensor.detach().float() for tensor in step.predicted)
        predicted = ModePredictions(*detached)
        with torch.no_grad():
            distances = mode_distances(predicted.trajectories, targets)
            winners = distances.argmin(dim=1)
            self.winner_counts.scatter_add_(
                0, winners, torch.ones_like(winners)
            )
            if self.batch_count % LOG_EVERY_BATCHES == 0:
                self.monitor.update(predicted, targets)

    def log_scalars(self, learning_rate: float) -> Scalars:
        """The table printed every LOG_EVERY_BATCHES, after which the
        never-win count starts again.
        """
        losses = self.loss_sums.over(self.batch_count)
        never_win_count = int((self.winner_counts == 0).sum())
        self.winner_counts.zero_()
        elapsed = time.perf_counter() - self.start
        return {
            "loss/total": losses.total,
            "loss/regression": losses.regression,
            "loss/classification": losses.classification,
            "monitor/ade_80step": self.monitor.min_ade(),
            "monitor/fde_80step": self.monitor.min_fde(),
            "monitor/never_win_anchors": never_win_count,
            "health/non_finite_losses": self.non_finite_count,
            "health/skipped_steps": self.skipped_count,
            "health/clipped_steps": self.clipped_count,
            "optimisation/learning_rate": learning_rate,
            "throughput/samples_per_second": self.sample_count / elapsed,
        }

    def summary(self) -> EpochSummary:
        losses = self.loss_sums.over(self.batch_count)
        seconds = time.perf_counter() - self.start
        ade = self.monitor.min_ade()
        fde = self.monitor.min_fde()
        return EpochSummary(losses, ade, fde, seconds)


def epoch_scalars(summary: EpochSummary) -> Scalars:
    """The epoch report's table, keyed by the names its charts use."""
    return {
        "epoch/loss_total": summary.losses.total,
        "epoch/loss_regression": summary.losses.regression,
        "epoch/loss_classification": summary.losses.classification,
        "epoch/ade_80step": summary.min_ade,
        "epoch/fde_80step": summary.min_fde,
        "epoch/seconds": summary.seconds,
    }
