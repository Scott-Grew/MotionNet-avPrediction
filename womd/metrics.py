"""Training-time monitor of minADE and minFDE.

It steers runs; reported numbers come only from Waymo's scorer.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, fields
import math
from typing import NamedTuple

import torch

from womd import contract
from womd.loader import SceneBatch
from womd.model import QUERY_COUNT
from womd.pruning import prune_modes_batched_with_kept_count

# One training step's three loss values and the raw predictions
# the training monitor reads.
TrainingStep = namedtuple(
    "TrainingStep",
    "total regression classification trajectories confidence_logits",
)

# A progress line is printed once per this many batches.
LOG_EVERY_BATCHES = 20


class MonitorResults(NamedTuple):
    """The training monitor over the batches it saw. Distances are in metres
    over the 80 future steps, and each value is nan when nothing was counted.
    """
    min_ade: float
    min_fde: float
    mean_kept_modes: float
    backfill_rate: float  # share of samples that kept fewer than 6 modes


class LossValues(NamedTuple):
    """The total loss and its two likelihood terms."""
    total: float
    regression: float
    classification: float


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


def mean_distance_per_mode(trajectories: torch.Tensor,
                           future_positions: torch.Tensor,
                           future_mask: torch.Tensor) -> torch.Tensor:
    """Averages per-step Euclidean distance to the ground truth over valid
    future steps, separately for each mode.
    """
    step_errors = trajectories - future_positions.unsqueeze(1)
    step_distances = step_errors.norm(dim=-1)
    validity = future_mask.unsqueeze(1).to(step_distances.dtype)
    valid_step_count = validity.sum(dim=-1).clamp_min(1.0)
    return (step_distances * validity).sum(dim=-1) / valid_step_count


class MetricAccumulator:
    """Accumulates minADE/minFDE and mode-pruning stats across batches as a
    training-time monitor; not the reported score.
    """

    def __init__(self) -> None:
        self.ade_sum = 0.0
        self.ade_count = 0
        self.fde_sum = 0.0
        self.fde_count = 0
        self.kept_mode_sum = 0
        self.backfilled_sample_count = 0
        self.sample_count = 0

    def update(self, trajectories: torch.Tensor,
               confidence_logits: torch.Tensor, future_positions: torch.Tensor,
               future_mask: torch.Tensor) -> None:
        """Prunes to the kept modes, then folds this batch's minADE, minFDE and
        mode-pruning counts into the running totals.
        """
        pruned = prune_modes_batched_with_kept_count(trajectories,
                                                     confidence_logits)
        kept_trajectories, _, kept_mode_count = pruned

        step_errors = kept_trajectories - future_positions.unsqueeze(1)
        distances = step_errors.norm(dim=-1)
        # Zeros invalid steps before summing so they don't bias
        # the per-mode average distance.
        valid_steps = future_mask.unsqueeze(1)
        valid_distances = torch.where(valid_steps, distances,
                                      torch.zeros_like(distances))
        valid_step_count = future_mask.sum(dim=-1, keepdim=True)
        valid_step_count = valid_step_count.clamp_min(1)
        average_distances = valid_distances.sum(dim=-1) / valid_step_count

        has_any_valid_step = future_mask.any(dim=-1)
        best_average_distance = average_distances.min(dim=-1).values
        self.ade_sum += (best_average_distance * has_any_valid_step).sum()
        self.ade_count += has_any_valid_step.sum()

        final_step_valid = future_mask[:, -1]
        best_final_distance = distances[:, :, -1].min(dim=-1).values
        self.fde_sum += (best_final_distance * final_step_valid).sum()
        self.fde_count += final_step_valid.sum()

        was_backfilled = kept_mode_count < contract.NUM_PREDICTED_MODES
        self.kept_mode_sum += kept_mode_count.sum()
        self.backfilled_sample_count += was_backfilled.sum()
        self.sample_count += confidence_logits.shape[0]

    @staticmethod
    def mean_or_nan(running_sum: torch.Tensor | float,
                    count: torch.Tensor | int) -> float:
        """A running sum over its count, or nan when nothing was counted."""
        return float(running_sum / count) if count else float("nan")

    def results(self) -> MonitorResults:
        return MonitorResults(
            min_ade=self.mean_or_nan(self.ade_sum, self.ade_count),
            min_fde=self.mean_or_nan(self.fde_sum, self.fde_count),
            mean_kept_modes=self.mean_or_nan(self.kept_mode_sum,
                                             self.sample_count),
            backfill_rate=self.mean_or_nan(self.backfilled_sample_count,
                                           self.sample_count),
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
        self.non_finite_total_count = 0
        self.gradient_scaler_skip_count = 0
        self.clipped_step_count = 0
        self.window_winner_counts = torch.zeros(QUERY_COUNT,
                                                dtype=torch.long,
                                                device=device)

    def record_step(self, step: TrainingStep, was_clipped: bool,
                    was_skipped: bool, sample_count: int) -> bool:
        """Folds one step's losses and optimiser health into the totals, and
        returns whether the total loss was finite.
        """
        loss_values = LossValues(
            total=float(step.total.detach()),
            regression=float(step.regression.detach()),
            classification=float(step.classification.detach()),
        )
        self.loss_sums = LossValues(
            *(sum_so_far + value
              for sum_so_far, value in zip(self.loss_sums, loss_values)))
        self.window_loss_sums = LossValues(
            *(sum_so_far + value
              for sum_so_far, value in zip(self.window_loss_sums, loss_values)))
        total_is_finite = math.isfinite(loss_values.total)
        self.non_finite_total_count += int(not total_is_finite)
        self.clipped_step_count += int(was_clipped)
        self.gradient_scaler_skip_count += int(was_skipped)
        self.batch_count += 1
        self.sample_count += sample_count
        return total_is_finite

    def record_monitor(self, step: TrainingStep, batch: SceneBatch) -> None:
        """Records the training monitor, which steers runs and is never a
        reported number. Every batch counts which mode came closest, to spot
        modes that never win; the pruned minADE and minFDE run on one batch
        in LOG_EVERY_BATCHES, because pruning is the monitor's slow part.
        """
        trajectories = step.trajectories.detach().float()
        confidence_logits = step.confidence_logits.detach().float()
        future_positions = batch.targets.future_positions
        future_mask = batch.targets.future_mask
        with torch.no_grad():
            mode_distances = mean_distance_per_mode(trajectories,
                                                    future_positions,
                                                    future_mask)
            window_winners = mode_distances.argmin(dim=1)
            self.window_winner_counts.scatter_add_(
                0, window_winners, torch.ones_like(window_winners))
            if self.batch_count % LOG_EVERY_BATCHES != 0:
                return
            for metric_accumulator in (self.accumulator,
                                       self.window_accumulator):
                metric_accumulator.update(trajectories, confidence_logits,
                                          future_positions, future_mask)

    def window_scalars(self, step_learning_rate: float) -> dict[str, float]:
        """The table reported every LOG_EVERY_BATCHES, the losses, the training
        monitor, optimiser health and where the time went.
        """
        monitor = self.accumulator.results()
        window_monitor = self.window_accumulator.results()
        never_win_count = int((self.window_winner_counts == 0).sum())
        peak_gigabytes = 0.0
        if self.device.type == "cuda":
            peak_gigabytes = torch.cuda.max_memory_allocated() / 1e9
        averages = self.averages()
        seconds = self.seconds
        elapsed = seconds.elapsed()
        return {
            "loss/total": averages.total,
            "loss/regression": averages.regression,
            "loss/classification": averages.classification,
            "loss_window/total": self.window_loss_sums.total /
                                 LOG_EVERY_BATCHES,
            "monitor/ade_80step": monitor.min_ade,
            "monitor/fde_80step": monitor.min_fde,
            "monitor_window/ade_80step": window_monitor.min_ade,
            "monitor_window/fde_80step": window_monitor.min_fde,
            "monitor_window/kept_modes": window_monitor.mean_kept_modes,
            "monitor_window/backfill_rate": window_monitor.backfill_rate,
            "monitor_window/never_win_anchors": never_win_count,
            "health/non_finite_losses": self.non_finite_total_count,
            "health/skipped_steps": self.gradient_scaler_skip_count,
            "health/clipped_steps": self.clipped_step_count,
            "optimisation/learning_rate": step_learning_rate,
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

    def averages(self) -> LossValues:
        """Mean of each loss term over the batches seen so far."""
        return LossValues(
            *(value / max(self.batch_count, 1) for value in self.loss_sums))

    def summary(self) -> EpochSummary:
        return EpochSummary(losses=self.averages(),
                            monitor=self.accumulator.results(),
                            seconds=self.seconds)


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
