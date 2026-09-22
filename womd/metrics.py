"""Training-time monitor of minADE and minFDE.

It steers runs; reported numbers come only from Waymo's scorer.
"""
from __future__ import annotations

import torch

from womd import contract
from womd.pruning import prune_modes_batched_with_kept_count


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

    def results(self) -> dict[str, float]:
        backfilled = self.backfilled_sample_count
        return {
            "min_ade": self.mean_or_nan(self.ade_sum, self.ade_count),
            "min_fde": self.mean_or_nan(self.fde_sum, self.fde_count),
            "mean_kept_modes": self.mean_or_nan(self.kept_mode_sum,
                                                self.sample_count),
            "backfill_rate": self.mean_or_nan(backfilled, self.sample_count),
        }
