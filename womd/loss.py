"""The two loss terms: Gaussian likelihood of the assigned anchor's path, and
cross-entropy over which anchor that was.
"""
from __future__ import annotations

import math

import torch

# The constant term of a Gaussian's negative log likelihood, per
# dimension.
HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def anchor_assigned_mode(unit_anchors: torch.Tensor,
                         future_positions: torch.Tensor,
                         future_mask: torch.Tensor) -> torch.Tensor:
    """Assigns each sample the anchor whose constant-speed straight path is
    closest on average to the logged future.
    """
    step_count = future_positions.shape[1]
    step_numbers = torch.arange(
        1,
        step_count + 1,
        device=future_positions.device,
        dtype=future_positions.dtype,
    )
    ramp = step_numbers / step_count
    # Scales each anchor endpoint down to a straight-line path over
    # the future steps, to compare against the real trajectory.
    anchor_paths = unit_anchors.unsqueeze(2) * ramp[None, None, :, None]
    step_errors = anchor_paths - future_positions.unsqueeze(1)
    step_distances = step_errors.norm(dim=-1)
    validity = future_mask.to(step_distances.dtype).unsqueeze(1)
    valid_step_count = validity.sum(dim=-1).clamp_min(1.0)
    mean_distance = (step_distances * validity).sum(dim=-1) / valid_step_count
    return mean_distance.argmin(dim=1)


def gaussian_negative_log_likelihood(predicted_mean: torch.Tensor,
                                     log_standard_deviation: torch.Tensor,
                                     target: torch.Tensor) -> torch.Tensor:
    """Per-step, per-mode Gaussian NLL against the target trajectory, diagonal
    covariance, log-std parameterised for stability.
    """
    error = predicted_mean - target.unsqueeze(1)
    standardised_error = error / log_standard_deviation.exp()
    per_axis_nll = (log_standard_deviation + 0.5 * standardised_error**2 +
                    HALF_LOG_TWO_PI)
    return per_axis_nll.sum(dim=-1)


def prediction_loss(
    trajectories: torch.Tensor, log_standard_deviation: torch.Tensor,
    confidence_logits: torch.Tensor, future_positions: torch.Tensor,
    future_mask: torch.Tensor, unit_anchors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combines the assigned anchor's Gaussian NLL with cross-entropy over
    anchors, averaged only over samples with a valid future.
    """
    step_nll = gaussian_negative_log_likelihood(trajectories,
                                                log_standard_deviation,
                                                future_positions)
    validity = future_mask.unsqueeze(1).to(step_nll.dtype)
    valid_step_count = validity.sum(dim=-1).clamp_min(1.0)
    mode_nll = (step_nll * validity).sum(dim=-1) / valid_step_count
    scoreable = future_mask.any(dim=-1).to(step_nll.dtype)
    scoreable_count = scoreable.sum().clamp_min(1.0)
    assigned_mode = anchor_assigned_mode(unit_anchors, future_positions,
                                         future_mask)
    # Only the assigned anchor's per-mode NLL counts toward the
    # regression term.
    assigned_nll = mode_nll.gather(1, assigned_mode[:, None]).squeeze(1)
    regression = (assigned_nll * scoreable).sum() / scoreable_count

    cross_entropy = torch.nn.functional.cross_entropy(confidence_logits,
                                                      assigned_mode,
                                                      reduction="none")
    classification = (cross_entropy * scoreable).sum() / scoreable_count
    return regression + classification, regression, classification
