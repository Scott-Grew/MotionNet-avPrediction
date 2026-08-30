import math

import torch

from womd import contract

HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def anchor_assigned_mode(unit_anchors, future_positions, future_mask):
    step_count = future_positions.shape[1]
    ramp = (
        torch.arange(
            1,
            step_count + 1,
            device=future_positions.device,
            dtype=future_positions.dtype,
        )
        / step_count
    )
    anchor_paths = (
        unit_anchors.unsqueeze(2) * ramp[None, None, :, None]
    )
    step_distances = (
        anchor_paths - future_positions.unsqueeze(1)
    ).norm(dim=-1)
    validity = future_mask.to(step_distances.dtype).unsqueeze(1)
    mean_distance = (step_distances * validity).sum(
        dim=-1
    ) / validity.sum(dim=-1).clamp_min(1.0)
    return mean_distance.argmin(dim=1)


def gaussian_negative_log_likelihood(
    predicted_mean, log_standard_deviation, target
):
    standardised_error = (
        predicted_mean - target.unsqueeze(1)
    ) / log_standard_deviation.exp()
    return (
        log_standard_deviation
        + 0.5 * standardised_error**2
        + HALF_LOG_TWO_PI
    ).sum(dim=-1)


def prediction_loss(
    trajectories,
    log_standard_deviation,
    confidence_logits,
    future_positions,
    future_mask,
    unit_anchors,
):
    step_nll = gaussian_negative_log_likelihood(
        trajectories, log_standard_deviation, future_positions
    )
    validity = future_mask.unsqueeze(1).to(step_nll.dtype)
    mode_nll = (step_nll * validity).sum(dim=-1) / validity.sum(
        dim=-1
    ).clamp_min(1.0)
    scoreable = future_mask.any(dim=-1).to(step_nll.dtype)
    scoreable_count = scoreable.sum().clamp_min(1.0)
    assigned_mode = anchor_assigned_mode(
        unit_anchors, future_positions, future_mask
    )
    regression = (
        mode_nll.gather(1, assigned_mode[:, None]).squeeze(1)
        * scoreable
    ).sum() / scoreable_count
    classification = (
        torch.nn.functional.cross_entropy(
            confidence_logits, assigned_mode, reduction="none"
        )
        * scoreable
    ).sum() / scoreable_count
    return regression + classification, regression, classification
