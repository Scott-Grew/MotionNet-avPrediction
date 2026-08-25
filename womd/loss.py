import math

import torch

from womd import contract

HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)

def anchor_assigned_mode(unit_anchors, future_positions, future_mask):
    validity = future_mask.to(future_positions.dtype)
    step_positions = torch.arange(
        future_mask.shape[-1], device=future_mask.device, dtype=future_positions.dtype
    )
    last_valid_step = (validity * step_positions).argmax(dim=-1)
    logged_endpoint = future_positions.gather(
        1, last_valid_step[:, None, None].expand(-1, -1, 2)
    ).squeeze(1)
    return (unit_anchors - logged_endpoint.unsqueeze(1)).norm(dim=-1).argmin(dim=1)

def gaussian_negative_log_likelihood(predicted_mean, log_standard_deviation, target):
    standardised_error = (predicted_mean - target.unsqueeze(1)) / log_standard_deviation.exp()
    return (
        log_standard_deviation + 0.5 * standardised_error ** 2 + HALF_LOG_TWO_PI
    ).sum(dim=-1)

def prediction_loss(
    trajectories, log_standard_deviation, confidence_logits,
    future_positions, future_mask, unit_anchors,
):
    step_nll = gaussian_negative_log_likelihood(trajectories, log_standard_deviation, future_positions)
    validity = future_mask.unsqueeze(1).to(step_nll.dtype)
    mode_nll = (step_nll * validity).sum(dim=-1)
    scoreable = future_mask.any(dim=-1).to(step_nll.dtype)
    scoreable_count = scoreable.sum().clamp_min(1.0)
    assigned_mode = anchor_assigned_mode(unit_anchors, future_positions, future_mask)
    regression = (
        mode_nll.gather(1, assigned_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    classification = (
        torch.nn.functional.cross_entropy(confidence_logits, assigned_mode, reduction="none")
        * scoreable
    ).sum() / scoreable_count
    return regression + classification, regression, classification
