import math

import torch

from womd import contract

HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)

def anchor_assigned_mode(selected_unit_anchors, future_positions, future_mask, mode_valid=None):
    validity = future_mask.to(future_positions.dtype)
    step_positions = torch.arange(
        future_mask.shape[-1], device=future_mask.device, dtype=future_positions.dtype
    )
    last_valid_step = (validity * step_positions).argmax(dim=-1)
    logged_endpoint = future_positions.gather(
        1, last_valid_step[:, None, None].expand(-1, -1, 2)
    ).squeeze(1)
    anchor_endpoints = selected_unit_anchors.detach()
    anchor_distances = (anchor_endpoints - logged_endpoint.unsqueeze(1)).norm(dim=-1)
    if mode_valid is not None:
        anchor_distances = anchor_distances.masked_fill(~mode_valid, float("inf"))
    return anchor_distances.argmin(dim=1)

def logged_speed_per_step(future_positions, future_mask):
    now_position = torch.zeros_like(future_positions[:, :1])
    step_positions = torch.cat([now_position, future_positions], dim=1)
    step_distances = (step_positions[:, 1:] - step_positions[:, :-1]).norm(dim=-1)
    now_valid = torch.ones_like(future_mask[:, :1])
    step_validity = torch.cat([now_valid, future_mask], dim=1)
    valid_step_pair = step_validity[:, 1:] & step_validity[:, :-1]
    return step_distances / contract.TIMESTEP_SECONDS, valid_step_pair

def gaussian_negative_log_likelihood(predicted_mean, log_standard_deviation, target):
    standardised_error = (predicted_mean - target.unsqueeze(1)) / log_standard_deviation.exp()
    return (
        log_standard_deviation + 0.5 * standardised_error ** 2 + HALF_LOG_TWO_PI
    ).sum(dim=-1)

def prediction_loss(
    trajectories, heading_cosine_sine, position_log_standard_deviation,
    heading_log_standard_deviation, confidence_logits,
    predicted_speed, speed_log_standard_deviation,
    future_positions, future_headings, future_mask,
    selected_unit_anchors,
    heading_loss_weight, classification_loss_weight, speed_loss_weight,
    mode_valid=None,
):
    step_position_nll = gaussian_negative_log_likelihood(
        trajectories, position_log_standard_deviation, future_positions
    )
    unit_heading_cosine_sine = heading_cosine_sine / heading_cosine_sine.norm(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(heading_cosine_sine.dtype).eps)
    step_heading_nll = gaussian_negative_log_likelihood(
        unit_heading_cosine_sine, heading_log_standard_deviation, future_headings
    )
    validity = future_mask.unsqueeze(1).to(step_position_nll.dtype)
    mode_position_nll = (step_position_nll * validity).sum(dim=-1)
    mode_heading_nll = (step_heading_nll * validity).sum(dim=-1)

    scoreable = future_mask.any(dim=-1).to(step_position_nll.dtype)
    scoreable_count = scoreable.sum().clamp_min(1.0)

    assigned_mode = anchor_assigned_mode(
        selected_unit_anchors, future_positions, future_mask, mode_valid
    )
    regression = (
        mode_position_nll.gather(1, assigned_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    heading = (
        mode_heading_nll.gather(1, assigned_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    classification = (
        torch.nn.functional.cross_entropy(
            confidence_logits, assigned_mode, reduction="none"
        )
        * scoreable
    ).sum() / scoreable_count

    mode_selector = assigned_mode[:, None, None].expand(-1, -1, contract.FUTURE_STEPS)
    assigned_mode_speed = predicted_speed.gather(1, mode_selector).squeeze(1)
    assigned_mode_speed_log_standard_deviation = speed_log_standard_deviation.gather(
        1, mode_selector
    ).squeeze(1)
    logged_speed, valid_speed_step = logged_speed_per_step(future_positions, future_mask)
    valid_speed_step = valid_speed_step.to(assigned_mode_speed.dtype)
    step_speed_nll = (
        assigned_mode_speed_log_standard_deviation
        + 0.5 * (
            (assigned_mode_speed - logged_speed)
            / assigned_mode_speed_log_standard_deviation.exp()
        ) ** 2
        + HALF_LOG_TWO_PI
    )
    speed = (
        (step_speed_nll * valid_speed_step).sum(dim=-1) * scoreable
    ).sum() / scoreable_count

    total = (
        regression
        + heading_loss_weight * heading
        + classification_loss_weight * classification
        + speed_loss_weight * speed
    )
    return total, regression, heading, classification, speed

def neighbour_future_loss(
    neighbour_future_positions, neighbour_log_standard_deviation,
    logged_positions, neighbour_future_mask, neighbour_readable,
):
    standardised_error = (
        neighbour_future_positions - logged_positions
    ) / neighbour_log_standard_deviation.exp()
    step_nll = (
        neighbour_log_standard_deviation + 0.5 * standardised_error ** 2 + HALF_LOG_TWO_PI
    ).sum(dim=-1)
    step_validity = neighbour_future_mask.to(step_nll.dtype)
    neighbour_nll = (step_nll * step_validity).sum(dim=-1)
    scoreable = neighbour_future_mask.any(dim=-1) & neighbour_readable
    scoreable = scoreable.to(step_nll.dtype)
    return (neighbour_nll * scoreable).sum() / scoreable.sum().clamp_min(1.0)
