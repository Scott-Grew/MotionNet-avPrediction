"""Physics baselines in the model's output shape, so they pass through the same
submission and scoring path as the model.
"""
from __future__ import annotations

import math

import torch

from womd import contract, model

FUTURE_HORIZON_SECONDS = contract.FUTURE_HORIZON_SECONDS
TIMESTEP_SECONDS = contract.TIMESTEP_SECONDS


def future_elapsed_seconds(device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
    """Seconds elapsed at each of the future steps, at 10 Hz."""
    return TIMESTEP_SECONDS * torch.arange(
        1, contract.FUTURE_STEPS + 1, device=device, dtype=dtype)


def current_state(
    batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reads each agent's current position, heading (from sin/cos) and velocity
    from the current history step.
    """
    now_row = batch["agent_history"][:, contract.CURRENT_STEP_INDEX]
    heading = torch.atan2(
        now_row[:, contract.AGENT_HEADING_SINE],
        now_row[:, contract.AGENT_HEADING_COSINE],
    )
    return (
        now_row[:, contract.AGENT_POSITION],
        heading,
        now_row[:, contract.AGENT_VELOCITY],
    )


def as_single_mode(
        trajectories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Wraps one trajectory per sample as a single-mode prediction with a dummy
    confidence logit, matching the model's output shape.
    """
    confidence_logits = torch.zeros(
        trajectories.shape[0],
        1,
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
    return trajectories.unsqueeze(1), confidence_logits


def constant_velocity(
        batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extrapolates the current position at the current velocity, unchanged
    over the horizon.
    """
    position, _, velocity = current_state(batch)
    elapsed_seconds = future_elapsed_seconds(position.device, position.dtype)
    displacement = velocity[:, None, :] * elapsed_seconds[None, :, None]
    return as_single_mode(position[:, None, :] + displacement)


def observed_yaw_rate(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Estimates a constant yaw rate from the heading change between the current
    step and the earliest valid history step.
    """
    agent_history = batch["agent_history"]
    step_indices = torch.arange(contract.HISTORY_STEPS,
                                device=agent_history.device)
    # Invalid steps are pushed past the last real index so that
    # min() finds the earliest step that actually has data.
    valid_step_indices = torch.where(
        batch["agent_history_mask"],
        step_indices,
        torch.full_like(step_indices, contract.HISTORY_STEPS),
    )
    earliest_valid_step = valid_step_indices.min(dim=1).values
    earliest_row = agent_history.gather(
        1,
        earliest_valid_step[:, None, None].expand(-1, 1,
                                                  contract.AGENT_FEATURE_DIM),
    ).squeeze(1)

    now_row = agent_history[:, contract.CURRENT_STEP_INDEX]
    # sin(a - b) = sin a cos b - cos a sin b and cos(a - b) = cos a cos b +
    # sin a sin b give the change without the wraparound a plain atan2
    # subtraction would have.
    now_sine = now_row[:, contract.AGENT_HEADING_SINE]
    now_cosine = now_row[:, contract.AGENT_HEADING_COSINE]
    earliest_sine = earliest_row[:, contract.AGENT_HEADING_SINE]
    earliest_cosine = earliest_row[:, contract.AGENT_HEADING_COSINE]
    change_sine = now_sine * earliest_cosine - now_cosine * earliest_sine
    change_cosine = now_cosine * earliest_cosine + now_sine * earliest_sine

    observed_steps = contract.CURRENT_STEP_INDEX - earliest_valid_step
    observed_seconds = observed_steps * TIMESTEP_SECONDS
    observed_seconds = observed_seconds.clamp(min=TIMESTEP_SECONDS)
    return torch.atan2(change_sine, change_cosine) / observed_seconds


def constant_turn_rate_and_velocity(
        batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extrapolates position along a constant-turn-rate, constant-speed arc
    from the observed yaw rate and forward speed.
    """
    position, heading, velocity = current_state(batch)
    heading_direction = torch.stack(
        [torch.cos(heading), torch.sin(heading)], dim=-1)
    forward_speed = (velocity * heading_direction).sum(dim=-1)

    elapsed_seconds = future_elapsed_seconds(position.device, position.dtype)
    turn_angle = observed_yaw_rate(batch)[:, None] * elapsed_seconds[None, :]
    # For turn angle theta the chord is arc * sin(theta / 2) / (theta / 2),
    # which is arc * sinc(theta / 2 pi), and the chord direction bisects
    # the turn from the heading.
    chord_to_arc_ratio = torch.sinc(turn_angle / (2.0 * math.pi))
    chord_length = (forward_speed[:, None] * elapsed_seconds[None, :] *
                    chord_to_arc_ratio)
    chord_direction = heading[:, None] + 0.5 * turn_angle

    displacement = torch.stack(
        [
            chord_length * torch.cos(chord_direction),
            chord_length * torch.sin(chord_direction),
        ],
        dim=-1,
    )
    return as_single_mode(position[:, None, :] + displacement)


def straight_lines_to_most_used_anchors(
        batch: dict[str, torch.Tensor],
        unit_anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight lines to each type's first NUM_PREDICTED_MODES anchor
    endpoints, with uniform confidence.
    """
    predicted_type_index = model.predicted_type_index(batch["agent_history"])
    endpoints = unit_anchors[
        predicted_type_index][:, :contract.NUM_PREDICTED_MODES].to(
            batch["agent_history"].dtype)
    elapsed_seconds = future_elapsed_seconds(endpoints.device, endpoints.dtype)
    horizon_fraction = elapsed_seconds / FUTURE_HORIZON_SECONDS
    trajectories = (endpoints[:, :, None, :] *
                    horizon_fraction[None, None, :, None])
    return trajectories, torch.zeros(
        trajectories.shape[:2],
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
