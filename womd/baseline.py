"""Physics baselines in the model's output shape, so they pass through the same
submission and scoring path as the model.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from womd import contract, model


class CurrentState(NamedTuple):
    """Each agent at the current step, read from its history rows."""

    position: Tensor  # (batch, 2)
    heading: Tensor  # (batch,)
    velocity: Tensor  # (batch, 2)


def future_elapsed_seconds(like: Tensor) -> Tensor:
    """Seconds elapsed at each of the future steps, at 10 Hz."""
    steps = torch.arange(1, contract.FUTURE_STEPS + 1, device=like.device)
    return contract.TIMESTEP_SECONDS * steps.to(like.dtype)


def current_state(history: Tensor) -> CurrentState:
    now = history[:, contract.CURRENT_STEP_INDEX]
    sine = now[:, contract.AGENT_HEADING_SINE]
    cosine = now[:, contract.AGENT_HEADING_COSINE]
    heading = torch.atan2(sine, cosine)
    return CurrentState(
        now[:, contract.AGENT_POSITION],
        heading,
        now[:, contract.AGENT_VELOCITY],
    )


def as_single_mode(trajectories: Tensor) -> tuple[Tensor, Tensor]:
    """Wraps one trajectory per sample as a single-mode prediction with a
    constant confidence logit, matching the model's output shape.
    """
    logits = torch.zeros(
        trajectories.shape[0],
        1,
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
    return trajectories.unsqueeze(1), logits


def constant_velocity(history: Tensor) -> tuple[Tensor, Tensor]:
    """Extrapolates the current position at the current velocity, unchanged
    over the horizon.
    """
    now = current_state(history)
    elapsed = future_elapsed_seconds(now.position)
    displacement = now.velocity[:, None, :] * elapsed[None, :, None]
    return as_single_mode(now.position[:, None, :] + displacement)


def observed_yaw_rate(history: Tensor, mask: Tensor) -> Tensor:
    """A constant yaw rate from the heading change between the current step
    and the earliest valid history step.
    """
    step_indices = torch.arange(contract.HISTORY_STEPS, device=history.device)
    # Invalid steps are pushed past the last real index so that min() finds
    # the earliest step that has data.
    past_the_end = torch.full_like(step_indices, contract.HISTORY_STEPS)
    valid_indices = torch.where(mask, step_indices, past_the_end)
    earliest_step = valid_indices.min(dim=1).values
    selector = earliest_step[:, None, None].expand(
        -1, 1, contract.AGENT_FEATURE_DIM
    )
    earliest = history.gather(1, selector).squeeze(1)
    now = history[:, contract.CURRENT_STEP_INDEX]

    # sin(a - b) = sin a cos b - cos a sin b and cos(a - b) = cos a cos b +
    # sin a sin b give the change without the wraparound of a plain atan2
    # subtraction.
    now_sine = now[:, contract.AGENT_HEADING_SINE]
    now_cosine = now[:, contract.AGENT_HEADING_COSINE]
    earliest_sine = earliest[:, contract.AGENT_HEADING_SINE]
    earliest_cosine = earliest[:, contract.AGENT_HEADING_COSINE]
    change_sine = now_sine * earliest_cosine - now_cosine * earliest_sine
    change_cosine = now_cosine * earliest_cosine + now_sine * earliest_sine

    observed_steps = contract.CURRENT_STEP_INDEX - earliest_step
    observed_seconds = observed_steps * contract.TIMESTEP_SECONDS
    observed_seconds = observed_seconds.clamp(min=contract.TIMESTEP_SECONDS)
    return torch.atan2(change_sine, change_cosine) / observed_seconds


def constant_turn_rate(history: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Extrapolates position along a constant-turn-rate, constant-speed arc
    from the observed yaw rate and forward speed.
    """
    now = current_state(history)
    heading_direction = torch.stack(
        [torch.cos(now.heading), torch.sin(now.heading)], dim=-1
    )
    forward_speed = (now.velocity * heading_direction).sum(dim=-1)
    elapsed = future_elapsed_seconds(now.position)
    turn_angle = observed_yaw_rate(history, mask)[:, None] * elapsed[None, :]
    # For turn angle theta the chord is arc * sin(theta / 2) / (theta / 2),
    # which is arc * sinc(theta / 2 pi), and the chord direction bisects the
    # turn from the heading.
    chord_to_arc = torch.sinc(turn_angle / (2.0 * math.pi))
    chord_length = forward_speed[:, None] * elapsed[None, :] * chord_to_arc
    chord_direction = now.heading[:, None] + 0.5 * turn_angle
    chord_x = chord_length * torch.cos(chord_direction)
    chord_y = chord_length * torch.sin(chord_direction)
    displacement = torch.stack([chord_x, chord_y], dim=-1)
    return as_single_mode(now.position[:, None, :] + displacement)


def anchor_lines(history: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor]:
    """Straight lines to each type's first NUM_PREDICTED_MODES anchor
    endpoints, with uniform confidence.
    """
    type_index = model.predicted_type_index(history)
    endpoints = anchors[type_index][:, : contract.NUM_PREDICTED_MODES].to(
        history.dtype
    )
    elapsed = future_elapsed_seconds(endpoints)
    horizon_fraction = elapsed / contract.FUTURE_HORIZON_SECONDS
    trajectories = (
        endpoints[:, :, None, :] * horizon_fraction[None, None, :, None]
    )
    logits = torch.zeros(
        trajectories.shape[:2],
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
    return trajectories, logits
