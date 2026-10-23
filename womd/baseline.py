"""Physics baselines in the model's output shape, so they pass through the same
submission and scoring path as the model.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from womd import layout

# Trajectories (B, modes, 80, 2) and confidence logits (B, modes), the
# model's output shape.
Modes = tuple[Tensor, Tensor]


class CurrentState(NamedTuple):
    """Each agent at the current step, read from its history rows."""

    position: Tensor  # (batch, 2)
    heading: Tensor  # (batch,)
    velocity: Tensor  # (batch, 2)


def future_elapsed_seconds(like: Tensor) -> Tensor:
    """Seconds elapsed at each of the future steps, at 10 Hz."""
    steps = torch.arange(1, layout.FUTURE_STEPS + 1, device=like.device)
    return layout.TIMESTEP_SECONDS * steps.to(like.dtype)


def current_state(history: Tensor) -> CurrentState:
    now = history[:, layout.CURRENT_STEP_INDEX]
    sine = now[:, layout.AGENT_HEADING_SINE]
    cosine = now[:, layout.AGENT_HEADING_COSINE]
    heading = torch.atan2(sine, cosine)
    return CurrentState(
        now[:, layout.AGENT_POSITION], heading, now[:, layout.AGENT_VELOCITY]
    )


def as_single_mode(trajectories: Tensor) -> Modes:
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


def constant_velocity(history: Tensor) -> Modes:
    """Extrapolates the current position at the current velocity, unchanged
    over the horizon.
    """
    now = current_state(history)
    elapsed = future_elapsed_seconds(now.position)
    displacement = now.velocity[:, None, :] * elapsed[None, :, None]
    return as_single_mode(now.position[:, None, :] + displacement)
