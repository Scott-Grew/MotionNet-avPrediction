import math

import numpy as np
import torch

from womd import contract, model

FUTURE_HORIZON_SECONDS = 8.0
TIMESTEP_SECONDS = FUTURE_HORIZON_SECONDS / contract.FUTURE_STEPS


def future_elapsed_seconds(device, dtype):
    return TIMESTEP_SECONDS * torch.arange(
        1, contract.FUTURE_STEPS + 1, device=device, dtype=dtype
    )


def current_state(batch):
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


def as_single_mode(trajectories):
    confidence_logits = torch.zeros(
        trajectories.shape[0],
        1,
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
    return trajectories.unsqueeze(1), confidence_logits


def constant_velocity(batch):
    position, _, velocity = current_state(batch)
    elapsed_seconds = future_elapsed_seconds(
        position.device, position.dtype
    )
    return as_single_mode(
        position[:, None, :]
        + velocity[:, None, :] * elapsed_seconds[None, :, None]
    )


def observed_yaw_rate(batch):
    agent_history = batch["agent_history"]
    step_indices = torch.arange(
        contract.HISTORY_STEPS, device=agent_history.device
    )
    valid_step_indices = torch.where(
        batch["agent_history_mask"],
        step_indices,
        torch.full_like(step_indices, contract.HISTORY_STEPS),
    )
    earliest_valid_step = valid_step_indices.min(dim=1).values
    earliest_row = agent_history.gather(
        1,
        earliest_valid_step[:, None, None].expand(
            -1, 1, contract.AGENT_FEATURE_DIM
        ),
    ).squeeze(1)

    now_row = agent_history[:, contract.CURRENT_STEP_INDEX]
    change_sine = (
        now_row[:, contract.AGENT_HEADING_SINE]
        * earliest_row[:, contract.AGENT_HEADING_COSINE]
        - now_row[:, contract.AGENT_HEADING_COSINE]
        * earliest_row[:, contract.AGENT_HEADING_SINE]
    )
    change_cosine = (
        now_row[:, contract.AGENT_HEADING_COSINE]
        * earliest_row[:, contract.AGENT_HEADING_COSINE]
        + now_row[:, contract.AGENT_HEADING_SINE]
        * earliest_row[:, contract.AGENT_HEADING_SINE]
    )
    observed_seconds = (
        contract.CURRENT_STEP_INDEX - earliest_valid_step
    ) * TIMESTEP_SECONDS
    return torch.atan2(
        change_sine, change_cosine
    ) / observed_seconds.clamp(min=TIMESTEP_SECONDS)


def constant_turn_rate_and_velocity(batch):
    position, heading, velocity = current_state(batch)
    heading_direction = torch.stack(
        [torch.cos(heading), torch.sin(heading)], dim=-1
    )
    forward_speed = (velocity * heading_direction).sum(dim=-1)

    elapsed_seconds = future_elapsed_seconds(
        position.device, position.dtype
    )
    turn_angle = (
        observed_yaw_rate(batch)[:, None] * elapsed_seconds[None, :]
    )
    chord_to_arc_ratio = torch.sinc(turn_angle / (2.0 * math.pi))
    chord_length = (
        forward_speed[:, None]
        * elapsed_seconds[None, :]
        * chord_to_arc_ratio
    )
    chord_direction = heading[:, None] + 0.5 * turn_angle

    displacement = torch.stack(
        [
            chord_length * torch.cos(chord_direction),
            chord_length * torch.sin(chord_direction),
        ],
        dim=-1,
    )
    return as_single_mode(position[:, None, :] + displacement)


def straight_lines_to_most_used_anchors(batch, unit_anchors):
    predicted_type_index = model.predicted_type_index(
        batch["agent_history"]
    )
    endpoints = unit_anchors[predicted_type_index][
        :, : contract.NUM_PREDICTED_MODES
    ].to(batch["agent_history"].dtype)
    elapsed_seconds = future_elapsed_seconds(
        endpoints.device, endpoints.dtype
    )
    trajectories = (
        endpoints[:, :, None, :]
        * (elapsed_seconds / FUTURE_HORIZON_SECONDS)[
            None, None, :, None
        ]
    )
    return trajectories, torch.zeros(
        trajectories.shape[:2],
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
