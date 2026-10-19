"""Each predicted agent's view of its scene, computed in torch inside the
model: every scene token's pose re-expressed in the agent's own frame, and
which tokens fall inside the agent's crop. Being part of the model, it is
part of the exported ONNX graph too.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from womd import contract
from womd.loader import SceneArrays, SceneBatch

# The map an agent sees is cropped around it. A faster agent
# covers more road in 8 s, so the crop stretches ahead with speed.
BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5

HEADING = slice(contract.AGENT_HEADING_COSINE, contract.AGENT_HEADING_SINE + 1)


class TargetFrames(NamedTuple):
    """Each target's own frame at the current step, in the scene frame."""

    origin: Tensor  # (B, 2) metres
    cosine: Tensor  # (B,) of the heading
    sine: Tensor  # (B,)
    speed: Tensor  # (B,) metres per second


class TargetView(NamedTuple):
    """Each target's view of its scene's tokens, agents then map chunks."""

    pose: Tensor  # (B, A + C, 4) x, y, cos, sin in the target's frame
    visible: Tensor  # (B, A + C) inside the crop and not the target itself


def target_frames(batch: SceneBatch) -> TargetFrames:
    """Each target's frame and speed, as the loader wrote them."""
    targets = batch.targets
    pose = targets.frame_pose
    return TargetFrames(pose[:, :2], pose[:, 2], pose[:, 3], targets.speed)


def rotate_into(vectors: Tensor, frames: TargetFrames) -> Tensor:
    """Vectors (B, N, 2) in the scene frame, rotated into each target's frame
    by x' = x cos h + y sin h and y' = y cos h - x sin h for heading h.
    """
    cosine = frames.cosine[:, None]
    sine = frames.sine[:, None]
    x = vectors[..., 0]
    y = vectors[..., 1]
    return torch.stack([x * cosine + y * sine, y * cosine - x * sine], dim=-1)


def positions_into(points: Tensor, frames: TargetFrames) -> Tensor:
    centred = points - frames.origin[:, None, :]
    return rotate_into(centred, frames)


def inside_crop(points: Tensor, stretch: Tensor) -> Tensor:
    """Ellipse crop test on points (B, N, 2) in each target's frame. Ahead of
    the agent the radius stretches by stretch (B, 1); behind it is a circle.
    """
    radius = BASE_RADIUS_METRES
    ahead = points[..., 0]
    sideways = points[..., 1]
    # A point (x, y) is inside when (x / (r s))^2 + (y / r)^2 <= 1 ahead of
    # the agent and (x / r)^2 + (y / r)^2 <= 1 behind it, for radius r and
    # stretch s.
    sideways_term = (sideways / radius) ** 2
    front_term = (ahead / (radius * stretch)) ** 2
    rear_term = (ahead / radius) ** 2
    inside_front = front_term + sideways_term <= 1.0
    inside_rear = rear_term + sideways_term <= 1.0
    is_ahead = ahead > 0.0
    return (is_ahead & inside_front) | (~is_ahead & inside_rear)


def agent_poses(scene: SceneArrays) -> Tensor:
    """Each scene agent's pose (S, A, 4) at its last valid history step: x, y
    and its heading's cosine and sine, in the scene frame.
    """
    history = scene.agent_history
    mask = scene.agent_history_mask
    steps = torch.arange(contract.HISTORY_STEPS, device=history.device)
    # Valid steps hold their own step number and the rest zero, so the
    # argmax is the last valid step; it exports where amax does not.
    last_valid = torch.where(mask, steps, 0).argmax(dim=-1)
    selector = last_valid[..., None, None].expand(-1, -1, 1, history.shape[-1])
    rows = history.gather(2, selector).squeeze(2)
    positions = rows[..., contract.AGENT_POSITION]
    return torch.cat([positions, rows[..., HEADING]], dim=-1)


def token_poses(batch: SceneBatch, frames: TargetFrames) -> Tensor:
    """Every token of each target's scene, re-expressed in its frame."""
    scene_poses = torch.cat([agent_poses(batch.scene), batch.map.chunk_pose], 1)
    poses = scene_poses.index_select(0, batch.targets.scene_index)
    positions = positions_into(poses[..., :2], frames)
    directions = rotate_into(poses[..., 2:], frames)
    return torch.cat([positions, directions], dim=-1)


def agents_visible(batch: SceneBatch) -> Tensor:
    """Every agent with a valid history step, except the target itself."""
    targets = batch.targets
    scene_present = batch.scene.agent_history_mask.any(dim=-1)
    present = scene_present.index_select(0, targets.scene_index)
    agents = torch.arange(present.shape[1], device=present.device)
    is_self = agents[None, :] == targets.track_index[:, None]
    return present & ~is_self


def chunks_visible(batch: SceneBatch, frames: TargetFrames) -> Tensor:
    """A map chunk is visible to a target when any of its dots falls inside
    the target's crop; each target is checked against its own scene's dots.
    """
    scene_map = batch.map
    scene_index = batch.targets.scene_index
    scene_dots = scene_map.rows[:, :, contract.MAP_POSITION]
    dot_positions = scene_dots.index_select(0, scene_index)
    local = positions_into(dot_positions, frames)
    stretch = 1.0 + STRETCH_GAIN * frames.speed[:, None]
    dot_present = scene_map.dot_present.index_select(0, scene_index)
    inside = inside_crop(local, stretch) & dot_present
    chunk_count = scene_map.chunk_pose.shape[1]
    counts = local.new_zeros(len(scene_index), chunk_count)
    dot_chunk = scene_map.dot_chunk.index_select(0, scene_index)
    counts = counts.scatter_add(1, dot_chunk, inside.to(counts.dtype))
    return counts > 0


def target_view(batch: SceneBatch) -> TargetView:
    frames = target_frames(batch)
    pose = token_poses(batch, frames)
    agents = agents_visible(batch)
    chunks = chunks_visible(batch, frames)
    return TargetView(pose, torch.cat([agents, chunks], dim=1))
