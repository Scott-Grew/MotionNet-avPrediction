"""Builds each predicted agent's model inputs: its own history and future, its
nearest agents and its nearest map chunks, all in its own frame at the current
step. Nearest means nearest to the straight path the agent would cover in 8 s
at its current velocity, so a fast agent sees far ahead and a slow one sees
around itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numpy import ndarray

from womd import contract, frame_ops
from womd.frame_ops import Frame
from womd.loader import (
    FUTURE,
    HISTORY,
    StagedScenario,
    agent_frame,
    rows_to_frame,
)

# How many agents (the target included) and map chunks each target sees.
# Chosen by Scott 2026-10-19 from their measured exclusions.
AGENT_SLOTS = 48
CHUNK_SLOTS = 256


class SceneChunks(NamedTuple):
    """A scene's map dots grouped into chunks of up to MAP_CHUNK_DOTS dots
    along each polyline, in the scene frame.
    """

    rows: ndarray  # (C, 20, 32)
    dot_mask: ndarray  # (C, 20) False for padding
    signal_history: ndarray  # (C, 11, 9) the chunk's polyline's lights


class TargetInputs(NamedTuple):
    """One predicted agent's inputs in its own frame, or a batch of them with a
    leading axis B. Slot 0 of the agents is the target. The two future fields
    are absent from an inference batch.
    """

    agent_history: ndarray  # (48, 11, 13)
    agent_history_mask: ndarray  # (48, 11) all False for an empty slot
    chunk_rows: ndarray  # (256, 20, 32)
    chunk_dot_mask: ndarray  # (256, 20)
    chunk_signal_history: ndarray  # (256, 11, 9)
    future_positions: ndarray | None = None  # (80, 2)
    future_mask: ndarray | None = None  # (80,)

    def each(self, function: Callable[[ndarray], Any]) -> TargetInputs:
        """The same inputs with function applied to every array."""
        applied = [None if array is None else function(array) for array in self]
        return TargetInputs(*applied)


def scene_chunks(scenario: StagedScenario) -> SceneChunks:
    """Cuts each polyline into chunks of MAP_CHUNK_DOTS dots, the last one
    shorter, and lays them out as one padded row block per chunk.
    """
    lengths = scenario.feature_lengths
    chunk_dots = contract.MAP_CHUNK_DOTS
    chunks_per_polyline = -(-lengths // chunk_dots)
    dot_polyline = scenario.map_dot_polyline_index
    # A dot's place along its polyline is its index minus the index of the
    # polyline's first dot, an exclusive cumulative sum of the lengths.
    first_dot = np.cumsum(lengths) - lengths
    along = np.arange(len(dot_polyline)) - first_dot[dot_polyline]
    first_chunk = np.cumsum(chunks_per_polyline) - chunks_per_polyline
    dot_chunk = first_chunk[dot_polyline] + along // chunk_dots
    dot_slot = along % chunk_dots

    chunk_count = int(chunks_per_polyline.sum())
    row_shape = (chunk_count, chunk_dots, contract.MAP_FEATURE_DIM)
    rows = np.zeros(row_shape, dtype=np.float32)
    rows[dot_chunk, dot_slot] = scenario.map_rows
    dot_mask = np.zeros((chunk_count, chunk_dots), dtype=bool)
    dot_mask[dot_chunk, dot_slot] = True
    chunk_polyline = np.repeat(np.arange(len(lengths)), chunks_per_polyline)
    signals = scenario.polyline_signal_histories[chunk_polyline]
    return SceneChunks(rows, dot_mask, signals)


def segment_distance(points: ndarray, path_end: ndarray) -> ndarray:
    """Distance from each point (..., 2) to the segment from the origin to
    path_end (2,).
    """
    # The nearest point on the segment is t * path_end with t the projection
    # p . e / e . e, clipped to [0, 1].
    length_squared = float(path_end @ path_end)
    if length_squared == 0.0:
        return np.linalg.norm(points, axis=-1)
    fraction = np.clip(points @ path_end / length_squared, 0.0, 1.0)
    nearest = fraction[..., None] * path_end
    return np.linalg.norm(points - nearest, axis=-1)


def map_rows_to_frame(rows: ndarray, frame: Frame) -> ndarray:
    """Re-expresses map rows' position, direction and stop point columns in a
    frame. A dot with no stop point keeps zeros there.
    """
    moved = rows.copy()
    moved[..., contract.MAP_POSITION] = frame_ops.positions_to_frame(
        rows[..., contract.MAP_POSITION], frame
    )
    moved[..., contract.MAP_DIRECTION] = frame_ops.directions_to_frame(
        rows[..., contract.MAP_DIRECTION], frame.heading
    )
    stop_points = rows[..., contract.MAP_STOP_POINT]
    has_stop_point = (stop_points != 0.0).any(axis=-1, keepdims=True)
    moved_stops = frame_ops.positions_to_frame(stop_points, frame)
    moved[..., contract.MAP_STOP_POINT] = np.where(
        has_stop_point, moved_stops, 0.0
    )
    return moved


def padded(array: ndarray, length: int) -> ndarray:
    """Zero-pads the first axis of array to length."""
    widths = [(0, length - len(array))] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, widths)


def last_valid_positions(history: ndarray, valid: ndarray) -> ndarray:
    """Each track's position (T, 2) at its last valid history step, from its
    history rows (T, 11, 13) and their validity (T, 11).
    """
    steps = np.arange(contract.HISTORY_STEPS)
    last_step = np.where(valid, steps, 0).max(axis=1)
    rows = history[np.arange(len(history)), last_step]
    return rows[:, contract.AGENT_POSITION]


def path_end(tracks: ndarray, target: int) -> ndarray:
    """Where the target would be after 8 s at its current velocity, in its
    own frame.
    """
    now = tracks[target, contract.CURRENT_STEP_INDEX]
    return now[contract.AGENT_VELOCITY] * contract.FUTURE_HORIZON_SECONDS


def nearest_agents(scenario: StagedScenario, history: ndarray, target: int):
    """The target, then up to AGENT_SLOTS - 1 other tracks with a valid history
    step, nearest first to the target's 8 s path. history holds every track's
    history rows (T, 11, 13) in the target's frame.
    """
    history_valid = scenario.track_valid[:, HISTORY]
    positions = last_valid_positions(history, history_valid)
    distances = segment_distance(positions, path_end(history, target))
    is_candidate = history_valid.any(axis=1)
    is_candidate[target] = False
    others = np.flatnonzero(is_candidate)
    nearest_first = others[np.argsort(distances[others], kind="stable")]
    return np.concatenate([[target], nearest_first[: AGENT_SLOTS - 1]])


def nearest_chunks(chunks: SceneChunks, scenario, target: int) -> ndarray:
    """Up to CHUNK_SLOTS chunk indices, nearest first by each chunk's nearest
    dot to the target's 8 s straight path at its current velocity. Distances
    do not change with the frame, so they are measured in the scene frame.
    """
    now = scenario.track_rows[target, contract.CURRENT_STEP_INDEX]
    start = now[contract.AGENT_POSITION]
    travel = now[contract.AGENT_VELOCITY] * contract.FUTURE_HORIZON_SECONDS
    scene_positions = chunks.rows[..., contract.MAP_POSITION]
    distances = segment_distance(scene_positions - start, travel)
    nearest_dot = np.where(chunks.dot_mask, distances, np.inf).min(axis=1)
    return np.argsort(nearest_dot, kind="stable")[:CHUNK_SLOTS]


def target_inputs(scenario, chunks: SceneChunks, target: int) -> TargetInputs:
    """One target's inputs, everything moved into its frame at the current
    step and padded to AGENT_SLOTS agents and CHUNK_SLOTS chunks.
    """
    frame = agent_frame(scenario.track_rows[target])
    history = rows_to_frame(scenario.track_rows[:, HISTORY], frame)
    own_track = rows_to_frame(scenario.track_rows[target], frame)
    agents = nearest_agents(scenario, history, target)
    chosen = nearest_chunks(chunks, scenario, target)
    chunk_rows = map_rows_to_frame(chunks.rows[chosen], frame)
    dot_mask = chunks.dot_mask[chosen]
    chunk_rows[~dot_mask] = 0.0
    agent_mask = scenario.track_valid[agents, HISTORY]
    signals = chunks.signal_history[chosen]
    return TargetInputs(
        agent_history=padded(history[agents], AGENT_SLOTS),
        agent_history_mask=padded(agent_mask, AGENT_SLOTS),
        chunk_rows=padded(chunk_rows, CHUNK_SLOTS),
        chunk_dot_mask=padded(dot_mask, CHUNK_SLOTS),
        chunk_signal_history=padded(signals, CHUNK_SLOTS),
        future_positions=own_track[FUTURE, contract.AGENT_POSITION],
        future_mask=scenario.track_valid[target, FUTURE],
    )


def scene_inputs(scenario: StagedScenario, targets: ndarray) -> TargetInputs:
    """Every given target of one scene, stacked on a leading axis."""
    if len(targets) == 0:
        raise ValueError(f"{scenario.scenario_id}: no targets to stack")
    chunks = scene_chunks(scenario)
    per_target = []
    for target in targets:
        per_target.append(target_inputs(scenario, chunks, int(target)))
    return TargetInputs(*map(np.stack, zip(*per_target)))
