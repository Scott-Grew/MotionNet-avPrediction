"""Turns staged scenarios into model inputs: each scene once, in the staged
scene frame, plus each predicted agent's own history and future in its own
frame. How each agent sees the scene is computed inside the model, in
womd/view.py. SceneBatch names every array the model reads.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, List, NamedTuple, Tuple, Union

import numpy as np
from numpy import ndarray

from womd import contract, frame_ops
from womd.contract import SIGNAL_HISTORY_SHAPE, StagedFile
from womd.frame_ops import Frame

# typing.List, since the scorer container's Python 3.8 cannot
# subscript list at import time.
Arrays = List[ndarray]

HISTORY = slice(0, contract.HISTORY_STEPS)
FUTURE = slice(contract.CURRENT_STEP_INDEX + 1, None)
HEADING = slice(contract.AGENT_HEADING_COSINE, contract.AGENT_HEADING_SINE + 1)
LANE_KIND_COLUMN = contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index(
    "lane"
)


class StagedScenario(NamedTuple):
    """One staged scenario in the scene frame, with T tracks, F map features
    and D map dots. The last field is derived on load.
    """

    scenario_id: str
    scene_frame: Frame  # the scene frame in the world
    track_ids: ndarray  # (T,)
    track_rows: ndarray  # (T, 91, 13)
    track_valid: ndarray  # (T, 91)
    is_designated_target: ndarray  # (T,)
    map_rows: ndarray  # (D, 32)
    feature_lengths: ndarray  # (F,) dots per feature
    polyline_signal_histories: ndarray  # (F, 11, 9)
    map_dot_polyline_index: ndarray  # (D,) feature each dot belongs to


def eligible_tracks(scenario: StagedScenario, designated_only: bool) -> ndarray:
    """Indices of tracks valid at the current step and of a predicted object
    type, optionally restricted to designated targets.
    """
    now_rows = scenario.track_rows[:, contract.CURRENT_STEP_INDEX]
    # The predicted types are the first NUM_OBJECT_TYPES columns of the type
    # one-hot; the column after them is TYPE_OTHER.
    type_one_hot = now_rows[:, contract.AGENT_TYPE]
    predicted_columns = type_one_hot[:, : contract.NUM_OBJECT_TYPES]
    is_predicted_type = predicted_columns.any(axis=1)
    now_valid = scenario.track_valid[:, contract.CURRENT_STEP_INDEX]
    is_eligible = now_valid & is_predicted_type
    if designated_only:
        is_eligible = is_eligible & scenario.is_designated_target
    return np.flatnonzero(is_eligible)


def agent_frame(rows: ndarray) -> Frame:
    """A track's own frame at the current step, from its rows (91, 13) in the
    scene frame.
    """
    now = rows[contract.CURRENT_STEP_INDEX]
    sine = now[contract.AGENT_HEADING_SINE]
    cosine = now[contract.AGENT_HEADING_COSINE]
    return Frame(now[contract.AGENT_POSITION], np.arctan2(sine, cosine))


def rows_to_frame(rows: ndarray, frame: Frame) -> ndarray:
    """Re-expresses agent rows' position, velocity and heading columns in
    another agent's frame.
    """
    moved = rows.copy()
    moved[..., contract.AGENT_POSITION] = frame_ops.positions_to_frame(
        rows[..., contract.AGENT_POSITION], frame
    )
    moved[..., contract.AGENT_VELOCITY] = frame_ops.directions_to_frame(
        rows[..., contract.AGENT_VELOCITY], frame.heading
    )
    # A heading's (cos, sin) is a unit direction, so it rotates like one.
    moved[..., HEADING] = frame_ops.directions_to_frame(
        rows[..., HEADING], frame.heading
    )
    return moved


def facing_lane_dot(lane_rows: ndarray, now_rows: ndarray) -> ndarray:
    """Nearest lane dot that runs each agent's way, so an agent is not handed
    the oncoming lane's light; any dot if none runs its way.
    """
    lane_positions = lane_rows[None, :, contract.MAP_POSITION]
    agent_positions = now_rows[:, None, contract.AGENT_POSITION]
    distances = np.linalg.norm(lane_positions - agent_positions, axis=2)
    lane_directions = lane_rows[:, contract.MAP_DIRECTION]
    faces_the_agent_way = now_rows[:, HEADING] @ lane_directions.T > 0.0
    none_face_it = ~faces_the_agent_way.any(axis=1, keepdims=True)
    candidates = faces_the_agent_way | none_face_it
    return np.where(candidates, distances, np.inf).argmin(axis=1)


def lane_signal_histories(scenario: StagedScenario) -> ndarray:
    """Each track's traffic-light history, that of the nearest facing lane
    dot's polyline at the current step; all zeros for tracks without one.
    """
    track_rows = scenario.track_rows
    shape = (len(track_rows),) + SIGNAL_HISTORY_SHAPE
    histories = np.zeros(shape, dtype=np.float32)
    lane_dots = np.flatnonzero(scenario.map_rows[:, LANE_KIND_COLUMN] == 1.0)
    now_valid = scenario.track_valid[:, contract.CURRENT_STEP_INDEX]
    assignable = np.flatnonzero(now_valid)
    if len(lane_dots) == 0 or len(assignable) == 0:
        return histories
    lane_rows = scenario.map_rows[lane_dots]
    now_rows = track_rows[assignable, contract.CURRENT_STEP_INDEX]
    assigned_dot = facing_lane_dot(lane_rows, now_rows)
    assigned_polyline = scenario.map_dot_polyline_index[lane_dots[assigned_dot]]
    signals = scenario.polyline_signal_histories[assigned_polyline]
    histories[assignable] = signals
    return histories


def staged_scenario(staged: StagedFile) -> StagedScenario:
    """The staged arrays the model reads, with the feature each map dot
    belongs to derived from the feature lengths.
    """
    feature_lengths = staged.feature_lengths
    dot_polyline = np.repeat(np.arange(len(feature_lengths)), feature_lengths)
    scene_frame = Frame(staged.frame_origin, float(staged.frame_heading))
    return StagedScenario(
        scenario_id=str(staged.scenario_id),
        scene_frame=scene_frame,
        track_ids=staged.track_ids,
        track_rows=staged.track_rows,
        track_valid=staged.track_valid,
        is_designated_target=staged.is_designated_target,
        map_rows=staged.map_rows,
        feature_lengths=feature_lengths,
        polyline_signal_histories=staged.polyline_signal_histories,
        map_dot_polyline_index=dot_polyline,
    )


def read_scenario(path: Path | str) -> StagedScenario:
    return staged_scenario(StagedFile.load(path))


Chunks = Tuple[ndarray, ndarray, ndarray]


def map_chunks(scenario: StagedScenario) -> Chunks:
    """Groups each polyline's dots into chunks of up to MAP_CHUNK_DOTS dots and
    returns the chunk of each dot (D,), the polyline of each chunk (C,) and
    each chunk's pose (C, 4), x, y, dx, dy of its middle dot.
    """
    dot_polyline = scenario.map_dot_polyline_index
    # The index is sorted, so np.unique's first-occurrence index gives each
    # dot's offset within its own polyline's block.
    polylines, first_dot, compact_index, dots_per_polyline = np.unique(
        dot_polyline, return_index=True, return_inverse=True, return_counts=True
    )
    dot_numbers = np.arange(len(compact_index))
    position_within_polyline = dot_numbers - first_dot[compact_index]
    chunk_dot_count = contract.MAP_CHUNK_DOTS
    chunks_per_polyline = -(-dots_per_polyline // chunk_dot_count)
    # Each polyline's chunks start where the previous polyline's left off,
    # an exclusive cumulative sum.
    first_chunk = np.cumsum(chunks_per_polyline) - chunks_per_polyline
    chunk_within_polyline = position_within_polyline // chunk_dot_count
    dot_chunk = first_chunk[compact_index] + chunk_within_polyline
    _, first_dot_of_chunk, dots_per_chunk = np.unique(
        dot_chunk, return_index=True, return_counts=True
    )
    middle_rows = scenario.map_rows[first_dot_of_chunk + dots_per_chunk // 2]
    middle_positions = middle_rows[:, contract.MAP_POSITION]
    middle_directions = middle_rows[:, contract.MAP_DIRECTION]
    chunk_pose = np.concatenate([middle_positions, middle_directions], axis=1)
    chunk_polyline = np.repeat(polylines, chunks_per_polyline)
    return dot_chunk.astype(np.int64), chunk_polyline, chunk_pose


def frame_pose(frame: Frame) -> ndarray:
    """A frame as (x, y, cos, sin) of its origin and heading."""
    heading = np.float32(frame.heading)
    direction = [np.cos(heading), np.sin(heading)]
    return np.concatenate([frame.origin, direction]).astype(np.float32)


def designated_tracks(scenario: StagedScenario) -> ndarray:
    """The tracks Waymo designated for scoring; raises if any of them cannot
    be predicted.
    """
    designated_count = int(scenario.is_designated_target.sum())
    track_indices = eligible_tracks(scenario, designated_only=True)
    if len(track_indices) != designated_count:
        raise ValueError(
            f"{scenario.scenario_id} designates {designated_count} targets,"
            f" {len(track_indices)} can be predicted"
        )
    return track_indices


class SceneArrays(NamedTuple):
    """Every agent of every scene in the batch, in the staged scene frame,
    zero-padded to the A agents of the largest of the S scenes.
    """

    agent_history: ndarray  # (S, A, 11, 13)
    agent_history_mask: ndarray  # (S, A, 11)
    agent_signal_history: ndarray  # (S, A, 11, 9)


class MapArrays(NamedTuple):
    """Every scene's map dots and chunks, zero-padded to the D dots and C
    chunks of the largest scene.
    """

    rows: ndarray  # (S, D, 32)
    dot_present: ndarray  # (S, D) False for padding
    dot_chunk: ndarray  # (S, D) chunk each dot belongs to
    chunk_pose: ndarray  # (S, C, 4) x, y, dx, dy of the middle dot
    chunk_signal_history: ndarray  # (S, C, 11, 9)


class TargetArrays(NamedTuple):
    """The B predicted agents of the batch, each in its own frame. The two
    future fields are absent from an inference batch.
    """

    scene_index: ndarray  # (B,) the target's scene in the batch
    track_index: ndarray  # (B,) the target's row among its scene's agents
    frame_pose: ndarray  # (B, 4) x, y, cos, sin of its frame, scene frame
    speed: ndarray  # (B,) metres per second
    agent_history: ndarray  # (B, 11, 13)
    agent_history_mask: ndarray  # (B, 11)
    agent_signal_history: ndarray  # (B, 11, 9)
    future_positions: ndarray | None = None  # (B, 80, 2)
    future_mask: ndarray | None = None  # (B, 80)


Group = Union[SceneArrays, MapArrays, TargetArrays]


def apply_to_group(group: Group, function: Callable[[ndarray], Any]) -> Group:
    applied = [None if array is None else function(array) for array in group]
    return type(group)(*applied)


class SceneBatch(NamedTuple):
    """Scenes and their predicted agents, as one_scene builds them and
    joined_batch stacks them. The DataLoader hands the same fields on as
    torch tensors.
    """

    scene: SceneArrays
    map: MapArrays
    targets: TargetArrays

    def each(self, function: Callable[[ndarray], Any]) -> SceneBatch:
        return SceneBatch(
            scene=apply_to_group(self.scene, function),
            map=apply_to_group(self.map, function),
            targets=apply_to_group(self.targets, function),
        )


def target_arrays(scenario: StagedScenario, tracks: ndarray, signals: ndarray):
    """The predicted agents at track rows tracks, each in its own frame, with
    their traffic-light histories from signals (T, 11, 9).
    """
    frame_poses = []
    speeds = []
    agent_tracks = []
    for track_index in tracks:
        rows = scenario.track_rows[track_index]
        frame = agent_frame(rows)
        now = rows[contract.CURRENT_STEP_INDEX]
        frame_poses.append(frame_pose(frame))
        speeds.append(np.linalg.norm(now[contract.AGENT_VELOCITY]))
        agent_tracks.append(rows_to_frame(rows, frame))
    agent_track = np.stack(agent_tracks)
    valid = scenario.track_valid[tracks]
    return TargetArrays(
        scene_index=np.zeros(len(tracks), dtype=np.int64),
        track_index=np.asarray(tracks, dtype=np.int64),
        frame_pose=np.stack(frame_poses),
        speed=np.array(speeds, dtype=np.float32),
        agent_history=agent_track[:, HISTORY],
        agent_history_mask=valid[:, HISTORY],
        agent_signal_history=signals[tracks],
        future_positions=agent_track[:, FUTURE, contract.AGENT_POSITION],
        future_mask=valid[:, FUTURE],
    )


def one_scene(scenario: StagedScenario, tracks: ndarray) -> SceneBatch:
    track_signals = lane_signal_histories(scenario)
    scene = SceneArrays(
        agent_history=scenario.track_rows[None, :, HISTORY],
        agent_history_mask=scenario.track_valid[None, :, HISTORY],
        agent_signal_history=track_signals[None],
    )
    dot_chunk, chunk_polyline, chunk_pose = map_chunks(scenario)
    chunk_signals = scenario.polyline_signal_histories[chunk_polyline]
    dot_count = len(scenario.map_rows)
    scene_map = MapArrays(
        rows=scenario.map_rows[None],
        dot_present=np.ones((1, dot_count), dtype=bool),
        dot_chunk=dot_chunk[None],
        chunk_pose=chunk_pose[None].astype(np.float32),
        chunk_signal_history=chunk_signals[None],
    )
    targets = target_arrays(scenario, tracks, track_signals)
    return SceneBatch(scene, scene_map, targets)


def padded_concatenate(arrays: Arrays) -> ndarray:
    """Concatenates arrays along the first axis, zero-padding each one's
    second axis to the longest.
    """
    length = max(array.shape[1] for array in arrays)
    padded = []
    for array in arrays:
        widths = [(0, 0)] * array.ndim
        widths[1] = (0, length - array.shape[1])
        padded.append(np.pad(array, widths))
    return np.concatenate(padded)


def joined_batch(batches: list[SceneBatch]) -> SceneBatch:
    """Stacks one-scene batches into one batch: scenes and maps padded to the
    largest, targets concatenated and pointed at their scene.
    """
    scene_columns = zip(*(batch.scene for batch in batches))
    map_columns = zip(*(batch.map for batch in batches))
    target_columns = zip(*(batch.targets for batch in batches))
    scene = SceneArrays(*map(padded_concatenate, scene_columns))
    scene_map = MapArrays(*map(padded_concatenate, map_columns))
    targets = TargetArrays(*map(np.concatenate, target_columns))
    target_counts = [len(batch.targets.track_index) for batch in batches]
    scene_index = np.repeat(np.arange(len(batches)), target_counts)
    targets = targets._replace(scene_index=scene_index)
    return SceneBatch(scene, scene_map, targets)
