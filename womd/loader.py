"""Turns a staged scenario into model inputs: the scene once, in the staged
scene frame, plus each predicted agent's own history and future in its own
frame. How each agent sees the scene is computed inside the model, in
womd/view.py. SceneBatch names every array the model reads, and the two row
layouts are drawn at the end of this file.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, List, NamedTuple, Union

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
    type_columns = now_rows[:, contract.AGENT_TYPE]
    is_predicted_type = type_columns[:, : contract.NUM_OBJECT_TYPES].sum(1) > 0
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
    heading_cosine = rows[..., contract.AGENT_HEADING_COSINE]
    heading_sine = rows[..., contract.AGENT_HEADING_SINE]
    rotation_cosine = np.cos(frame.heading)
    rotation_sine = np.sin(frame.heading)
    # A heading h relative to the frame heading r is cos(h - r) = cos h cos r
    # + sin h sin r and sin(h - r) = sin h cos r - cos h sin r.
    moved[..., contract.AGENT_HEADING_COSINE] = (
        heading_cosine * rotation_cosine + heading_sine * rotation_sine
    )
    moved[..., contract.AGENT_HEADING_SINE] = (
        heading_sine * rotation_cosine - heading_cosine * rotation_sine
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


class MapChunks(NamedTuple):
    """The scene's map dots grouped into chunks of up to MAP_CHUNK_DOTS dots
    along each polyline. A chunk's pose is that of its middle dot.
    """

    dot_chunk: ndarray  # (D,) chunk each dot belongs to
    chunk_polyline: ndarray  # (C,) polyline each chunk belongs to
    chunk_pose: ndarray  # (C, 4) x, y, dx, dy in the scene frame


def map_chunks(scenario: StagedScenario) -> MapChunks:
    """Groups each polyline's dots into chunks of MAP_CHUNK_DOTS."""
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
    return MapChunks(
        dot_chunk=dot_chunk.astype(np.int64),
        chunk_polyline=np.repeat(polylines, chunks_per_polyline),
        chunk_pose=chunk_pose.astype(np.float32),
    )


class TargetSample(NamedTuple):
    """One predicted agent: its history and future in its own frame, and
    where it sits in its scene.
    """

    agent_history: ndarray  # (11, 13)
    agent_history_mask: ndarray  # (11,)
    agent_signal_history: ndarray  # (11, 9)
    future_positions: ndarray  # (80, 2)
    future_mask: ndarray  # (80,)
    track_index: int  # the agent's row among its scene's agents
    frame: Frame  # the agent's frame in the scene frame
    track_id: int


class SceneSample(NamedTuple):
    """One scene in the staged scene frame plus one TargetSample per predicted
    agent.
    """

    scenario_id: str
    scene_agent_history: ndarray  # (agents, 11, 13)
    scene_agent_history_mask: ndarray  # (agents, 11)
    scene_agent_signal_history: ndarray  # (agents, 11, 9)
    map_rows: ndarray  # (dots, 32)
    map_dot_chunk: ndarray  # (dots,) which chunk each dot belongs to
    map_chunk_pose: ndarray  # (chunks, 4)
    map_chunk_signal_history: ndarray  # (chunks, 11, 9)
    targets: list[TargetSample]


class SceneView:
    """A staged scenario with its map chunked and each track's lane signal
    found, from which each predicted agent's sample is built.
    """

    def __init__(self, scenario: StagedScenario) -> None:
        self.scenario = scenario
        self.chunks = map_chunks(scenario)
        self.track_signals = lane_signal_histories(scenario)

    def target(self, track_index: int) -> TargetSample:
        """One predicted agent's history and future in its own frame."""
        scenario = self.scenario
        rows = scenario.track_rows[track_index]
        frame = agent_frame(rows)
        agent_track = rows_to_frame(rows, frame)
        return TargetSample(
            agent_history=agent_track[HISTORY],
            agent_history_mask=scenario.track_valid[track_index, HISTORY],
            agent_signal_history=self.track_signals[track_index],
            future_positions=agent_track[FUTURE, contract.AGENT_POSITION],
            future_mask=scenario.track_valid[track_index, FUTURE],
            track_index=track_index,
            frame=frame,
            track_id=int(scenario.track_ids[track_index]),
        )

    def sample(self, track_indices: list[int]) -> SceneSample:
        """The whole scene in the scene frame plus one target per index."""
        scenario = self.scenario
        chunks = self.chunks
        targets = [self.target(index) for index in track_indices]
        signals = scenario.polyline_signal_histories[chunks.chunk_polyline]
        return SceneSample(
            scenario_id=scenario.scenario_id,
            scene_agent_history=scenario.track_rows[:, HISTORY],
            scene_agent_history_mask=scenario.track_valid[:, HISTORY],
            scene_agent_signal_history=self.track_signals,
            map_rows=scenario.map_rows,
            map_dot_chunk=chunks.dot_chunk,
            map_chunk_pose=chunks.chunk_pose,
            map_chunk_signal_history=signals,
            targets=targets,
        )


def scene_sample(scenario: StagedScenario, indices: list[int]) -> SceneSample:
    """One scene sample holding the predicted agents at the given track
    indices.
    """
    return SceneView(scenario).sample(indices)


def pad_and_stack(arrays: Arrays, length: int, dtype: type) -> ndarray:
    """Stacks arrays whose first axis varies, zero-padding each to length."""
    padded = np.zeros((len(arrays), length) + arrays[0].shape[1:], dtype=dtype)
    for row, array in zip(padded, arrays):
        row[: len(array)] = array
    return padded


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
    """One batch from scene_batch. Padding is zero, and the DataLoader hands
    the same fields on as torch tensors.
    """

    scene: SceneArrays
    map: MapArrays
    targets: TargetArrays

    def each(self, function: Callable[[ndarray], Any]) -> SceneBatch:
        """The same batch with function applied to every array."""
        return SceneBatch(
            scene=apply_to_group(self.scene, function),
            map=apply_to_group(self.map, function),
            targets=apply_to_group(self.targets, function),
        )


class BatchBuilder:
    """Joins scene samples into one batch. Each scene's agents, map dots and
    map chunks are zero-padded to the largest scene's, and every scene's
    targets are flattened into one target axis that points back at its scene.
    """

    def __init__(self, samples: list[SceneSample]) -> None:
        self.samples = samples
        self.targets = []
        self.scene_of_target = []
        for index, sample in enumerate(samples):
            self.targets.extend(sample.targets)
            self.scene_of_target.extend([index] * len(sample.targets))
        self.agents = max(len(sample.scene_agent_history) for sample in samples)
        self.dots = max(len(sample.map_rows) for sample in samples)
        self.chunks = max(len(sample.map_chunk_pose) for sample in samples)

    def scene_arrays(self) -> SceneArrays:
        samples = self.samples
        histories = [sample.scene_agent_history for sample in samples]
        masks = [sample.scene_agent_history_mask for sample in samples]
        signals = [sample.scene_agent_signal_history for sample in samples]
        return SceneArrays(
            agent_history=pad_and_stack(histories, self.agents, np.float32),
            agent_history_mask=pad_and_stack(masks, self.agents, bool),
            agent_signal_history=pad_and_stack(
                signals, self.agents, np.float32
            ),
        )

    def map_arrays(self) -> MapArrays:
        samples = self.samples
        rows = [sample.map_rows for sample in samples]
        present = [np.ones(len(sample.map_rows), bool) for sample in samples]
        dot_chunks = [sample.map_dot_chunk for sample in samples]
        poses = [sample.map_chunk_pose for sample in samples]
        signals = [sample.map_chunk_signal_history for sample in samples]
        return MapArrays(
            rows=pad_and_stack(rows, self.dots, np.float32),
            dot_present=pad_and_stack(present, self.dots, bool),
            dot_chunk=pad_and_stack(dot_chunks, self.dots, np.int64),
            chunk_pose=pad_and_stack(poses, self.chunks, np.float32),
            chunk_signal_history=pad_and_stack(
                signals, self.chunks, np.float32
            ),
        )

    def target_arrays(self) -> TargetArrays:
        targets = self.targets
        track_indices = [target.track_index for target in targets]
        histories = [target.agent_history for target in targets]
        masks = [target.agent_history_mask for target in targets]
        signals = [target.agent_signal_history for target in targets]
        futures = [target.future_positions for target in targets]
        future_masks = [target.future_mask for target in targets]
        return TargetArrays(
            scene_index=np.array(self.scene_of_target, dtype=np.int64),
            track_index=np.array(track_indices, dtype=np.int64),
            agent_history=np.stack(histories),
            agent_history_mask=np.stack(masks),
            agent_signal_history=np.stack(signals),
            future_positions=np.stack(futures),
            future_mask=np.stack(future_masks),
        )

    def build(self) -> SceneBatch:
        scene = self.scene_arrays()
        scene_map = self.map_arrays()
        return SceneBatch(scene, scene_map, self.target_arrays())


def scene_batch(samples: list[SceneSample]) -> SceneBatch:
    """One batch from scene samples, padded as BatchBuilder describes."""
    return BatchBuilder(samples).build()


# ------------------------------------------------------------------
# THE TWO ROW LAYOUTS a SceneBatch is built from.
#
# One agent row, per 0.1 s step.
#
#     0   1   2   3   4   5   6   7   8   9  10  11  12
#   +-------+---+---+-------+-------+---------------+---+
#   | x   y |cos|sin|vx  vy |len wid|veh ped cyc oth|sdc|
#   +-------+---+---+-------+-------+---------------+---+
#
# One map row, per metre of a map feature.
#
#     0-1   2-3    4-10    11-14   15    16-27   28-29  30  31
#   +-----+-----+--------+-------+-----+--------+------+---+---+
#   | x y |dx dy| kind   | lane  |speed|boundary| stop | L | R |
#   |     |     | 1-hot  | type  |limit| type   | x  y |   |   |
#   +-----+-----+--------+-------+-----+--------+------+---+---+
#
#   L and R are the marking on the lane's left and right, 0 for none.
# ------------------------------------------------------------------
