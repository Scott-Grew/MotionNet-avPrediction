"""Turns a staged scenario into model inputs, the scene once in the staged
scene frame plus each predicted agent's own view of it. SceneBatch names every
array the model reads, and the two row layouts are drawn at the end of this
file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np
from numpy import ndarray

from womd import contract, frame_ops
from womd.frame_ops import Frame

# The map an agent sees is cropped around it. A faster agent
# covers more road in 8 s, so the crop stretches ahead with speed.
BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5

HISTORY = slice(0, contract.HISTORY_STEPS)
FUTURE = slice(contract.CURRENT_STEP_INDEX + 1, None)
HEADING = slice(contract.AGENT_HEADING_COSINE, contract.AGENT_HEADING_SINE + 1)
LANE_KIND_COLUMN = contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index(
    "lane"
)
SIGNAL_HISTORY_SHAPE = (
    contract.HISTORY_STEPS,
    contract.NUM_TRAFFIC_SIGNAL_STATES,
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


def inside_crop(points: ndarray, radius: float, stretch: float) -> ndarray:
    """Ellipse crop test in agent-frame coordinates. Ahead of the agent the
    radius stretches by stretch; behind it is a circle.
    """
    ahead = points[:, 0]
    sideways = points[:, 1]
    # A point (x, y) is inside when (x / (r s))^2 + (y / r)^2 <= 1 ahead of
    # the agent and (x / r)^2 + (y / r)^2 <= 1 behind it, for radius r and
    # stretch s.
    sideways_term = (sideways / radius) ** 2
    front_term = (ahead / (radius * stretch)) ** 2
    rear_term = (ahead / radius) ** 2
    inside_front = front_term + sideways_term <= 1.0
    inside_rear = rear_term + sideways_term <= 1.0
    return np.where(ahead > 0.0, inside_front, inside_rear)


def eligible_tracks(scenario: StagedScenario, designated_only: bool) -> ndarray:
    """Indices of tracks valid at the current step and of a predicted object
    type, optionally restricted to designated targets.
    """
    now_rows = scenario.track_rows[:, contract.CURRENT_STEP_INDEX]
    predicted_type_columns = now_rows[:, contract.AGENT_TYPE][
        :, : contract.NUM_OBJECT_TYPES
    ]
    is_predicted_type = predicted_type_columns.sum(axis=1) > 0
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
    offsets = (
        lane_rows[None, :, contract.MAP_POSITION]
        - now_rows[:, None, contract.AGENT_POSITION]
    )
    distances = np.linalg.norm(offsets, axis=2)
    faces_the_agent_way = (
        now_rows[:, HEADING] @ lane_rows[:, contract.MAP_DIRECTION].T > 0.0
    )
    none_face_it = ~faces_the_agent_way.any(axis=1, keepdims=True)
    candidates = faces_the_agent_way | none_face_it
    return np.where(candidates, distances, np.inf).argmin(axis=1)


def lane_signal_histories(scenario: StagedScenario) -> ndarray:
    """Each track's traffic-light history, that of the nearest facing lane
    dot's polyline at the current step; all zeros for tracks without one.
    """
    track_rows = scenario.track_rows
    histories = np.zeros(
        (len(track_rows),) + SIGNAL_HISTORY_SHAPE, dtype=np.float32
    )
    lane_dots = np.flatnonzero(scenario.map_rows[:, LANE_KIND_COLUMN] == 1.0)
    assignable = np.flatnonzero(
        scenario.track_valid[:, contract.CURRENT_STEP_INDEX]
    )
    if len(lane_dots) == 0 or len(assignable) == 0:
        return histories
    lane_rows = scenario.map_rows[lane_dots]
    now_rows = track_rows[assignable, contract.CURRENT_STEP_INDEX]
    assigned_dot = facing_lane_dot(lane_rows, now_rows)
    assigned_polyline = scenario.map_dot_polyline_index[lane_dots[assigned_dot]]
    histories[assignable] = scenario.polyline_signal_histories[
        assigned_polyline
    ]
    return histories


def staged_scenario(staged: Mapping[str, ndarray]) -> StagedScenario:
    """Builds a StagedScenario from the arrays of a staged file, deriving
    which feature each map dot belongs to.
    """
    feature_lengths = staged["feature_lengths"]
    dot_polyline = np.repeat(np.arange(len(feature_lengths)), feature_lengths)
    return StagedScenario(
        scenario_id=str(staged["scenario_id"]),
        scene_frame=Frame(
            staged["frame_origin"], float(staged["frame_heading"])
        ),
        track_ids=staged["track_ids"],
        track_rows=staged["track_rows"],
        track_valid=staged["track_valid"],
        is_designated_target=staged["is_designated_target"],
        map_rows=staged["map_rows"],
        feature_lengths=feature_lengths,
        polyline_signal_histories=staged["polyline_signal_histories"],
        map_dot_polyline_index=dot_polyline,
    )


def read_scenario(path: Path | str) -> StagedScenario:
    """Loads one staged .npz scenario and checks its provenance stamp."""
    with np.load(path) as scenario_file:
        staged = {name: scenario_file[name] for name in scenario_file.files}
    if "provenance" in staged:
        hint = "Restage the directory with stage.py."
        contract.check_provenance(staged["provenance"], path, hint)
    dot_count = int(staged["feature_lengths"].sum())
    row_count = len(staged["map_rows"])
    if dot_count != row_count:
        raise ValueError(
            f"{path}: feature_lengths sum to {dot_count}"
            f" over {row_count} map rows"
        )
    return staged_scenario(staged)


def chunk_dots(dot_polyline: ndarray) -> tuple[ndarray, ndarray]:
    """Groups each polyline's dots into chunks of MAP_CHUNK_DOTS, returning the
    chunk each dot falls in and the polyline each chunk belongs to.
    """
    # The index is sorted, so np.unique's first-occurrence index gives each
    # dot's offset within its own polyline's block.
    polylines, first_dot, compact_index, dots_per_polyline = np.unique(
        dot_polyline, return_index=True, return_inverse=True, return_counts=True
    )
    position_within_polyline = (
        np.arange(len(compact_index)) - first_dot[compact_index]
    )
    chunk_dot_count = contract.MAP_CHUNK_DOTS
    chunks_per_polyline = (
        dots_per_polyline + chunk_dot_count - 1
    ) // chunk_dot_count
    # Each polyline's chunks start where the previous polyline's left off,
    # an exclusive cumulative sum.
    first_chunk = np.cumsum(chunks_per_polyline) - chunks_per_polyline
    dot_chunk = (
        first_chunk[compact_index] + position_within_polyline // chunk_dot_count
    )
    return dot_chunk, np.repeat(polylines, chunks_per_polyline)


class SceneTokens(NamedTuple):
    """Where each scene token sits in the scene frame, one per agent and one
    per map chunk.
    """

    agent_present: ndarray  # (A,)
    agent_positions: ndarray  # (A, 2)
    agent_directions: ndarray  # (A, 2)
    dot_chunk: ndarray  # (D,) chunk each dot belongs to
    chunk_polyline: ndarray  # (C,)
    chunk_positions: ndarray  # (C, 2)
    chunk_directions: ndarray  # (C, 2)


def scene_tokens(scenario: StagedScenario) -> SceneTokens:
    """Places every agent at its last valid history step and every map chunk at
    its middle dot.
    """
    history_valid = scenario.track_valid[:, HISTORY]
    # The first True in the reversed mask is the last True in the original
    # order.
    steps_from_end = history_valid[:, ::-1].argmax(axis=1)
    last_valid_step = contract.HISTORY_STEPS - 1 - steps_from_end
    agent_rows = scenario.track_rows[
        np.arange(len(history_valid)), last_valid_step
    ]
    dot_chunk, chunk_polyline = chunk_dots(scenario.map_dot_polyline_index)
    _, first_dot_of_chunk, dots_per_chunk = np.unique(
        dot_chunk, return_index=True, return_counts=True
    )
    chunk_rows = scenario.map_rows[first_dot_of_chunk + dots_per_chunk // 2]
    return SceneTokens(
        agent_present=history_valid.any(axis=1),
        agent_positions=agent_rows[:, contract.AGENT_POSITION],
        agent_directions=agent_rows[:, HEADING],
        dot_chunk=dot_chunk,
        chunk_polyline=chunk_polyline,
        chunk_positions=chunk_rows[:, contract.MAP_POSITION],
        chunk_directions=chunk_rows[:, contract.MAP_DIRECTION],
    )


class TargetSample(NamedTuple):
    """One predicted agent's view of its scene, in that agent's own frame. The
    token fields run over the scene's agents and then its map chunks.
    """

    agent_history: ndarray  # (11, 13)
    agent_history_mask: ndarray  # (11,)
    agent_signal_history: ndarray  # (11, 9)
    future_positions: ndarray  # (80, 2)
    future_mask: ndarray  # (80,)
    agent_token_visible: ndarray  # (agents,)
    chunk_token_visible: ndarray  # (chunks,)
    agent_token_pose: ndarray  # (agents, 4) x, y, cos, sin
    chunk_token_pose: ndarray  # (chunks, 4)
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
    map_chunk_index: ndarray  # (dots,) which chunk each dot belongs to
    map_chunk_signal_history: ndarray  # (chunks, 11, 9)
    targets: list[TargetSample]


def token_poses(points: ndarray, directions: ndarray, frame: Frame) -> ndarray:
    """One pose row per token, (x, y, cosine, sine), in the given frame."""
    frame_positions = frame_ops.positions_to_frame(points, frame)
    frame_directions = frame_ops.directions_to_frame(directions, frame.heading)
    return np.concatenate([frame_positions, frame_directions], axis=1).astype(
        np.float32
    )


class SceneView:
    """A staged scenario with its tokens placed and each track's lane signal
    found, from which each predicted agent's sample is built.
    """

    def __init__(self, scenario: StagedScenario) -> None:
        self.scenario = scenario
        self.tokens = scene_tokens(scenario)
        self.track_signals = lane_signal_histories(scenario)

    def target(self, track_index: int) -> TargetSample:
        """One predicted agent's view of its scene: its own history and future
        in its own frame, which scene tokens fall inside its crop, and every
        token's pose re-expressed in its frame.
        """
        scenario = self.scenario
        tokens = self.tokens
        frame = agent_frame(scenario.track_rows[track_index])
        agent_track = rows_to_frame(scenario.track_rows[track_index], frame)
        now = scenario.track_rows[track_index, contract.CURRENT_STEP_INDEX]
        speed = float(np.linalg.norm(now[contract.AGENT_VELOCITY]))
        dots_in_frame = frame_ops.positions_to_frame(
            scenario.map_rows[:, contract.MAP_POSITION], frame
        )
        dot_inside = inside_crop(
            dots_in_frame, BASE_RADIUS_METRES, 1.0 + STRETCH_GAIN * speed
        )
        # A chunk is visible if any of its dots falls inside the crop.
        chunk_count = len(tokens.chunk_polyline)
        dots_inside_per_chunk = np.bincount(
            tokens.dot_chunk, weights=dot_inside, minlength=chunk_count
        )
        # A target never sees itself as a scene token.
        agent_visible = tokens.agent_present.copy()
        agent_visible[track_index] = False
        return TargetSample(
            agent_history=agent_track[HISTORY],
            agent_history_mask=scenario.track_valid[track_index, HISTORY],
            agent_signal_history=self.track_signals[track_index],
            future_positions=agent_track[FUTURE, contract.AGENT_POSITION],
            future_mask=scenario.track_valid[track_index, FUTURE],
            agent_token_visible=agent_visible,
            chunk_token_visible=dots_inside_per_chunk > 0,
            agent_token_pose=token_poses(
                tokens.agent_positions, tokens.agent_directions, frame
            ),
            chunk_token_pose=token_poses(
                tokens.chunk_positions, tokens.chunk_directions, frame
            ),
            frame=frame,
            track_id=int(scenario.track_ids[track_index]),
        )

    def sample(self, track_indices: list[int]) -> SceneSample:
        """The whole scene in the scene frame plus one target per index."""
        scenario = self.scenario
        tokens = self.tokens
        targets = [self.target(index) for index in track_indices]
        return SceneSample(
            scenario_id=scenario.scenario_id,
            scene_agent_history=scenario.track_rows[:, HISTORY],
            scene_agent_history_mask=scenario.track_valid[:, HISTORY],
            scene_agent_signal_history=self.track_signals,
            map_rows=scenario.map_rows,
            map_chunk_index=tokens.dot_chunk.astype(np.int64),
            map_chunk_signal_history=scenario.polyline_signal_histories[
                tokens.chunk_polyline
            ],
            targets=targets,
        )


def scene_sample(scenario: StagedScenario, indices: list[int]) -> SceneSample:
    """One scene sample holding the predicted agents at the given track
    indices.
    """
    return SceneView(scenario).sample(indices)


def pad_and_stack(arrays: list[ndarray], length: int, dtype: type) -> ndarray:
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
    """Every map dot of every scene in the batch, D dots concatenated, each
    with a slot naming its scene and chunk out of the C chunks of the largest
    scene.
    """

    rows: ndarray  # (D, 32)
    dot_chunk_slot: ndarray  # (D,) scene * C + chunk
    chunk_signal_history: ndarray  # (S, C, 11, 9)


class TargetArrays(NamedTuple):
    """The B predicted agents of the batch, each in its own frame, with the
    tokens of its scene, A agents and then C chunks, seen through its crop.
    The two future fields are absent from an inference batch.
    """

    scene_index: ndarray  # (B,)
    agent_history: ndarray  # (B, 11, 13)
    agent_history_mask: ndarray  # (B, 11)
    agent_signal_history: ndarray  # (B, 11, 9)
    token_visible: ndarray  # (B, A + C)
    token_pose: ndarray  # (B, A + C, 4) x, y, cos, sin
    future_positions: ndarray | None = None  # (B, 80, 2)
    future_mask: ndarray | None = None  # (B, 80)


Group = Union[SceneArrays, MapArrays, TargetArrays]


def apply_to_group(group: Group, function: Callable[[ndarray], Any]) -> Group:
    applied = {}
    for name, array in group._asdict().items():
        applied[name] = None if array is None else function(array)
    return type(group)(**applied)


class SceneBatch(NamedTuple):
    """One batch from scene_batch. Padding is zero, and the DataLoader hands
    the same fields on as torch tensors.
    """

    scene: SceneArrays
    map: MapArrays
    targets: TargetArrays

    def each(self, function: Callable[[ndarray], Any]) -> SceneBatch:
        """The same batch with function applied to every array, by name."""
        return SceneBatch(
            scene=apply_to_group(self.scene, function),
            map=apply_to_group(self.map, function),
            targets=apply_to_group(self.targets, function),
        )


class BatchBuilder:
    """Joins scene samples into one batch. Scene agents and map chunks are
    zero-padded to the largest scene, map dots are concatenated with a slot
    saying whose they are, and every scene's targets are flattened into one
    target axis that points back at its scene.
    """

    def __init__(self, samples: list[SceneSample]) -> None:
        self.samples = samples
        self.targets = [
            target for sample in samples for target in sample.targets
        ]
        self.agents = max(len(sample.scene_agent_history) for sample in samples)
        self.chunks = max(
            len(sample.map_chunk_signal_history) for sample in samples
        )

    def scene_field(self, name: str, width: int, dtype: type) -> ndarray:
        arrays = [getattr(sample, name) for sample in self.samples]
        return pad_and_stack(arrays, width, dtype)

    def target_field(self, name: str) -> ndarray:
        return np.stack([getattr(target, name) for target in self.targets])

    def token_field(self, names: tuple[str, str], dtype: type) -> ndarray:
        """One per-token entry across every target, in token order, the padded
        scene agents and then the padded map chunks.
        """
        agent_name, chunk_name = names
        agent_arrays = [getattr(target, agent_name) for target in self.targets]
        chunk_arrays = [getattr(target, chunk_name) for target in self.targets]
        agent_part = pad_and_stack(agent_arrays, self.agents, dtype)
        chunk_part = pad_and_stack(chunk_arrays, self.chunks, dtype)
        return np.concatenate([agent_part, chunk_part], axis=1)

    def scene_arrays(self) -> SceneArrays:
        return SceneArrays(
            agent_history=self.scene_field(
                "scene_agent_history", self.agents, np.float32
            ),
            agent_history_mask=self.scene_field(
                "scene_agent_history_mask", self.agents, bool
            ),
            agent_signal_history=self.scene_field(
                "scene_agent_signal_history", self.agents, np.float32
            ),
        )

    def map_arrays(self) -> MapArrays:
        # A dot's slot is scene_index * chunks + its chunk index, which is
        # unique across the batch.
        dot_slots = []
        for index, sample in enumerate(self.samples):
            dot_slots.append(sample.map_chunk_index + index * self.chunks)
        rows = [sample.map_rows for sample in self.samples]
        return MapArrays(
            rows=np.concatenate(rows, dtype=np.float32),
            dot_chunk_slot=np.concatenate(dot_slots, dtype=np.int64),
            chunk_signal_history=self.scene_field(
                "map_chunk_signal_history", self.chunks, np.float32
            ),
        )

    def target_arrays(self) -> TargetArrays:
        scene_of_target = []
        for index, sample in enumerate(self.samples):
            scene_of_target.extend([index] * len(sample.targets))
        visible_names = ("agent_token_visible", "chunk_token_visible")
        pose_names = ("agent_token_pose", "chunk_token_pose")
        return TargetArrays(
            scene_index=np.array(scene_of_target, dtype=np.int64),
            agent_history=self.target_field("agent_history"),
            agent_history_mask=self.target_field("agent_history_mask"),
            agent_signal_history=self.target_field("agent_signal_history"),
            token_visible=self.token_field(visible_names, bool),
            token_pose=self.token_field(pose_names, np.float32),
            future_positions=self.target_field("future_positions"),
            future_mask=self.target_field("future_mask"),
        )

    def build(self) -> SceneBatch:
        return SceneBatch(
            self.scene_arrays(), self.map_arrays(), self.target_arrays()
        )


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
