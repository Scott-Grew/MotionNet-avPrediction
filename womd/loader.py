"""Turns a staged scenario into model inputs, the scene once in the staged
scene frame plus each predicted agent's own view of it. SceneBatch names every
array the model reads, and the two row layouts are drawn at the end of this
file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np

from womd import contract, frame_ops

# The map an agent sees is cropped around it. A faster agent
# covers more road in 8 s, so the crop stretches ahead with speed.
BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5


def inside_crop(agent_frame_points: np.ndarray, *, base_radius: float,
                forward_stretch: float) -> np.ndarray:
    """Ellipse crop test in agent-frame coordinates. Ahead of the agent the
    radius stretches by forward_stretch; behind it is a circle.
    """
    ahead = agent_frame_points[:, 0]
    sideways = agent_frame_points[:, 1]
    # A point (x, y) is inside when (x / (r s))^2 + (y / r)^2 <= 1 ahead of
    # the agent and (x / r)^2 + (y / r)^2 <= 1 behind it, for radius r and
    # stretch s.
    sideways_term = (sideways / base_radius)**2
    front_term = (ahead / (base_radius * forward_stretch))**2
    rear_term = (ahead / base_radius)**2
    inside_front_ellipse = front_term + sideways_term <= 1.0
    inside_rear_circle = rear_term + sideways_term <= 1.0
    return np.where(ahead > 0.0, inside_front_ellipse, inside_rear_circle)


def eligible_track_indices(track_rows: np.ndarray, track_valid: np.ndarray,
                           is_designated_target: np.ndarray, *,
                           designated_targets_only: bool) -> np.ndarray:
    """Indices of tracks valid at the current step and of a predicted object
    type, optionally restricted to designated targets.
    """
    now_valid = track_valid[:, contract.CURRENT_STEP_INDEX]
    predicted_type = (
        track_rows[:, contract.CURRENT_STEP_INDEX,
                   contract.AGENT_TYPE][:, :contract.NUM_OBJECT_TYPES].sum(
                       axis=1) > 0)
    is_eligible = now_valid & predicted_type
    if designated_targets_only:
        is_eligible = is_eligible & is_designated_target
    return np.flatnonzero(is_eligible)


def sample_frame(track_rows: np.ndarray,
                 track_index: int) -> tuple[np.ndarray, float]:
    """A track's agent frame at the current step, its position and heading."""
    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    origin = now_row[contract.AGENT_POSITION]
    heading = np.arctan2(
        now_row[contract.AGENT_HEADING_SINE],
        now_row[contract.AGENT_HEADING_COSINE],
    )
    return origin, heading


def track_rows_to_agent_frame(track_rows: np.ndarray, origin: np.ndarray,
                              heading: float) -> np.ndarray:
    """Re-expresses a track's per-step position, velocity and heading columns in
    another agent's frame.
    """
    agent_frame_rows = track_rows.copy()
    agent_frame_rows[...,
                     contract.AGENT_POSITION] = (frame_ops.positions_to_frame(
                         track_rows[..., contract.AGENT_POSITION], origin,
                         heading))
    agent_frame_rows[...,
                     contract.AGENT_VELOCITY] = (frame_ops.directions_to_frame(
                         track_rows[..., contract.AGENT_VELOCITY], heading))
    heading_cosine = track_rows[..., contract.AGENT_HEADING_COSINE]
    heading_sine = track_rows[..., contract.AGENT_HEADING_SINE]
    rotation_cosine, rotation_sine = np.cos(heading), np.sin(heading)
    # A heading h relative to the frame heading r is cos(h - r) = cos h cos r
    # + sin h sin r and sin(h - r) = sin h cos r - cos h sin r.
    agent_frame_rows[..., contract.AGENT_HEADING_COSINE] = (
        heading_cosine * rotation_cosine + heading_sine * rotation_sine)
    agent_frame_rows[..., contract.AGENT_HEADING_SINE] = (
        heading_sine * rotation_cosine - heading_cosine * rotation_sine)
    return agent_frame_rows


def nearest_same_direction_lane_dot(
        lane_dot_rows: np.ndarray, agent_distances: np.ndarray,
        agent_heading_cosine_sine: np.ndarray) -> np.ndarray:
    """Nearest lane dot that runs the agent's way, so an agent is not handed the
    oncoming lane's light; any dot if none runs its way.
    """
    faces_the_agent_way = (
        agent_heading_cosine_sine @ lane_dot_rows[:, contract.MAP_DIRECTION].T
        > 0.0)
    candidates = faces_the_agent_way | ~faces_the_agent_way.any(axis=1,
                                                                keepdims=True)
    return np.where(candidates, agent_distances, np.inf).argmin(axis=1)


def lane_dots_of_scenario(
        scenario_array: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    map_rows = scenario_array["map_rows"]
    dot_polyline_index = scenario_array["map_dot_polyline_index"]
    lane_kind_index = contract.MAP_POLYLINE_KINDS.index("lane")
    lane_kind_column = contract.MAP_KIND.start + lane_kind_index
    lane_dot_indices = np.flatnonzero(map_rows[:, lane_kind_column] == 1.0)
    return (
        map_rows[lane_dot_indices],
        dot_polyline_index[lane_dot_indices],
    )


def assigned_lane_signal_histories(
        lane_dot_rows: np.ndarray, polyline_row_of_lane_dot: np.ndarray,
        polyline_signal_histories: np.ndarray, agent_now_rows: np.ndarray,
        agent_has_now_step: np.ndarray) -> np.ndarray:
    """Assigns each agent at the current step the signal history of its nearest
    facing lane dot's polyline; others get all zeros.
    """
    signal_histories = np.zeros(
        (
            len(agent_now_rows),
            contract.HISTORY_STEPS,
            contract.NUM_TRAFFIC_SIGNAL_STATES,
        ),
        dtype=np.float32,
    )
    assignable = np.flatnonzero(agent_has_now_step)
    if len(lane_dot_rows) == 0 or len(assignable) == 0:
        return signal_histories

    assignable_rows = agent_now_rows[assignable]
    assignable_positions = assignable_rows[:, contract.AGENT_POSITION]
    lane_dot_positions = lane_dot_rows[:, contract.MAP_POSITION]
    # Broadcasts the assignable agents against the lane dots to
    # get one (agent, lane dot) distance matrix.
    offsets_x = lane_dot_positions[None, :, 0] - assignable_positions[:, :1]
    offsets_y = lane_dot_positions[None, :, 1] - assignable_positions[:, 1:]
    agent_distances = np.sqrt(offsets_x * offsets_x + offsets_y * offsets_y)
    assigned_lane_dot = nearest_same_direction_lane_dot(
        lane_dot_rows,
        agent_distances,
        assignable_rows[
            :,
            contract.AGENT_HEADING_COSINE:contract.AGENT_HEADING_SINE + 1,
        ],
    )
    signal_histories[assignable] = polyline_signal_histories[
        polyline_row_of_lane_dot[assigned_lane_dot]]
    return signal_histories


def chunk_dots_by_polyline(
        dot_polyline_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Groups each polyline's dots into chunks of MAP_CHUNK_DOTS.

    Returns the chunk each dot falls in and the polyline each chunk belongs to.
    """
    # The index is sorted, so np.unique's first-occurrence index gives
    # each dot's offset within its own polyline's block.
    (
        polylines,
        first_dot_position,
        compact_polyline_index,
        dots_per_polyline,
    ) = np.unique(
        dot_polyline_index,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    dot_positions = np.arange(len(compact_polyline_index))
    first_dot_of_own_polyline = first_dot_position[compact_polyline_index]
    position_within_polyline = dot_positions - first_dot_of_own_polyline

    chunk_dots = contract.MAP_CHUNK_DOTS
    chunks_per_polyline = (dots_per_polyline + chunk_dots - 1) // chunk_dots
    # Each polyline's chunk indices start where the previous polyline's
    # chunks left off, an exclusive cumulative sum.
    chunks_up_to_polyline = np.cumsum(chunks_per_polyline)
    first_chunk_of_polyline = chunks_up_to_polyline - chunks_per_polyline
    first_chunk_of_own_polyline = first_chunk_of_polyline[
        compact_polyline_index]
    chunk_within_polyline = position_within_polyline // chunk_dots
    dot_chunk_index = first_chunk_of_own_polyline + chunk_within_polyline

    chunk_polyline = np.repeat(polylines, chunks_per_polyline)
    return dot_chunk_index, chunk_polyline


def with_derived_arrays(
        scenario_array: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Adds map_dot_polyline_index (which polyline each map row belongs to) and
    track_signal_histories to a loaded scenario.
    """
    feature_lengths = scenario_array["feature_lengths"]
    scenario_array["map_dot_polyline_index"] = np.repeat(
        np.arange(len(feature_lengths)), feature_lengths)
    lane_dot_rows, polyline_row_of_lane_dot = lane_dots_of_scenario(
        scenario_array)
    scenario_array["track_signal_histories"] = assigned_lane_signal_histories(
        lane_dot_rows,
        polyline_row_of_lane_dot,
        scenario_array["polyline_signal_histories"],
        scenario_array["track_rows"][:, contract.CURRENT_STEP_INDEX],
        scenario_array["track_valid"][:, contract.CURRENT_STEP_INDEX],
    )
    return scenario_array


def read_scenario(scenario_path: Path | str) -> dict[str, np.ndarray]:
    """Loads one staged .npz scenario, checks its provenance stamp, and returns
    it with derived arrays added.
    """
    with np.load(scenario_path) as scenario_file:
        scenario_array = {
            name: scenario_file[name] for name in scenario_file.files
        }
    if "provenance" in scenario_array:
        contract.check_artifact_provenance(
            scenario_array["provenance"],
            scenario_path,
            "Restage the directory with stage.py.",
        )
    feature_lengths = scenario_array["feature_lengths"]
    map_row_count = len(scenario_array["map_rows"])
    assert feature_lengths.sum() == map_row_count, (
        f"feature_lengths sum to {feature_lengths.sum()} but"
        f" {scenario_path} holds {map_row_count} map rows")
    return with_derived_arrays(scenario_array)


def poses_in_agent_frame(positions: np.ndarray,
                         direction_cosine_sine: np.ndarray, origin: np.ndarray,
                         heading: float) -> np.ndarray:
    """One pose row per token, (x, y, cosine, sine), in the given agent's frame.
    """
    return np.concatenate(
        [
            frame_ops.positions_to_frame(positions, origin, heading),
            frame_ops.directions_to_frame(direction_cosine_sine, heading),
        ],
        axis=1,
    ).astype(np.float32)


class SceneTokens(NamedTuple):
    """Where each scene token sits in the scene frame, one per agent and one
    per map chunk.
    """
    agent_present: np.ndarray
    agent_positions: np.ndarray
    agent_directions: np.ndarray
    dot_chunk_index: np.ndarray
    chunk_polyline: np.ndarray
    chunk_positions: np.ndarray
    chunk_directions: np.ndarray


def scene_tokens(scenario_array: dict[str, np.ndarray]) -> SceneTokens:
    """Places every agent at its last valid history step and every map chunk at
    its middle dot.
    """
    track_rows = scenario_array["track_rows"]
    map_rows = scenario_array["map_rows"]
    history_valid = scenario_array["track_valid"][:, :contract.HISTORY_STEPS]
    # The first True in the reversed mask is the last True in the
    # original order.
    steps_from_end = history_valid[:, ::-1].argmax(axis=1)
    last_valid_step = contract.HISTORY_STEPS - 1 - steps_from_end
    agent_rows = track_rows[np.arange(len(track_rows)), last_valid_step]
    agent_directions = np.stack(
        [
            agent_rows[:, contract.AGENT_HEADING_COSINE],
            agent_rows[:, contract.AGENT_HEADING_SINE],
        ],
        axis=1,
    )

    dot_chunk_index, chunk_polyline = chunk_dots_by_polyline(
        scenario_array["map_dot_polyline_index"])
    _, first_dot_of_chunk, dots_per_chunk = np.unique(dot_chunk_index,
                                                      return_index=True,
                                                      return_counts=True)
    chunk_rows = map_rows[first_dot_of_chunk + dots_per_chunk // 2]
    return SceneTokens(
        agent_present=history_valid.any(axis=1),
        agent_positions=agent_rows[:, contract.AGENT_POSITION],
        agent_directions=agent_directions,
        dot_chunk_index=dot_chunk_index,
        chunk_polyline=chunk_polyline,
        chunk_positions=chunk_rows[:, contract.MAP_POSITION],
        chunk_directions=chunk_rows[:, contract.MAP_DIRECTION],
    )


class TargetSample(NamedTuple):
    """One predicted agent's view of its scene, in that agent's own frame. The
    token fields run over the scene's agents and then its map chunks.
    """
    agent_history: np.ndarray  # (11, 13)
    agent_history_mask: np.ndarray  # (11,)
    agent_signal_history: np.ndarray  # (11, 9)
    future_positions: np.ndarray  # (80, 2)
    future_mask: np.ndarray  # (80,)
    agent_token_visible: np.ndarray  # (agents,)
    chunk_token_visible: np.ndarray  # (chunks,)
    agent_token_pose: np.ndarray  # (agents, 4) x, y, cos, sin
    chunk_token_pose: np.ndarray  # (chunks, 4)
    frame_origin: np.ndarray  # (2,) the agent's position in the scene frame
    frame_heading: float  # the agent's heading in the scene frame
    track_id: int


class SceneSample(NamedTuple):
    """One scene in the staged scene frame plus one TargetSample per predicted
    agent.
    """
    scenario_id: str
    scene_agent_history: np.ndarray  # (agents, 11, 13)
    scene_agent_history_mask: np.ndarray  # (agents, 11)
    scene_agent_signal_history: np.ndarray  # (agents, 11, 9)
    map_rows: np.ndarray  # (dots, 32)
    map_chunk_index: np.ndarray  # (dots,) which chunk each dot belongs to
    map_chunk_signal_history: np.ndarray  # (chunks, 11, 9)
    targets: list[TargetSample]


def build_target(scenario_array: dict[str, np.ndarray], track_index: int,
                 tokens: SceneTokens) -> TargetSample:
    """One predicted agent's view of its scene.

    Holds its own history and future in its own frame, which scene tokens fall
    inside its crop, and every token's pose re-expressed in its frame.
    """
    track_rows = scenario_array["track_rows"]
    track_valid = scenario_array["track_valid"]
    map_rows = scenario_array["map_rows"]
    history = slice(0, contract.HISTORY_STEPS)
    future = slice(contract.CURRENT_STEP_INDEX + 1, None)

    origin, heading = sample_frame(track_rows, track_index)
    agent_track = track_rows_to_agent_frame(track_rows[track_index], origin,
                                            heading)

    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    speed = float(np.linalg.norm(now_row[contract.AGENT_VELOCITY]))
    dot_inside_crop = inside_crop(
        frame_ops.positions_to_frame(map_rows[:, contract.MAP_POSITION], origin,
                                     heading),
        base_radius=BASE_RADIUS_METRES,
        forward_stretch=1.0 + STRETCH_GAIN * speed,
    )
    # A chunk is visible if any of its dots falls inside the crop.
    dots_inside_per_chunk = np.bincount(tokens.dot_chunk_index,
                                        weights=dot_inside_crop,
                                        minlength=len(tokens.chunk_polyline))
    # A target never sees itself as a scene token.
    agent_visible = tokens.agent_present.copy()
    agent_visible[track_index] = False

    signal_histories = scenario_array["track_signal_histories"]
    return TargetSample(
        agent_history=agent_track[history],
        agent_history_mask=track_valid[track_index, history],
        agent_signal_history=signal_histories[track_index],
        future_positions=agent_track[future, contract.AGENT_POSITION],
        future_mask=track_valid[track_index, future],
        agent_token_visible=agent_visible,
        chunk_token_visible=dots_inside_per_chunk > 0,
        agent_token_pose=poses_in_agent_frame(tokens.agent_positions,
                                              tokens.agent_directions, origin,
                                              heading),
        chunk_token_pose=poses_in_agent_frame(tokens.chunk_positions,
                                              tokens.chunk_directions, origin,
                                              heading),
        frame_origin=origin,
        frame_heading=heading,
        track_id=scenario_array["track_ids"][track_index],
    )


def build_scene_sample(scenario_array: dict[str, np.ndarray],
                       track_indices: list[int]) -> SceneSample:
    """Builds one scene sample, every agent and the whole staged map in the
    scene frame plus one target entry per predicted agent.
    """
    tokens = scene_tokens(scenario_array)
    history = slice(0, contract.HISTORY_STEPS)
    polyline_signal_histories = scenario_array["polyline_signal_histories"]
    return SceneSample(
        scenario_id=scenario_array["scenario_id"],
        scene_agent_history=scenario_array["track_rows"][:, history],
        scene_agent_history_mask=scenario_array["track_valid"][:, history],
        scene_agent_signal_history=scenario_array["track_signal_histories"],
        map_rows=scenario_array["map_rows"],
        map_chunk_index=tokens.dot_chunk_index.astype(np.int64),
        map_chunk_signal_history=polyline_signal_histories[
            tokens.chunk_polyline],
        targets=[
            build_target(scenario_array, track_index, tokens)
            for track_index in track_indices
        ],
    )


def pad_and_stack(arrays: list[np.ndarray], padded_length: int,
                  dtype: np.dtype | type) -> np.ndarray:
    """Stacks arrays whose first axis varies, zero-padding each to
    padded_length.
    """
    padded_shape = (len(arrays), padded_length) + arrays[0].shape[1:]
    padded = np.zeros(padded_shape, dtype=dtype)
    for row, array in zip(padded, arrays):
        row[:len(array)] = array
    return padded


class SceneArrays(NamedTuple):
    """Every agent of every scene in the batch, in the staged scene frame,
    zero-padded to the A agents of the largest of the S scenes.
    """
    agent_history: np.ndarray  # (S, A, 11, 13)
    agent_history_mask: np.ndarray  # (S, A, 11)
    agent_signal_history: np.ndarray  # (S, A, 11, 9)


class MapArrays(NamedTuple):
    """Every map dot of every scene in the batch, D dots concatenated, each
    with a slot naming its scene and chunk out of the C chunks of the largest
    scene.
    """
    rows: np.ndarray  # (D, 32)
    dot_chunk_slot: np.ndarray  # (D,) scene * C + chunk
    chunk_signal_history: np.ndarray  # (S, C, 11, 9)


class TargetArrays(NamedTuple):
    """The B predicted agents of the batch, each in its own frame, with the
    tokens of its scene, A agents and then C chunks, seen through its crop.
    The two future fields are absent from an inference batch.
    """
    scene_index: np.ndarray  # (B,)
    agent_history: np.ndarray  # (B, 11, 13)
    agent_history_mask: np.ndarray  # (B, 11)
    agent_signal_history: np.ndarray  # (B, 11, 9)
    token_visible: np.ndarray  # (B, A + C)
    token_pose: np.ndarray  # (B, A + C, 4) x, y, cos, sin
    future_positions: np.ndarray | None = None  # (B, 80, 2)
    future_mask: np.ndarray | None = None  # (B, 80)


class SceneBatch(NamedTuple):
    """One batch from build_scene_batch. Padding is zero, and the DataLoader
    hands the same fields on as torch tensors.
    """
    scene: SceneArrays
    map: MapArrays
    targets: TargetArrays

    def each(self, function: Callable[[np.ndarray], Any]) -> SceneBatch:
        """The same batch with function applied to every array, by name."""

        def applied(group: NamedTuple) -> NamedTuple:
            return type(group)(
                **{
                    name: None if array is None else function(array)
                    for name, array in group._asdict().items()
                })

        return SceneBatch(
            scene=applied(self.scene),
            map=applied(self.map),
            targets=applied(self.targets),
        )


def build_scene_batch(scene_samples: list[SceneSample]) -> SceneBatch:
    """Joins scene samples into one batch.

    Scene agents and map chunks are zero-padded to the largest scene, map dots
    are concatenated with a slot saying whose they are, and every scene's
    targets are flattened into one target axis that points back at its scene.
    """
    max_agents = max(len(scene.scene_agent_history) for scene in scene_samples)
    max_chunks = max(
        len(scene.map_chunk_signal_history) for scene in scene_samples)
    targets = [(scene_index, target)
               for scene_index, scene in enumerate(scene_samples)
               for target in scene.targets]

    def scene_entry(name: str, padded_length: int,
                    dtype: np.dtype | type) -> np.ndarray:
        return pad_and_stack(
            [getattr(scene, name) for scene in scene_samples], padded_length,
            dtype)

    def target_entry(name: str) -> np.ndarray:
        return np.stack([getattr(target, name) for _, target in targets])

    def target_tokens(agent_name: str, chunk_name: str,
                      dtype: np.dtype | type) -> np.ndarray:
        """One per-token entry across every target, in token order, the padded
        scene agents and then the padded map chunks.
        """
        agent_part = pad_and_stack(
            [getattr(target, agent_name) for _, target in targets], max_agents,
            dtype)
        chunk_part = pad_and_stack(
            [getattr(target, chunk_name) for _, target in targets], max_chunks,
            dtype)
        return np.concatenate([agent_part, chunk_part], axis=1)

    # A dot's slot is scene_index * max_chunks + its chunk index,
    # which is unique across the batch.
    dot_chunk_slots = [
        scene.map_chunk_index + scene_index * max_chunks
        for scene_index, scene in enumerate(scene_samples)
    ]

    return SceneBatch(
        scene=SceneArrays(
            agent_history=scene_entry("scene_agent_history", max_agents,
                                      np.float32),
            agent_history_mask=scene_entry("scene_agent_history_mask",
                                           max_agents, bool),
            agent_signal_history=scene_entry("scene_agent_signal_history",
                                             max_agents, np.float32),
        ),
        map=MapArrays(
            rows=np.concatenate([scene.map_rows for scene in scene_samples],
                                dtype=np.float32),
            dot_chunk_slot=np.concatenate(dot_chunk_slots, dtype=np.int64),
            chunk_signal_history=scene_entry("map_chunk_signal_history",
                                             max_chunks, np.float32),
        ),
        targets=TargetArrays(
            scene_index=np.array([scene_index for scene_index, _ in targets],
                                 dtype=np.int64),
            agent_history=target_entry("agent_history"),
            agent_history_mask=target_entry("agent_history_mask"),
            agent_signal_history=target_entry("agent_signal_history"),
            token_visible=target_tokens("agent_token_visible",
                                        "chunk_token_visible", bool),
            token_pose=target_tokens("agent_token_pose", "chunk_token_pose",
                                     np.float32),
            future_positions=target_entry("future_positions"),
            future_mask=target_entry("future_mask"),
        ),
    )


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
