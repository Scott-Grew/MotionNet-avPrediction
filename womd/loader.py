"""Turns a staged scenario into model inputs, one sample per predicted agent, in
that agent's frame. Layout drawn at the end.
"""
from pathlib import Path
from typing import Any

import numpy as np

from womd import contract, frame_ops

# The map an agent sees is cropped around it. A faster agent
# covers more road in 8 s, so the crop stretches ahead with speed.
BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5


def inside_crop(agent_frame_points: np.ndarray, base_radius: float,
                forward_stretch: float) -> np.ndarray:
    """Ellipse crop test in agent-frame coordinates: ahead of the agent it
    stretches by forward_stretch; behind it is a circle.
    """
    ahead = agent_frame_points[:, 0]
    sideways = agent_frame_points[:, 1]
    sideways_term = (sideways / base_radius)**2
    front_term = (ahead / (base_radius * forward_stretch))**2
    rear_term = (ahead / base_radius)**2
    inside_front_ellipse = front_term + sideways_term <= 1.0
    inside_rear_circle = rear_term + sideways_term <= 1.0
    return np.where(ahead > 0.0, inside_front_ellipse, inside_rear_circle)


def eligible_track_indices(track_rows: np.ndarray, track_valid: np.ndarray,
                           is_designated_target: np.ndarray,
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
    """A track's agent frame at the current step: its position and heading."""
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
    """Selects the map rows and polyline indices belonging to lane-kind dots
    only.
    """
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
        cropped_polyline_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Groups each polyline's surviving dots into chunks of MAP_CHUNK_DOTS.

    Returns the chunk each dot falls in and the polyline each chunk belongs to.
    """
    # The index is sorted, so np.unique's first-occurrence index gives
    # each dot's offset within its own polyline's block.
    (
        surviving_polylines,
        first_dot_position,
        compact_polyline_index,
        dots_per_polyline,
    ) = np.unique(
        cropped_polyline_index,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    dot_positions = np.arange(len(compact_polyline_index))
    first_dot_of_own_polyline = first_dot_position[compact_polyline_index]
    position_within_polyline = dot_positions - first_dot_of_own_polyline

    chunk_dots = contract.MAP_CHUNK_DOTS
    chunks_per_polyline = (dots_per_polyline + chunk_dots - 1) // chunk_dots
    # Each polyline's chunk indices start where the previous
    # polyline's chunks left off: an exclusive cumulative sum.
    chunks_up_to_polyline = np.cumsum(chunks_per_polyline)
    first_chunk_of_polyline = chunks_up_to_polyline - chunks_per_polyline
    first_chunk_of_own_polyline = first_chunk_of_polyline[
        compact_polyline_index]
    chunk_within_polyline = position_within_polyline // chunk_dots
    dot_chunk_index = first_chunk_of_own_polyline + chunk_within_polyline

    chunk_polyline = np.repeat(surviving_polylines, chunks_per_polyline)
    return dot_chunk_index, chunk_polyline


def crop_and_reframe_map(
        map_rows: np.ndarray, dot_polyline_index: np.ndarray,
        polyline_signal_histories: np.ndarray, origin: np.ndarray,
        heading: float,
        speed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crops map dots to a speed-stretched ellipse, reframes them, then chunks
    survivors per polyline in groups of MAP_CHUNK_DOTS.
    """
    agent_frame_positions = frame_ops.positions_to_frame(
        map_rows[:, contract.MAP_POSITION], origin, heading)
    crop_mask = inside_crop(
        agent_frame_positions,
        BASE_RADIUS_METRES,
        1.0 + STRETCH_GAIN * speed,
    )
    cropped_polyline_index = dot_polyline_index[crop_mask]

    cropped_rows = map_rows[crop_mask]
    cropped_rows[:, contract.MAP_POSITION] = agent_frame_positions[crop_mask]
    cropped_rows[:, contract.MAP_DIRECTION] = frame_ops.directions_to_frame(
        cropped_rows[:, contract.MAP_DIRECTION], heading)

    # Only lanes under a traffic light carry a real stop point;
    # elsewhere the columns are zero and must stay zero.
    polyline_has_signal = polyline_signal_histories.any(axis=(1, 2))
    has_signal = polyline_has_signal[cropped_polyline_index]
    stop_points = cropped_rows[has_signal, contract.MAP_STOP_POINT]
    cropped_rows[has_signal,
                 contract.MAP_STOP_POINT] = (frame_ops.positions_to_frame(
                     stop_points, origin, heading))

    dot_chunk_index, chunk_polyline = chunk_dots_by_polyline(
        cropped_polyline_index)
    return (
        cropped_rows,
        dot_chunk_index,
        polyline_signal_histories[chunk_polyline],
    )


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


def build_sample(scenario_array: dict[str, np.ndarray],
                 track_index: int) -> dict[str, Any]:
    """Builds one sample for a single agent: its track, the other tracks as
    neighbours, and the cropped map, in its own frame.
    """
    track_rows = scenario_array["track_rows"]
    track_valid = scenario_array["track_valid"]
    signal_histories = scenario_array["track_signal_histories"]
    history = slice(0, contract.HISTORY_STEPS)
    future = slice(contract.CURRENT_STEP_INDEX + 1, None)

    origin, heading = sample_frame(track_rows, track_index)

    # The agent itself.
    agent_track = track_rows_to_agent_frame(track_rows[track_index], origin,
                                            heading)

    # Every other track is a neighbour.
    is_neighbour = np.arange(len(track_rows)) != track_index
    neighbour_indices = np.flatnonzero(is_neighbour)
    neighbour_history = track_rows_to_agent_frame(
        track_rows[neighbour_indices, history], origin, heading)

    # The map, cropped around the agent by its current speed.
    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    speed = float(np.linalg.norm(now_row[contract.AGENT_VELOCITY]))
    map_rows, map_chunk_index, map_chunk_signal_history = crop_and_reframe_map(
        scenario_array["map_rows"],
        scenario_array["map_dot_polyline_index"],
        scenario_array["polyline_signal_histories"],
        origin,
        heading,
        speed,
    )

    is_target = scenario_array["is_designated_target"][track_index]
    is_of_interest = scenario_array["is_object_of_interest"][track_index]
    return {
        "agent_history": agent_track[history],
        "agent_history_mask": track_valid[track_index, history],
        "agent_signal_history": signal_histories[track_index],
        "future_positions": agent_track[future, contract.AGENT_POSITION],
        "future_mask": track_valid[track_index, future],
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": track_valid[neighbour_indices, history],
        "neighbour_signal_history": signal_histories[neighbour_indices],
        "map_rows": map_rows,
        "map_chunk_index": map_chunk_index.astype(np.int64),
        "map_chunk_signal_history": map_chunk_signal_history,
        "frame_origin": origin,
        "frame_heading": heading,
        "scenario_id": scenario_array["scenario_id"],
        "track_id": scenario_array["track_ids"][track_index],
        "is_designated_target": is_target,
        "is_object_of_interest": is_of_interest,
    }


def stack_entry(samples: list[dict[str, Any]], key: str) -> np.ndarray:
    """Stacks one fixed-shape entry of the samples along a new batch axis."""
    return np.stack([sample[key] for sample in samples])


def pad_and_stack_entry(samples: list[dict[str, Any]], key: str,
                        padded_length: int) -> np.ndarray:
    """Stacks one entry whose first axis varies between samples (its neighbours,
    its map chunks), zero-padding each to padded_length.
    """
    first_entry = samples[0][key]
    padded_shape = (len(samples), padded_length) + first_entry.shape[1:]
    padded = np.zeros(padded_shape, dtype=first_entry.dtype)
    for sample_index, sample in enumerate(samples):
        padded[sample_index, :len(sample[key])] = sample[key]
    return padded


def build_batch(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Joins samples into one batch.

    Fixed-shape entries are stacked, neighbours and map chunks are zero-padded
    to the largest sample, and map dots are concatenated with a slot saying
    whose they are.
    """
    max_neighbours = max(len(sample["neighbour_history"]) for sample in samples)
    max_chunks = max(
        len(sample["map_chunk_signal_history"]) for sample in samples)

    # A dot's slot is sample_index * max_chunks + its chunk index,
    # which is unique across the batch.
    dot_chunk_slots = [
        sample["map_chunk_index"] + sample_index * max_chunks
        for sample_index, sample in enumerate(samples)
    ]
    map_rows = [sample["map_rows"] for sample in samples]

    batch = {}
    for key in ("agent_history", "agent_history_mask", "agent_signal_history",
                "future_positions", "future_mask"):
        batch[key] = stack_entry(samples, key)
    for key in ("neighbour_history", "neighbour_history_mask",
                "neighbour_signal_history"):
        batch[key] = pad_and_stack_entry(samples, key, max_neighbours)
    batch["map_rows"] = np.concatenate(map_rows, dtype=np.float32)
    batch["map_dot_chunk_slot"] = np.concatenate(dot_chunk_slots,
                                                 dtype=np.int64)
    batch["map_chunk_signal_history"] = pad_and_stack_entry(
        samples, "map_chunk_signal_history", max_chunks)
    batch["max_chunks_in_batch"] = np.array(max_chunks, dtype=np.int64)
    return batch


# ------------------------------------------------------------------
# WHAT THE MODEL READS: one batch from build_batch(), in each
# predicted agent's own frame. B agents, N neighbours, C map chunks,
# D map dots in the whole batch. Padding is zero.
#
#   agent_history             (B, 11, 13)     the agent, 11 steps
#   agent_history_mask        (B, 11)
#   agent_signal_history      (B, 11, 9)      its lane's traffic light
#   neighbour_history         (B, N, 11, 13)  the other agents
#   neighbour_history_mask    (B, N, 11)
#   neighbour_signal_history  (B, N, 11, 9)
#   map_rows                  (D, 32)         one row per map dot
#   map_dot_chunk_slot        (D,)            sample * C + chunk
#   map_chunk_signal_history  (B, C, 11, 9)
#   future_positions          (B, 80, 2)      the answer, for the loss
#   future_mask               (B, 80)
#
# One agent row, per 0.1 s step:
#
#     0   1   2   3   4   5   6   7   8   9  10  11  12
#   +-------+---+---+-------+-------+---------------+---+
#   | x   y |cos|sin|vx  vy |len wid|veh ped cyc oth|sdc|
#   +-------+---+---+-------+-------+---------------+---+
#
# One map row, per metre of a map feature:
#
#     0-1   2-3    4-10    11-14   15    16-27   28-29  30  31
#   +-----+-----+--------+-------+-----+--------+------+---+---+
#   | x y |dx dy| kind   | lane  |speed|boundary| stop | L | R |
#   |     |     | 1-hot  | type  |limit| type   | x  y |   |   |
#   +-----+-----+--------+-------+-----+--------+------+---+---+
#
#   L, R: marking on the lane's left and right, 0 for none.
# ------------------------------------------------------------------
