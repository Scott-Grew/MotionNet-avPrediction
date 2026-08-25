import numpy as np

from womd import contract, frame_ops

BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5

def inside_crop(agent_frame_points, base_radius, forward_stretch):
    x = agent_frame_points[:, 0]
    y = agent_frame_points[:, 1]
    forward = (x / (base_radius * forward_stretch)) ** 2 + (y / base_radius) ** 2 <= 1.0
    rear = (x / base_radius) ** 2 + (y / base_radius) ** 2 <= 1.0
    return np.where(x > 0.0, forward, rear)

def eligible_track_indices(track_rows, track_valid, is_designated_target, designated_targets_only):
    now_valid = track_valid[:, contract.CURRENT_STEP_INDEX]
    predicted_type = track_rows[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE][:,
                                            :contract.NUM_OBJECT_TYPES].sum(axis=1) > 0
    selected = now_valid & predicted_type
    if designated_targets_only:
        selected = selected & is_designated_target
    return np.flatnonzero(selected)

def sample_frame(track_rows, track_index):
    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    origin = now_row[contract.AGENT_POSITION]
    heading = np.arctan2(now_row[contract.AGENT_HEADING_SINE], now_row[contract.AGENT_HEADING_COSINE])
    return origin, heading

def track_rows_to_agent_frame(rows, origin, heading):
    reframed = rows.copy()
    reframed[..., contract.AGENT_POSITION] = frame_ops.positions_to_agent_frame(
        rows[..., contract.AGENT_POSITION], origin, heading
    )
    reframed[..., contract.AGENT_VELOCITY] = frame_ops.directions_to_agent_frame(
        rows[..., contract.AGENT_VELOCITY], heading
    )
    heading_cosine = rows[..., contract.AGENT_HEADING_COSINE]
    heading_sine = rows[..., contract.AGENT_HEADING_SINE]
    rotation_cosine, rotation_sine = np.cos(heading), np.sin(heading)
    reframed[..., contract.AGENT_HEADING_COSINE] = heading_cosine * rotation_cosine + heading_sine * rotation_sine
    reframed[..., contract.AGENT_HEADING_SINE] = heading_sine * rotation_cosine - heading_cosine * rotation_sine
    return reframed

def nearest_lane_dot_facing_the_agent_way(lane_dot_rows, agent_distances, agent_heading_cosine_sine):
    faces_the_agent_way = (
        agent_heading_cosine_sine @ lane_dot_rows[:, contract.MAP_DIRECTION].T > 0.0
    )
    candidates = faces_the_agent_way | ~faces_the_agent_way.any(axis=1, keepdims=True)
    return np.where(candidates, agent_distances, np.inf).argmin(axis=1)

def lane_dots_of_scenario(scenario_array):
    map_rows = scenario_array["map_rows"]
    dot_polyline_index = scenario_array["map_dot_polyline_index"]
    lane_kind_column = contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")
    lane_dot_indices = np.flatnonzero(map_rows[:, lane_kind_column] == 1.0)
    return map_rows[lane_dot_indices], dot_polyline_index[lane_dot_indices]

def assigned_lane_signal_histories(
    lane_dot_rows, polyline_row_of_lane_dot, polyline_signal_histories,
    agent_now_rows, agent_has_now_step,
):
    signal_histories = np.zeros(
        (len(agent_now_rows), contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        dtype=np.float32,
    )
    assignable = np.flatnonzero(agent_has_now_step)
    if len(lane_dot_rows) == 0 or len(assignable) == 0:
        return signal_histories

    assignable_rows = agent_now_rows[assignable]
    assignable_positions = assignable_rows[:, contract.AGENT_POSITION]
    lane_dot_positions = lane_dot_rows[:, contract.MAP_POSITION]
    offsets_x = lane_dot_positions[None, :, 0] - assignable_positions[:, :1]
    offsets_y = lane_dot_positions[None, :, 1] - assignable_positions[:, 1:]
    agent_distances = np.sqrt(offsets_x * offsets_x + offsets_y * offsets_y)
    assigned_lane_dot = nearest_lane_dot_facing_the_agent_way(
        lane_dot_rows,
        agent_distances,
        assignable_rows[:, contract.AGENT_HEADING_COSINE:contract.AGENT_HEADING_SINE + 1],
    )
    signal_histories[assignable] = polyline_signal_histories[
        polyline_row_of_lane_dot[assigned_lane_dot]
    ]
    return signal_histories

def crop_and_reframe_map(map_rows, dot_polyline_index, polyline_signal_histories, origin, heading, speed):
    agent_frame_positions = frame_ops.positions_to_agent_frame(map_rows[:, contract.MAP_POSITION], origin, heading)
    crop_mask = inside_crop(agent_frame_positions, BASE_RADIUS_METRES, 1.0 + STRETCH_GAIN * speed)
    reframed = map_rows[crop_mask]
    reframed[:, contract.MAP_POSITION] = agent_frame_positions[crop_mask]
    reframed[:, contract.MAP_DIRECTION] = frame_ops.directions_to_agent_frame(reframed[:, contract.MAP_DIRECTION], heading)
    has_signal = polyline_signal_histories.any(axis=(1, 2))[dot_polyline_index[crop_mask]]
    reframed[has_signal, contract.MAP_STOP_POINT] = frame_ops.positions_to_agent_frame(reframed[has_signal, contract.MAP_STOP_POINT], origin, heading)
    surviving_polylines, first_dot_position, compact_polyline_index, dots_per_polyline = np.unique(
        dot_polyline_index[crop_mask], return_index=True, return_inverse=True, return_counts=True
    )
    position_within_polyline = np.arange(len(compact_polyline_index)) - first_dot_position[compact_polyline_index]
    chunks_per_polyline = (dots_per_polyline + contract.MAP_CHUNK_DOTS - 1) // contract.MAP_CHUNK_DOTS
    first_chunk_of_polyline = np.cumsum(chunks_per_polyline) - chunks_per_polyline
    dot_chunk_index = (first_chunk_of_polyline[compact_polyline_index]
                       + position_within_polyline // contract.MAP_CHUNK_DOTS)
    chunk_polyline = np.repeat(surviving_polylines, chunks_per_polyline)
    return reframed, dot_chunk_index, polyline_signal_histories[chunk_polyline]

def with_derived_arrays(scenario_array):
    feature_lengths = scenario_array["feature_lengths"]
    scenario_array["map_dot_polyline_index"] = np.repeat(
        np.arange(len(feature_lengths)), feature_lengths
    )
    lane_dot_rows, polyline_row_of_lane_dot = lane_dots_of_scenario(scenario_array)
    scenario_array["track_signal_histories"] = assigned_lane_signal_histories(
        lane_dot_rows, polyline_row_of_lane_dot,
        scenario_array["polyline_signal_histories"],
        scenario_array["track_rows"][:, contract.CURRENT_STEP_INDEX],
        scenario_array["track_valid"][:, contract.CURRENT_STEP_INDEX],
    )
    return scenario_array

def read_scenario(scenario_path):
    with np.load(scenario_path) as scenario_file:
        scenario_array = {name: scenario_file[name] for name in scenario_file.files}
    if "provenance" in scenario_array:
        contract.check_artifact_provenance(
            scenario_array["provenance"], scenario_path,
            "Restage the directory with stage.py.",
        )
    feature_lengths = scenario_array["feature_lengths"]
    map_row_count = len(scenario_array["map_rows"])
    assert feature_lengths.sum() == map_row_count, (
        f"feature_lengths sum to {feature_lengths.sum()} but {scenario_path} holds"
        f" {map_row_count} map rows"
    )
    return with_derived_arrays(scenario_array)

def build_sample(scenario_array, track_index):
    track_rows = scenario_array["track_rows"]
    track_valid = scenario_array["track_valid"]
    origin, heading = sample_frame(track_rows, track_index)

    agent_track = track_rows_to_agent_frame(track_rows[track_index], origin, heading)
    neighbour_indices = np.flatnonzero(np.arange(len(track_rows)) != track_index)
    neighbour_history = track_rows_to_agent_frame(
        track_rows[neighbour_indices, :contract.HISTORY_STEPS], origin, heading
    )

    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    speed = float(np.linalg.norm(now_row[contract.AGENT_VELOCITY]))
    track_signal_histories = scenario_array["track_signal_histories"]
    agent_map, map_chunk_index, map_chunk_signal_history = crop_and_reframe_map(
        scenario_array["map_rows"],
        scenario_array["map_dot_polyline_index"],
        scenario_array["polyline_signal_histories"],
        origin,
        heading,
        speed,
    )

    return {
        "agent_history": agent_track[:contract.HISTORY_STEPS],
        "agent_history_mask": track_valid[track_index, :contract.HISTORY_STEPS],
        "agent_signal_history": track_signal_histories[track_index],
        "future_positions": agent_track[contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION],
        "future_mask": track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:],
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": track_valid[neighbour_indices, :contract.HISTORY_STEPS],
        "neighbour_signal_history": track_signal_histories[neighbour_indices],
        "map_rows": agent_map,
        "map_chunk_index": map_chunk_index.astype(np.int64),
        "map_chunk_signal_history": map_chunk_signal_history,
        "frame_origin": origin,
        "frame_heading": heading,
        "scenario_id": scenario_array["scenario_id"],
        "track_id": scenario_array["track_ids"][track_index],
        "is_designated_target": scenario_array["is_designated_target"][track_index],
        "is_object_of_interest": scenario_array["is_object_of_interest"][track_index],
    }

def build_batch(samples):
    batch_size = len(samples)
    max_neighbours = max(sample["neighbour_history"].shape[0] for sample in samples)
    max_chunks_in_batch = max(
        int(sample["map_chunk_index"].max()) + 1 if len(sample["map_chunk_index"]) else 0
        for sample in samples
    )

    neighbour_history = np.zeros(
        (batch_size, max_neighbours, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32
    )
    neighbour_history_mask = np.zeros((batch_size, max_neighbours, contract.HISTORY_STEPS), dtype=bool)
    neighbour_signal_history = np.zeros(
        (batch_size, max_neighbours, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        dtype=np.float32,
    )
    map_chunk_signal_history = np.zeros(
        (batch_size, max_chunks_in_batch, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        dtype=np.float32,
    )

    for sample_index, sample in enumerate(samples):
        neighbour_count = sample["neighbour_history"].shape[0]
        neighbour_history[sample_index, :neighbour_count] = sample["neighbour_history"]
        neighbour_history_mask[sample_index, :neighbour_count] = sample["neighbour_history_mask"]
        neighbour_signal_history[sample_index, :neighbour_count] = sample["neighbour_signal_history"]
        chunk_count = sample["map_chunk_signal_history"].shape[0]
        map_chunk_signal_history[sample_index, :chunk_count] = sample["map_chunk_signal_history"]

    return {
        "agent_history": np.stack([sample["agent_history"] for sample in samples]),
        "agent_history_mask": np.stack([sample["agent_history_mask"] for sample in samples]),
        "agent_signal_history": np.stack([sample["agent_signal_history"] for sample in samples]),
        "future_positions": np.stack([sample["future_positions"] for sample in samples]),
        "future_mask": np.stack([sample["future_mask"] for sample in samples]),
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": neighbour_history_mask,
        "neighbour_signal_history": neighbour_signal_history,
        "map_rows": np.concatenate(
            [sample["map_rows"] for sample in samples], dtype=np.float32
        ),
        "map_dot_polyline_slot": np.concatenate(
            [
                sample["map_chunk_index"] + sample_index * max_chunks_in_batch
                for sample_index, sample in enumerate(samples)
            ],
            dtype=np.int64,
        ),
        "map_chunk_signal_history": map_chunk_signal_history,
        "max_polylines_in_batch": np.array(max_chunks_in_batch, dtype=np.int64),
    }
