import numpy as np

from womd import contract, frame_ops

BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5


def inside_crop(agent_frame_points, base_radius, forward_stretch):
    x = agent_frame_points[:, 0]
    y = agent_frame_points[:, 1]
    forward = (x / (base_radius * forward_stretch)) ** 2 + (
        y / base_radius
    ) ** 2 <= 1.0
    rear = (x / base_radius) ** 2 + (y / base_radius) ** 2 <= 1.0
    return np.where(x > 0.0, forward, rear)


def eligible_track_indices(
    track_rows,
    track_valid,
    is_designated_target,
    designated_targets_only,
):
    now_valid = track_valid[:, contract.CURRENT_STEP_INDEX]
    predicted_type = (
        track_rows[
            :, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE
        ][:, : contract.NUM_OBJECT_TYPES].sum(axis=1)
        > 0
    )
    selected = now_valid & predicted_type
    if designated_targets_only:
        selected = selected & is_designated_target
    return np.flatnonzero(selected)


def sample_frame(track_rows, track_index):
    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    origin = now_row[contract.AGENT_POSITION]
    heading = np.arctan2(
        now_row[contract.AGENT_HEADING_SINE],
        now_row[contract.AGENT_HEADING_COSINE],
    )
    return origin, heading


def track_rows_to_agent_frame(rows, origin, heading):
    reframed = rows.copy()
    reframed[..., contract.AGENT_POSITION] = (
        frame_ops.positions_to_agent_frame(
            rows[..., contract.AGENT_POSITION], origin, heading
        )
    )
    reframed[..., contract.AGENT_VELOCITY] = (
        frame_ops.directions_to_agent_frame(
            rows[..., contract.AGENT_VELOCITY], heading
        )
    )
    heading_cosine = rows[..., contract.AGENT_HEADING_COSINE]
    heading_sine = rows[..., contract.AGENT_HEADING_SINE]
    rotation_cosine, rotation_sine = np.cos(heading), np.sin(heading)
    reframed[..., contract.AGENT_HEADING_COSINE] = (
        heading_cosine * rotation_cosine
        + heading_sine * rotation_sine
    )
    reframed[..., contract.AGENT_HEADING_SINE] = (
        heading_sine * rotation_cosine
        - heading_cosine * rotation_sine
    )
    return reframed


def nearest_lane_dot_facing_the_agent_way(
    lane_dot_rows, agent_distances, agent_heading_cosine_sine
):
    faces_the_agent_way = (
        agent_heading_cosine_sine
        @ lane_dot_rows[:, contract.MAP_DIRECTION].T
        > 0.0
    )
    candidates = faces_the_agent_way | ~faces_the_agent_way.any(
        axis=1, keepdims=True
    )
    return np.where(candidates, agent_distances, np.inf).argmin(
        axis=1
    )


def lane_dots_of_scenario(scenario_array):
    map_rows = scenario_array["map_rows"]
    dot_polyline_index = scenario_array["map_dot_polyline_index"]
    lane_kind_column = (
        contract.MAP_KIND.start
        + contract.MAP_POLYLINE_KINDS.index("lane")
    )
    lane_dot_indices = np.flatnonzero(
        map_rows[:, lane_kind_column] == 1.0
    )
    return (
        map_rows[lane_dot_indices],
        dot_polyline_index[lane_dot_indices],
    )


def assigned_lane_signal_histories(
    lane_dot_rows,
    polyline_row_of_lane_dot,
    polyline_signal_histories,
    agent_now_rows,
    agent_has_now_step,
):
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
    offsets_x = (
        lane_dot_positions[None, :, 0] - assignable_positions[:, :1]
    )
    offsets_y = (
        lane_dot_positions[None, :, 1] - assignable_positions[:, 1:]
    )
    agent_distances = np.sqrt(
        offsets_x * offsets_x + offsets_y * offsets_y
    )
    assigned_lane_dot = nearest_lane_dot_facing_the_agent_way(
        lane_dot_rows,
        agent_distances,
        assignable_rows[
            :,
            contract.AGENT_HEADING_COSINE : contract.AGENT_HEADING_SINE
            + 1,
        ],
    )
    signal_histories[assignable] = polyline_signal_histories[
        polyline_row_of_lane_dot[assigned_lane_dot]
    ]
    return signal_histories


def with_derived_arrays(scenario_array):
    feature_lengths = scenario_array["feature_lengths"]
    scenario_array["map_dot_polyline_index"] = np.repeat(
        np.arange(len(feature_lengths)), feature_lengths
    )
    lane_dot_rows, polyline_row_of_lane_dot = lane_dots_of_scenario(
        scenario_array
    )
    scenario_array["track_signal_histories"] = (
        assigned_lane_signal_histories(
            lane_dot_rows,
            polyline_row_of_lane_dot,
            scenario_array["polyline_signal_histories"],
            scenario_array["track_rows"][
                :, contract.CURRENT_STEP_INDEX
            ],
            scenario_array["track_valid"][
                :, contract.CURRENT_STEP_INDEX
            ],
        )
    )
    return scenario_array


def read_scenario(scenario_path):
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
        f"feature_lengths sum to {feature_lengths.sum()} but {scenario_path} holds"
        f" {map_row_count} map rows"
    )
    return with_derived_arrays(scenario_array)


def chunk_index_of_dots(dot_polyline_index):
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
    position_within_polyline = (
        np.arange(len(compact_polyline_index))
        - first_dot_position[compact_polyline_index]
    )
    chunks_per_polyline = (
        dots_per_polyline + contract.MAP_CHUNK_DOTS - 1
    ) // contract.MAP_CHUNK_DOTS
    first_chunk_of_polyline = (
        np.cumsum(chunks_per_polyline) - chunks_per_polyline
    )
    dot_chunk_index = (
        first_chunk_of_polyline[compact_polyline_index]
        + position_within_polyline // contract.MAP_CHUNK_DOTS
    )
    return dot_chunk_index, np.repeat(polylines, chunks_per_polyline)


def poses_in_agent_frame(
    positions, direction_cosine_sine, origin, heading
):
    return np.concatenate(
        [
            frame_ops.positions_to_agent_frame(
                positions, origin, heading
            ),
            frame_ops.directions_to_agent_frame(
                direction_cosine_sine, heading
            ),
        ],
        axis=1,
    ).astype(np.float32)


def build_scene_sample(scenario_array, track_indices):
    track_rows = scenario_array["track_rows"]
    track_valid = scenario_array["track_valid"]
    track_signal_histories = scenario_array["track_signal_histories"]
    map_rows = scenario_array["map_rows"]
    history_valid = track_valid[:, : contract.HISTORY_STEPS]
    agent_present = history_valid.any(axis=1)
    last_valid_step = (
        contract.HISTORY_STEPS
        - 1
        - history_valid[:, ::-1].argmax(axis=1)
    )
    agent_reference_rows = track_rows[
        np.arange(len(track_rows)), last_valid_step
    ]
    agent_reference_directions = np.stack(
        [
            agent_reference_rows[:, contract.AGENT_HEADING_COSINE],
            agent_reference_rows[:, contract.AGENT_HEADING_SINE],
        ],
        axis=1,
    )

    dot_chunk_index, chunk_polyline = chunk_index_of_dots(
        scenario_array["map_dot_polyline_index"]
    )
    chunk_count = len(chunk_polyline)
    _, first_dot_of_chunk, dots_per_chunk = np.unique(
        dot_chunk_index, return_index=True, return_counts=True
    )
    chunk_reference_rows = map_rows[
        first_dot_of_chunk + dots_per_chunk // 2
    ]

    targets = []
    for track_index in track_indices:
        origin, heading = sample_frame(track_rows, track_index)
        agent_track = track_rows_to_agent_frame(
            track_rows[track_index], origin, heading
        )
        now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
        speed = float(
            np.linalg.norm(now_row[contract.AGENT_VELOCITY])
        )
        dot_inside_crop = inside_crop(
            frame_ops.positions_to_agent_frame(
                map_rows[:, contract.MAP_POSITION], origin, heading
            ),
            BASE_RADIUS_METRES,
            1.0 + STRETCH_GAIN * speed,
        )
        chunk_visible = (
            np.bincount(
                dot_chunk_index,
                weights=dot_inside_crop,
                minlength=chunk_count,
            )
            > 0
        )
        agent_visible = agent_present.copy()
        agent_visible[track_index] = False
        targets.append(
            {
                "agent_history": agent_track[
                    : contract.HISTORY_STEPS
                ],
                "agent_history_mask": history_valid[track_index],
                "agent_signal_history": track_signal_histories[
                    track_index
                ],
                "future_positions": agent_track[
                    contract.CURRENT_STEP_INDEX + 1 :,
                    contract.AGENT_POSITION,
                ],
                "future_mask": track_valid[
                    track_index, contract.CURRENT_STEP_INDEX + 1 :
                ],
                "agent_token_visible": agent_visible,
                "chunk_token_visible": chunk_visible,
                "agent_token_pose": poses_in_agent_frame(
                    agent_reference_rows[:, contract.AGENT_POSITION],
                    agent_reference_directions,
                    origin,
                    heading,
                ),
                "chunk_token_pose": poses_in_agent_frame(
                    chunk_reference_rows[:, contract.MAP_POSITION],
                    chunk_reference_rows[:, contract.MAP_DIRECTION],
                    origin,
                    heading,
                ),
                "frame_origin": origin,
                "frame_heading": heading,
                "track_id": scenario_array["track_ids"][track_index],
            }
        )
    return {
        "scenario_id": scenario_array["scenario_id"],
        "scene_agent_history": track_rows[
            :, : contract.HISTORY_STEPS
        ],
        "scene_agent_history_mask": history_valid,
        "scene_agent_signal_history": track_signal_histories,
        "map_rows": map_rows,
        "map_chunk_index": dot_chunk_index.astype(np.int64),
        "map_chunk_signal_history": scenario_array[
            "polyline_signal_histories"
        ][chunk_polyline],
        "targets": targets,
    }


def padded_stack(arrays, length, dtype):
    stacked = np.zeros(
        (len(arrays), length) + arrays[0].shape[1:], dtype=dtype
    )
    for row, array in zip(stacked, arrays):
        row[: len(array)] = array
    return stacked


def build_scene_batch(scene_samples):
    max_agents = max(
        len(scene["scene_agent_history"]) for scene in scene_samples
    )
    max_chunks = max(
        len(scene["map_chunk_signal_history"])
        for scene in scene_samples
    )
    targets = [
        (scene_index, target)
        for scene_index, scene in enumerate(scene_samples)
        for target in scene["targets"]
    ]

    def stacked_target(name):
        return np.stack([target[name] for _, target in targets])

    def padded_target_tokens(agent_name, chunk_name, dtype):
        return np.concatenate(
            [
                padded_stack(
                    [target[agent_name] for _, target in targets],
                    max_agents,
                    dtype,
                ),
                padded_stack(
                    [target[chunk_name] for _, target in targets],
                    max_chunks,
                    dtype,
                ),
            ],
            axis=1,
        )

    return {
        "scene_agent_history": padded_stack(
            [scene["scene_agent_history"] for scene in scene_samples],
            max_agents,
            np.float32,
        ),
        "scene_agent_history_mask": padded_stack(
            [
                scene["scene_agent_history_mask"]
                for scene in scene_samples
            ],
            max_agents,
            bool,
        ),
        "scene_agent_signal_history": padded_stack(
            [
                scene["scene_agent_signal_history"]
                for scene in scene_samples
            ],
            max_agents,
            np.float32,
        ),
        "map_rows": np.concatenate(
            [scene["map_rows"] for scene in scene_samples],
            dtype=np.float32,
        ),
        "map_dot_polyline_slot": np.concatenate(
            [
                scene["map_chunk_index"] + scene_index * max_chunks
                for scene_index, scene in enumerate(scene_samples)
            ],
            dtype=np.int64,
        ),
        "map_chunk_signal_history": padded_stack(
            [
                scene["map_chunk_signal_history"]
                for scene in scene_samples
            ],
            max_chunks,
            np.float32,
        ),
        "target_scene_index": np.array(
            [scene_index for scene_index, _ in targets],
            dtype=np.int64,
        ),
        "agent_history": stacked_target("agent_history"),
        "agent_history_mask": stacked_target("agent_history_mask"),
        "agent_signal_history": stacked_target(
            "agent_signal_history"
        ),
        "future_positions": stacked_target("future_positions"),
        "future_mask": stacked_target("future_mask"),
        "token_visible": padded_target_tokens(
            "agent_token_visible", "chunk_token_visible", bool
        ),
        "token_pose": padded_target_tokens(
            "agent_token_pose", "chunk_token_pose", np.float32
        ),
    }
