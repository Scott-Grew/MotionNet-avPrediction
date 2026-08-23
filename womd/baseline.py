import math

import numpy as np
import torch

from womd import contract, frame_ops, loader, model

FUTURE_HORIZON_SECONDS = 8.0
TIMESTEP_SECONDS = FUTURE_HORIZON_SECONDS / contract.FUTURE_STEPS

def future_elapsed_seconds(device, dtype):
    return TIMESTEP_SECONDS * torch.arange(1, contract.FUTURE_STEPS + 1, device=device, dtype=dtype)

def current_state(batch):
    now_row = batch["agent_history"][:, contract.CURRENT_STEP_INDEX]
    heading = torch.atan2(now_row[:, contract.AGENT_HEADING_SINE], now_row[:, contract.AGENT_HEADING_COSINE])
    return now_row[:, contract.AGENT_POSITION], heading, now_row[:, contract.AGENT_VELOCITY]

def as_single_mode(trajectories):
    confidence_logits = torch.zeros(
        trajectories.shape[0], 1, device=trajectories.device, dtype=trajectories.dtype
    )
    return trajectories.unsqueeze(1), confidence_logits

def constant_velocity(batch):
    position, _, velocity = current_state(batch)
    elapsed_seconds = future_elapsed_seconds(position.device, position.dtype)
    return as_single_mode(position[:, None, :] + velocity[:, None, :] * elapsed_seconds[None, :, None])

def observed_yaw_rate(batch):
    agent_history = batch["agent_history"]
    step_indices = torch.arange(contract.HISTORY_STEPS, device=agent_history.device)
    valid_step_indices = torch.where(
        batch["agent_history_mask"], step_indices, torch.full_like(step_indices, contract.HISTORY_STEPS)
    )
    earliest_valid_step = valid_step_indices.min(dim=1).values
    earliest_row = agent_history.gather(
        1, earliest_valid_step[:, None, None].expand(-1, 1, contract.AGENT_FEATURE_DIM)
    ).squeeze(1)

    now_row = agent_history[:, contract.CURRENT_STEP_INDEX]
    change_sine = (
        now_row[:, contract.AGENT_HEADING_SINE] * earliest_row[:, contract.AGENT_HEADING_COSINE]
        - now_row[:, contract.AGENT_HEADING_COSINE] * earliest_row[:, contract.AGENT_HEADING_SINE]
    )
    change_cosine = (
        now_row[:, contract.AGENT_HEADING_COSINE] * earliest_row[:, contract.AGENT_HEADING_COSINE]
        + now_row[:, contract.AGENT_HEADING_SINE] * earliest_row[:, contract.AGENT_HEADING_SINE]
    )
    observed_seconds = (contract.CURRENT_STEP_INDEX - earliest_valid_step) * TIMESTEP_SECONDS
    return torch.atan2(change_sine, change_cosine) / observed_seconds.clamp(min=TIMESTEP_SECONDS)

def constant_turn_rate_and_velocity(batch):
    position, heading, velocity = current_state(batch)
    heading_direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    forward_speed = (velocity * heading_direction).sum(dim=-1)

    elapsed_seconds = future_elapsed_seconds(position.device, position.dtype)
    turn_angle = observed_yaw_rate(batch)[:, None] * elapsed_seconds[None, :]
    chord_to_arc_ratio = torch.sinc(turn_angle / (2.0 * math.pi))
    chord_length = forward_speed[:, None] * elapsed_seconds[None, :] * chord_to_arc_ratio
    chord_direction = heading[:, None] + 0.5 * turn_angle

    displacement = torch.stack(
        [chord_length * torch.cos(chord_direction), chord_length * torch.sin(chord_direction)], dim=-1
    )
    return as_single_mode(position[:, None, :] + displacement)

def lane_polylines_by_id(scenario_array):
    map_rows = scenario_array["map_rows"]
    feature_lengths = scenario_array["feature_lengths"]
    feature_ids = scenario_array["feature_ids"]
    lane_kind_column = contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")
    first_dot_of_polyline = np.cumsum(feature_lengths) - feature_lengths
    polyline_is_lane = map_rows[first_dot_of_polyline, lane_kind_column] == 1.0

    lanes = {}
    for polyline_row in np.flatnonzero(polyline_is_lane):
        first_dot = int(first_dot_of_polyline[polyline_row])
        dot_count = int(feature_lengths[polyline_row])
        positions = map_rows[first_dot:first_dot + dot_count, contract.MAP_POSITION]
        arc_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        lanes[int(feature_ids[polyline_row])] = (first_dot, dot_count, arc_length)
    return lanes

def nearest_lane_dot(map_rows, lane_dot_indices, agent_position, agent_heading_cosine_sine):
    lane_dot_rows = map_rows[lane_dot_indices]
    agent_distances = np.linalg.norm(
        lane_dot_rows[:, contract.MAP_POSITION] - agent_position, axis=1
    )
    return int(lane_dot_indices[loader.nearest_lane_dot_facing_the_agent_way(
        lane_dot_rows, agent_distances[None], agent_heading_cosine_sine[None]
    )[0]])

def exit_lanes_in_connection_order(lane_connections):
    exits_of_lane = {}
    for source_id, destination_id, _ in lane_connections.tolist():
        onward = exits_of_lane.setdefault(source_id, [])
        if destination_id not in onward:
            onward.append(destination_id)
    return exits_of_lane

def routes_onward(lane_id, remaining_metres, lanes_already_taken, exits_of_lane, lanes):
    onward_lane_ids = [
        destination_id
        for destination_id in exits_of_lane.get(lane_id, ())
        if destination_id in lanes and destination_id not in lanes_already_taken
    ]
    if remaining_metres <= 0.0 or not onward_lane_ids:
        return [[]]

    routes = []
    for destination_id in onward_lane_ids:
        for continuation in routes_onward(
            destination_id,
            remaining_metres - lanes[destination_id][2],
            lanes_already_taken | {destination_id},
            exits_of_lane,
            lanes,
        ):
            routes.append([destination_id] + continuation)
            if len(routes) == contract.NUM_PREDICTED_MODES:
                return routes
    return routes

def positions_along_polyline(polyline_positions, final_direction, travelled_metres):
    arc_length = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(polyline_positions, axis=0), axis=1))]
    )
    sampled = np.stack(
        [np.interp(travelled_metres, arc_length, polyline_positions[:, axis]) for axis in (0, 1)],
        axis=1,
    )
    beyond_end = travelled_metres > arc_length[-1]
    sampled[beyond_end] = (
        polyline_positions[-1]
        + (travelled_metres[beyond_end] - arc_length[-1])[:, None] * final_direction
    )
    return sampled

def as_repeated_modes(storage_frame_trajectories, origin, heading):
    agent_frame = frame_ops.positions_to_agent_frame(storage_frame_trajectories, origin, heading)
    filled = np.arange(contract.NUM_PREDICTED_MODES) % len(agent_frame)
    return agent_frame[filled].astype(np.float32)

def follow_the_lane_predictions(scenario_array, track_index):
    track_rows = scenario_array["track_rows"]
    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    origin, heading = loader.sample_frame(track_rows, track_index)
    elapsed_seconds = TIMESTEP_SECONDS * np.arange(1, contract.FUTURE_STEPS + 1)
    speed = float(np.linalg.norm(now_row[contract.AGENT_VELOCITY]))

    lanes = lane_polylines_by_id(scenario_array)
    if not lanes:
        straight_line = (
            now_row[contract.AGENT_POSITION]
            + now_row[contract.AGENT_VELOCITY] * elapsed_seconds[:, None]
        )
        return as_repeated_modes(straight_line[None], origin, heading), False

    map_rows = scenario_array["map_rows"]
    lane_dot_indices = np.concatenate(
        [np.arange(first_dot, first_dot + dot_count) for first_dot, dot_count, _ in lanes.values()]
    )
    start_dot = nearest_lane_dot(
        map_rows,
        lane_dot_indices,
        now_row[contract.AGENT_POSITION],
        now_row[contract.AGENT_HEADING_COSINE:contract.AGENT_HEADING_SINE + 1],
    )
    start_lane_id = int(
        scenario_array["feature_ids"][scenario_array["map_dot_polyline_index"][start_dot]]
    )
    first_dot, dot_count, _ = lanes[start_lane_id]
    start_lane_dots = np.arange(start_dot, first_dot + dot_count)
    metres_left_on_start_lane = float(
        np.linalg.norm(
            np.diff(map_rows[start_lane_dots][:, contract.MAP_POSITION], axis=0), axis=1
        ).sum()
    )

    routes = routes_onward(
        start_lane_id,
        speed * FUTURE_HORIZON_SECONDS - metres_left_on_start_lane,
        {start_lane_id},
        exit_lanes_in_connection_order(scenario_array["lane_connections"]),
        lanes,
    )

    travelled_metres = speed * elapsed_seconds
    route_trajectories = []
    for route in routes:
        route_dots = np.concatenate(
            [start_lane_dots]
            + [np.arange(lanes[lane_id][0], lanes[lane_id][0] + lanes[lane_id][1]) for lane_id in route]
        )
        route_trajectories.append(
            positions_along_polyline(
                map_rows[route_dots][:, contract.MAP_POSITION],
                map_rows[route_dots[-1], contract.MAP_DIRECTION],
                travelled_metres,
            )
        )
    return as_repeated_modes(np.stack(route_trajectories), origin, heading), True

def straight_lines_to_most_used_anchors(batch, unit_anchors):
    predicted_type_index = model.predicted_type_index(batch["agent_history"])
    endpoints = unit_anchors[predicted_type_index][
        :, : contract.NUM_PREDICTED_MODES
    ].to(batch["agent_history"].dtype)
    elapsed_seconds = future_elapsed_seconds(endpoints.device, endpoints.dtype)
    trajectories = (
        endpoints[:, :, None, :]
        * (elapsed_seconds / FUTURE_HORIZON_SECONDS)[None, None, :, None]
    )
    return trajectories, torch.zeros(
        trajectories.shape[:2], device=trajectories.device, dtype=trajectories.dtype
    )
