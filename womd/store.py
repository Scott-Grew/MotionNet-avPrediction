import numpy as np

from womd import contract
from womd import frame_ops
from womd_protos import map_pb2, scenario_pb2

def scenario_storage_frame(scenario):
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_state = sdc_track.states[contract.CURRENT_STEP_INDEX]
    assert sdc_state.valid, f"invalid SDC at current step, scenario {scenario.scenario_id}"
    origin = np.array([sdc_state.center_x, sdc_state.center_y])
    heading = sdc_state.heading
    return origin, heading

def track_to_storage_frame(track, origin, heading):
    positions = np.array([[state.center_x, state.center_y] for state in track.states])
    headings = np.array([state.heading for state in track.states])
    velocities = np.array([[state.velocity_x, state.velocity_y] for state in track.states])
    valid = np.array([state.valid for state in track.states])

    stored_positions = frame_ops.positions_to_agent_frame(positions, origin, heading)
    stored_headings = frame_ops.headings_to_agent_frame(headings, heading)
    stored_velocities = frame_ops.directions_to_agent_frame(velocities, heading)

    return stored_positions, stored_headings, stored_velocities, valid

def track_to_feature_rows(track, origin, heading, is_sdc):
    positions, headings, velocities, valid = track_to_storage_frame(track, origin, heading)
    dimensions = np.array([[state.length, state.width] for state in track.states])

    type_name = scenario_pb2.Track.ObjectType.Name(track.object_type)
    type_onehot = np.zeros(contract.NUM_AGENT_TYPES)
    if type_name in contract.AGENT_TYPES:
        type_onehot[contract.AGENT_TYPES.index(type_name)] = 1.0
    else:
        type_onehot[contract.AGENT_TYPES.index("TYPE_OTHER")] = 1.0
    type_rows = np.tile(type_onehot, (contract.TOTAL_STEPS, 1))

    rows = np.zeros((contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM))
    rows[:, contract.AGENT_POSITION] = positions
    rows[:, contract.AGENT_HEADING_COSINE] = np.cos(headings)
    rows[:, contract.AGENT_HEADING_SINE] = np.sin(headings)
    rows[:, contract.AGENT_VELOCITY] = velocities
    rows[:, contract.AGENT_DIMENSIONS] = dimensions
    rows[:, contract.AGENT_TYPE] = type_rows
    rows[:, contract.AGENT_IS_SDC] = 1.0 if is_sdc else 0.0

    return rows, valid

def scenario_track_arrays(scenario):
    origin, heading = scenario_storage_frame(scenario)

    all_rows = []
    all_valid = []
    for track_index, track in enumerate(scenario.tracks):
        rows, valid = track_to_feature_rows(track, origin, heading, track_index 
                                            == scenario.sdc_track_index)
        all_rows.append(rows)
        all_valid.append(valid)

    return np.stack(all_rows), np.stack(all_valid)

def scenario_track_labels(scenario):
    track_ids = np.array([track.id for track in scenario.tracks], dtype=np.int64)
    designated_indices = {required.track_index for required in scenario.tracks_to_predict}
    is_designated_target = np.array(
        [track_index in designated_indices for track_index in range(len(scenario.tracks))]
    )
    interest_ids = set(scenario.objects_of_interest)
    is_object_of_interest = np.array([track.id in interest_ids for track in scenario.tracks])
    return track_ids, is_designated_target, is_object_of_interest

MAP_POLYGON_KINDS = ("crosswalk", "speed_bump", "driveway")

def map_feature_points(feature):
    kind = feature.WhichOneof("feature_data")
    if kind is None:
        return None, None
    if kind == "stop_sign":
        raw_points = [feature.stop_sign.position]
    elif kind in MAP_POLYGON_KINDS:
        corners = getattr(feature, kind).polygon
        if len(corners) < 2:
            return None, None
        raw_points = list(corners) + [corners[0]]
    else:
        raw_points = getattr(feature, kind).polyline
    if len(raw_points) == 0:
        return None, None
    points = np.array([[point.x, point.y] for point in raw_points])
    return points, contract.MAP_POLYLINE_KINDS.index(kind)

def polyline_arc_lengths(points):
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])

def polyline_sample_distances(arc_lengths, spacing):
    return np.append(np.arange(0.0, arc_lengths[-1], spacing), arc_lengths[-1])

def column_along_polyline(points, column_values, spacing):
    if len(points) < 2:
        return column_values
    arc_lengths = polyline_arc_lengths(points)
    if arc_lengths[-1] == 0.0:
        return column_values[:1]
    return np.interp(polyline_sample_distances(arc_lengths, spacing), arc_lengths, column_values)

def points_along_polyline(points, spacing):
    if len(points) < 2:
        return points
    return np.stack(
        [column_along_polyline(points, points[:, 0], spacing),
         column_along_polyline(points, points[:, 1], spacing)],
        axis=1,
    )

def polyline_directions(points):
    if len(points) < 2:
        return np.zeros_like(points)
    steps = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1, keepdims=True)
    zero_step_indices = np.flatnonzero(step_lengths[:, 0] == 0.0)
    step_lengths[zero_step_indices] = 1.0
    directions = steps / step_lengths
    for zero_index in zero_step_indices:
        directions[zero_index] = directions[zero_index - 1]
    return np.concatenate([directions, directions[-1:]])

def map_feature_boundary_crossing_codes(feature, raw_points, dot_count):
    crossing_codes = np.zeros((dot_count, 2))
    if feature.WhichOneof("feature_data") != "lane":
        return crossing_codes
    arc_lengths = polyline_arc_lengths(raw_points)
    sample_distances = polyline_sample_distances(arc_lengths, contract.MAP_POINT_SPACING_METRES)
    for side_index, side_name in enumerate(contract.LANE_SIDES):
        for segment in getattr(feature.lane, f"{side_name}_boundaries"):
            assert segment.boundary_type < len(contract.ROAD_LINE_TYPES), (
                f"boundary type {segment.boundary_type} on the {side_name} of lane {feature.id}"
                f" is past the {len(contract.ROAD_LINE_TYPES)} road line types a crossing code"
                f" encodes, so it would store a code the map dot encoder cannot one-hot"
            )
            first_dot = np.searchsorted(sample_distances, arc_lengths[segment.lane_start_index], side="left")
            last_dot = np.searchsorted(sample_distances, arc_lengths[segment.lane_end_index], side="right")
            crossing_codes[first_dot:last_dot, side_index] = 1.0 + segment.boundary_type
    return crossing_codes

def map_feature_to_storage_frame(feature, origin, heading):
    raw_points, kind_index = map_feature_points(feature)
    if raw_points is None:
        return None, None, None, None
    spaced_points = points_along_polyline(raw_points, contract.MAP_POINT_SPACING_METRES)
    arrows = polyline_directions(spaced_points)
    crossing_codes = map_feature_boundary_crossing_codes(feature, raw_points, len(spaced_points))
    stored_points = frame_ops.positions_to_agent_frame(spaced_points, origin, heading)
    stored_arrows = frame_ops.directions_to_agent_frame(arrows, heading)
    keep = np.linalg.norm(stored_points, axis=1) <= contract.STAGING_CROP_RADIUS_METRES
    return stored_points[keep], stored_arrows[keep], crossing_codes[keep], kind_index

def scenario_traffic_signal_histories(scenario):
    histories = {}
    stop_points = {}
    for step_index, dynamic_state in enumerate(scenario.dynamic_map_states[:contract.HISTORY_STEPS]):
        for lane_state in dynamic_state.lane_states:
            state_name = map_pb2.TrafficSignalLaneState.State.Name(lane_state.state)
            history = histories.setdefault(
                lane_state.lane,
                np.zeros((contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)),
            )
            history[step_index, contract.TRAFFIC_SIGNAL_STATES.index(state_name)] = 1.0
            stop_points.setdefault(
                lane_state.lane,
                np.array([lane_state.stop_point.x, lane_state.stop_point.y]),
            )
    return histories, stop_points

def map_feature_rows(feature, origin, heading, signal_histories, signal_stop_points):
    stored_points, stored_arrows, crossing_codes, kind_index = map_feature_to_storage_frame(
        feature, origin, heading
    )
    if stored_points is None or len(stored_points) == 0:
        return None
    kind = contract.MAP_POLYLINE_KINDS[kind_index]

    rows = np.zeros((len(stored_points), contract.MAP_FEATURE_DIM))
    rows[:, contract.MAP_POSITION] = stored_points
    rows[:, contract.MAP_DIRECTION] = stored_arrows
    rows[:, contract.MAP_KIND.start + kind_index] = 1.0
    rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING] = crossing_codes[:, 0]
    rows[:, contract.MAP_RIGHT_BOUNDARY_CROSSING] = crossing_codes[:, 1]

    if kind == "lane":
        if feature.id in signal_histories:
            rows[:, contract.MAP_STOP_POINT] = frame_ops.positions_to_agent_frame(
                signal_stop_points[feature.id], origin, heading
            )
        assert feature.lane.type < contract.NUM_LANE_TYPES
        rows[:, contract.MAP_LANE_TYPE.start + feature.lane.type] = 1.0
        rows[:, contract.MAP_SPEED_LIMIT] = feature.lane.speed_limit_mph
    elif kind == "road_line":
        assert feature.road_line.type < len(contract.ROAD_LINE_TYPES)
        rows[:, contract.MAP_BOUNDARY_TYPE.start + feature.road_line.type] = 1.0
    elif kind == "road_edge":
        assert feature.road_edge.type < len(contract.ROAD_EDGE_TYPES)
        rows[:, contract.MAP_BOUNDARY_TYPE.start + len(contract.ROAD_LINE_TYPES) + feature.road_edge.type] = 1.0

    return rows

def map_feature_signal_history(feature, signal_histories):
    if feature.WhichOneof("feature_data") == "lane" and feature.id in signal_histories:
        return signal_histories[feature.id]
    return np.zeros((contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES))

def map_feature_is_interpolating(feature):
    return feature.WhichOneof("feature_data") == "lane" and feature.lane.interpolating

def scenario_map_arrays(scenario):
    origin, heading = scenario_storage_frame(scenario)
    signal_histories, signal_stop_points = scenario_traffic_signal_histories(scenario)

    feature_tables = []
    feature_ids = []
    feature_is_interpolating = []
    feature_signal_histories = []
    for feature in scenario.map_features:
        rows = map_feature_rows(feature, origin, heading, signal_histories, signal_stop_points)
        if rows is None:
            continue
        feature_tables.append(rows)
        feature_ids.append(feature.id)
        feature_is_interpolating.append(map_feature_is_interpolating(feature))
        feature_signal_histories.append(map_feature_signal_history(feature, signal_histories))

    if not feature_tables:
        return (
            np.zeros((0, contract.MAP_FEATURE_DIM)),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)),
            np.zeros(0, dtype=bool),
        )

    map_rows = np.concatenate(feature_tables)
    feature_lengths = np.array([len(table) for table in feature_tables], dtype=np.int64)
    return (
        map_rows,
        feature_lengths,
        np.array(feature_ids, dtype=np.int64),
        np.stack(feature_signal_histories),
        np.array(feature_is_interpolating, dtype=bool),
    )

def write_scenario(scenario, output_path):
    assert scenario.current_time_index == contract.CURRENT_STEP_INDEX, (
        f"current_time_index {scenario.current_time_index}, scenario {scenario.scenario_id}"
    )
    timestamp_gaps = np.diff(np.array(scenario.timestamps_seconds))
    worst_spacing_deviation = float(np.max(np.abs(timestamp_gaps - 0.1))) if len(timestamp_gaps) else 0.0
    track_rows, track_valid = scenario_track_arrays(scenario)
    track_ids, is_designated_target, is_object_of_interest = scenario_track_labels(scenario)
    (map_rows, feature_lengths, feature_ids, polyline_signal_histories,
     feature_is_interpolating) = scenario_map_arrays(scenario)
    origin, heading = scenario_storage_frame(scenario)

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    with open(partial_path, "wb") as partial_file:
        np.savez_compressed(
            partial_file,
            track_rows=track_rows.astype(np.float32),
            track_valid=track_valid,
            track_ids=track_ids,
            is_designated_target=is_designated_target,
            is_object_of_interest=is_object_of_interest,
            map_rows=map_rows.astype(np.float32),
            feature_lengths=feature_lengths,
            feature_ids=feature_ids,
            feature_is_interpolating=feature_is_interpolating,
            polyline_signal_histories=polyline_signal_histories.astype(np.float32),
            frame_origin=origin.astype(np.float32),
            frame_heading=np.float32(heading),
            scenario_id=scenario.scenario_id,
            provenance=contract.artifact_provenance("stage.py", scenario.scenario_id),
        )
    partial_path.replace(output_path)
    return worst_spacing_deviation