"""Turns one Waymo Scenario proto into staged arrays of agent tracks, map dots
one metre apart and traffic-light histories.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import numpy as np

from womd import contract
from womd import frame_ops
from womd_protos import map_pb2, scenario_pb2

# One track in the scene frame, one entry per step.
SceneFrameTrack = namedtuple("SceneFrameTrack",
                             "positions headings velocities valid")

# One map feature resampled to dots in the scene frame.
SceneFrameFeature = namedtuple(
    "SceneFrameFeature",
    "points directions crossing_codes kind_index",
)

# The map features of a scenario, all dots in one array and then one
# entry per feature saying how many of those dots are its own.
StagedMap = namedtuple(
    "StagedMap",
    "map_rows feature_lengths feature_ids"
    " polyline_signal_histories feature_is_interpolating",
)


def scenario_scene_frame(
        scenario: scenario_pb2.Scenario) -> tuple[np.ndarray, float]:
    """Returns the scene frame's origin and heading, the SDC's world position
    and heading at the current step.
    """
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_state = sdc_track.states[contract.CURRENT_STEP_INDEX]
    assert (sdc_state.valid
           ), f"invalid SDC at current step, scenario {scenario.scenario_id}"
    origin = np.array([sdc_state.center_x, sdc_state.center_y])
    heading = sdc_state.heading
    return origin, heading


def track_to_scene_frame(track: scenario_pb2.Track, origin: np.ndarray,
                         heading: float) -> SceneFrameTrack:
    """Converts one track's raw world-frame positions, headings and velocities
    into the scene frame, step by step.
    """
    positions = np.array(
        [[state.center_x, state.center_y] for state in track.states])
    headings = np.array([state.heading for state in track.states])
    velocities = np.array(
        [[state.velocity_x, state.velocity_y] for state in track.states])
    step_valid = np.array([state.valid for state in track.states])

    scene_positions = frame_ops.positions_to_frame(positions, origin, heading)
    scene_headings = frame_ops.headings_to_frame(headings, heading)
    scene_velocities = frame_ops.directions_to_frame(velocities, heading)

    return SceneFrameTrack(scene_positions, scene_headings, scene_velocities,
                           step_valid)


def track_to_feature_rows(track: scenario_pb2.Track, origin: np.ndarray,
                          heading: float, *,
                          is_sdc: bool) -> tuple[np.ndarray, np.ndarray]:
    """Builds the (TOTAL_STEPS, AGENT_FEATURE_DIM) feature row array for one
    track in the scene frame, plus its per-step validity.
    """
    scene_track = track_to_scene_frame(track, origin, heading)
    dimensions = np.array([[state.length, state.width] for state in track.states
                          ])

    type_name = scenario_pb2.Track.ObjectType.Name(track.object_type)
    type_onehot = np.zeros(contract.NUM_AGENT_TYPES)
    if type_name in contract.AGENT_TYPES:
        type_onehot[contract.AGENT_TYPES.index(type_name)] = 1.0
    else:
        type_onehot[contract.AGENT_TYPES.index("TYPE_OTHER")] = 1.0
    type_rows = np.tile(type_onehot, (contract.TOTAL_STEPS, 1))

    track_rows = np.zeros((contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM))
    track_rows[:, contract.AGENT_POSITION] = scene_track.positions
    track_rows[:, contract.AGENT_HEADING_COSINE] = np.cos(scene_track.headings)
    track_rows[:, contract.AGENT_HEADING_SINE] = np.sin(scene_track.headings)
    track_rows[:, contract.AGENT_VELOCITY] = scene_track.velocities
    track_rows[:, contract.AGENT_DIMENSIONS] = dimensions
    track_rows[:, contract.AGENT_TYPE] = type_rows
    track_rows[:, contract.AGENT_IS_SDC] = 1.0 if is_sdc else 0.0

    return track_rows, scene_track.valid


def scenario_track_arrays(
        scenario: scenario_pb2.Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Stacks feature rows and validity for each track in the scenario into
    (num_tracks, TOTAL_STEPS, ...) arrays.
    """
    origin, heading = scenario_scene_frame(scenario)

    rows_of_every_track = []
    valid_of_every_track = []
    for track_index, track in enumerate(scenario.tracks):
        track_rows, step_valid = track_to_feature_rows(
            track,
            origin,
            heading,
            is_sdc=track_index == scenario.sdc_track_index,
        )
        rows_of_every_track.append(track_rows)
        valid_of_every_track.append(step_valid)

    return np.stack(rows_of_every_track), np.stack(valid_of_every_track)


def scenario_track_labels(
    scenario: scenario_pb2.Scenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-track labels, the track id, whether it is a designated prediction
    target and whether it is an object of interest.
    """
    track_ids = np.array([track.id for track in scenario.tracks],
                         dtype=np.int64)
    designated_indices = {
        required.track_index for required in scenario.tracks_to_predict
    }
    is_designated_target = np.array([
        track_index in designated_indices
        for track_index in range(len(scenario.tracks))
    ])
    interest_ids = set(scenario.objects_of_interest)
    is_object_of_interest = np.array(
        [track.id in interest_ids for track in scenario.tracks])
    return track_ids, is_designated_target, is_object_of_interest


# Map feature kinds whose points form a closed polygon rather
# than an open polyline.
MAP_POLYGON_KINDS = ("crosswalk", "speed_bump", "driveway")


def map_feature_points(
        feature: map_pb2.MapFeature) -> tuple[np.ndarray, int] | None:
    """One map feature's raw world-frame points and its kind index, or None if
    the feature has no usable geometry.
    """
    kind_name = feature.WhichOneof("feature_data")
    if kind_name is None:
        return None

    if kind_name == "stop_sign":
        raw_points = [feature.stop_sign.position]
    elif kind_name in MAP_POLYGON_KINDS:
        corners = getattr(feature, kind_name).polygon
        if len(corners) < 2:
            return None
        # Closes the polygon by repeating its first corner.
        raw_points = list(corners) + [corners[0]]
    else:
        raw_points = getattr(feature, kind_name).polyline
    if len(raw_points) == 0:
        return None

    world_points = np.array([[point.x, point.y] for point in raw_points])
    return world_points, contract.MAP_POLYLINE_KINDS.index(kind_name)


def polyline_arc_lengths(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a polyline, starting at 0 for the first
    point.
    """
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def polyline_sample_distances(arc_lengths: np.ndarray,
                              spacing: float) -> np.ndarray:
    """Arc length positions to resample at, evenly spaced plus the polyline's
    final endpoint.
    """
    return np.append(np.arange(0.0, arc_lengths[-1], spacing), arc_lengths[-1])


def column_along_polyline(points: np.ndarray, column_values: np.ndarray,
                          spacing: float) -> np.ndarray:
    """Resamples one column of values at fixed spacing along the polyline's arc
    length by linear interpolation.
    """
    if len(points) < 2:
        return column_values
    arc_lengths = polyline_arc_lengths(points)
    if arc_lengths[-1] == 0.0:
        return column_values[:1]
    return np.interp(
        polyline_sample_distances(arc_lengths, spacing),
        arc_lengths,
        column_values,
    )


def points_along_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resamples x and y together at fixed spacing along the polyline, producing
    one-metre-spaced map dots.
    """
    if len(points) < 2:
        return points
    return np.stack(
        [
            column_along_polyline(points, points[:, 0], spacing),
            column_along_polyline(points, points[:, 1], spacing),
        ],
        axis=1,
    )


def polyline_directions(points: np.ndarray) -> np.ndarray:
    """Unit direction vector per point along a polyline; the last point repeats
    the final segment's direction.
    """
    if len(points) < 2:
        return np.zeros_like(points)
    steps = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1, keepdims=True)
    zero_step_indices = np.flatnonzero(step_lengths[:, 0] == 0.0)
    # Avoids a divide by zero on coincident points; those
    # directions get overwritten from the previous step below.
    step_lengths[zero_step_indices] = 1.0
    directions = steps / step_lengths
    for zero_index in zero_step_indices:
        directions[zero_index] = directions[zero_index - 1]
    return np.concatenate([directions, directions[-1:]])


def map_feature_boundary_crossing_codes(feature: map_pb2.MapFeature,
                                        raw_points: np.ndarray,
                                        dot_count: int) -> np.ndarray:
    """Per-dot left and right boundary crossing codes for a lane feature, 0 for
    no boundary and otherwise 1 + the road line type.
    """
    crossing_codes = np.zeros((dot_count, 2))
    if feature.WhichOneof("feature_data") != "lane":
        return crossing_codes
    arc_lengths = polyline_arc_lengths(raw_points)
    sample_distances = polyline_sample_distances(
        arc_lengths, contract.MAP_POINT_SPACING_METRES)
    for side_index, side_name in enumerate(contract.LANE_SIDES):
        for segment in getattr(feature.lane, f"{side_name}_boundaries"):
            assert segment.boundary_type < len(contract.ROAD_LINE_TYPES), (
                f"lane {feature.id}: {side_name} boundary type"
                f" {segment.boundary_type} is not a known road"
                f" line type")
            # Maps the boundary segment's original lane-point
            # range onto the resampled dot indices by arc length.
            first_dot = np.searchsorted(
                sample_distances,
                arc_lengths[segment.lane_start_index],
                side="left",
            )
            last_dot = np.searchsorted(
                sample_distances,
                arc_lengths[segment.lane_end_index],
                side="right",
            )
            crossing_codes[first_dot:last_dot,
                           side_index] = (1.0 + segment.boundary_type)
    return crossing_codes


def map_feature_to_scene_frame(feature: map_pb2.MapFeature, origin: np.ndarray,
                               heading: float) -> SceneFrameFeature | None:
    """Resamples a map feature to one-metre spacing, derives directions and
    crossing codes, and crops it to the scene frame.
    """
    geometry = map_feature_points(feature)
    if geometry is None:
        return None
    raw_points, kind_index = geometry

    spaced_points = points_along_polyline(raw_points,
                                          contract.MAP_POINT_SPACING_METRES)
    directions = polyline_directions(spaced_points)
    crossing_codes = map_feature_boundary_crossing_codes(
        feature, raw_points, len(spaced_points))
    scene_points = frame_ops.positions_to_frame(spaced_points, origin, heading)
    scene_directions = frame_ops.directions_to_frame(directions, heading)

    inside_staging_radius = (np.linalg.norm(scene_points, axis=1)
                             <= contract.STAGING_CROP_RADIUS_METRES)
    return SceneFrameFeature(
        scene_points[inside_staging_radius],
        scene_directions[inside_staging_radius],
        crossing_codes[inside_staging_radius],
        kind_index,
    )


def scenario_traffic_signal_histories(
    scenario: scenario_pb2.Scenario
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per-lane one-hot signal state history over the history steps, and each
    signalled lane's stop point in world coordinates.
    """
    histories = {}
    stop_points = {}
    for step_index, dynamic_state in enumerate(
            scenario.dynamic_map_states[:contract.HISTORY_STEPS]):
        for lane_state in dynamic_state.lane_states:
            state_name = map_pb2.TrafficSignalLaneState.State.Name(
                lane_state.state)
            history = histories.setdefault(
                lane_state.lane,
                np.zeros((
                    contract.HISTORY_STEPS,
                    contract.NUM_TRAFFIC_SIGNAL_STATES,
                )),
            )
            history[
                step_index,
                contract.TRAFFIC_SIGNAL_STATES.index(state_name),
            ] = 1.0
            stop_points.setdefault(
                lane_state.lane,
                np.array([lane_state.stop_point.x, lane_state.stop_point.y]),
            )
    return histories, stop_points


def map_feature_rows(
        feature: map_pb2.MapFeature, origin: np.ndarray, heading: float,
        signal_histories: dict[int, np.ndarray],
        signal_stop_points: dict[int, np.ndarray]) -> np.ndarray | None:
    """Builds the per-dot feature row array for one map feature, including
    columns specific to its kind (lane, road line, ...).
    """
    scene_feature = map_feature_to_scene_frame(feature, origin, heading)
    if scene_feature is None or len(scene_feature.points) == 0:
        return None
    scene_points, scene_directions, crossing_codes, kind_index = scene_feature
    kind_name = contract.MAP_POLYLINE_KINDS[kind_index]

    dot_rows = np.zeros((len(scene_points), contract.MAP_FEATURE_DIM))
    dot_rows[:, contract.MAP_POSITION] = scene_points
    dot_rows[:, contract.MAP_DIRECTION] = scene_directions
    dot_rows[:, contract.MAP_KIND.start + kind_index] = 1.0
    dot_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING] = crossing_codes[:, 0]
    dot_rows[:, contract.MAP_RIGHT_BOUNDARY_CROSSING] = crossing_codes[:, 1]

    if kind_name == "lane":
        if feature.id in signal_histories:
            dot_rows[:, contract.MAP_STOP_POINT] = frame_ops.positions_to_frame(
                signal_stop_points[feature.id], origin, heading)
        assert feature.lane.type < contract.NUM_LANE_TYPES
        dot_rows[:, contract.MAP_LANE_TYPE.start + feature.lane.type] = 1.0
        dot_rows[:, contract.MAP_SPEED_LIMIT] = feature.lane.speed_limit_mph
    elif kind_name == "road_line":
        assert feature.road_line.type < len(contract.ROAD_LINE_TYPES)
        dot_rows[
            :,
            contract.MAP_BOUNDARY_TYPE.start + feature.road_line.type,
        ] = 1.0
    elif kind_name == "road_edge":
        assert feature.road_edge.type < len(contract.ROAD_EDGE_TYPES)
        first_edge_column = (contract.MAP_BOUNDARY_TYPE.start +
                             len(contract.ROAD_LINE_TYPES))
        dot_rows[:, first_edge_column + feature.road_edge.type] = 1.0

    return dot_rows


def map_feature_signal_history(
        feature: map_pb2.MapFeature,
        signal_histories: dict[int, np.ndarray]) -> np.ndarray:
    """A lane feature's signal history, or all zeros if it is not a lane or
    carries no signal.
    """
    if (feature.WhichOneof("feature_data") == "lane" and
            feature.id in signal_histories):
        return signal_histories[feature.id]
    return np.zeros(
        (contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES))


def map_feature_is_interpolating(feature: map_pb2.MapFeature) -> bool:
    return (feature.WhichOneof("feature_data") == "lane" and
            feature.lane.interpolating)


def scenario_map_arrays(scenario: scenario_pb2.Scenario) -> StagedMap:
    """Builds the concatenated map row array plus per-feature metadata for each
    map feature with usable geometry.
    """
    origin, heading = scenario_scene_frame(scenario)
    signal_histories, signal_stop_points = scenario_traffic_signal_histories(
        scenario)

    rows_of_every_feature = []
    feature_ids = []
    feature_is_interpolating = []
    feature_signal_histories = []
    for feature in scenario.map_features:
        dot_rows = map_feature_rows(
            feature,
            origin,
            heading,
            signal_histories,
            signal_stop_points,
        )
        if dot_rows is None:
            continue
        rows_of_every_feature.append(dot_rows)
        feature_ids.append(feature.id)
        feature_is_interpolating.append(map_feature_is_interpolating(feature))
        feature_signal_histories.append(
            map_feature_signal_history(feature, signal_histories))

    if not rows_of_every_feature:
        return StagedMap(
            np.zeros((0, contract.MAP_FEATURE_DIM)),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((
                0,
                contract.HISTORY_STEPS,
                contract.NUM_TRAFFIC_SIGNAL_STATES,
            )),
            np.zeros(0, dtype=bool),
        )

    map_rows = np.concatenate(rows_of_every_feature)
    feature_lengths = np.array(
        [len(dot_rows) for dot_rows in rows_of_every_feature],
        dtype=np.int64,
    )
    return StagedMap(
        map_rows,
        feature_lengths,
        np.array(feature_ids, dtype=np.int64),
        np.stack(feature_signal_histories),
        np.array(feature_is_interpolating, dtype=bool),
    )


def write_scenario(scenario: scenario_pb2.Scenario, output_path: Path) -> float:
    """Writes one scenario's staged .npz with its track and map arrays, labels,
    the scene frame and a provenance stamp.
    """
    assert scenario.current_time_index == contract.CURRENT_STEP_INDEX, (
        f"current_time_index {scenario.current_time_index},"
        f" scenario {scenario.scenario_id}")
    timestamp_gaps = np.diff(np.array(scenario.timestamps_seconds))
    worst_spacing_deviation = (float(
        np.max(np.abs(timestamp_gaps - contract.TIMESTEP_SECONDS)))
                               if len(timestamp_gaps) else 0.0)

    track_rows, track_valid = scenario_track_arrays(scenario)
    track_ids, is_designated_target, is_object_of_interest = (
        scenario_track_labels(scenario))
    staged_map = scenario_map_arrays(scenario)
    signal_histories = staged_map.polyline_signal_histories
    origin, heading = scenario_scene_frame(scenario)

    staged_arrays = {
        "track_rows": track_rows.astype(np.float32),
        "track_valid": track_valid,
        "track_ids": track_ids,
        "is_designated_target": is_designated_target,
        "is_object_of_interest": is_object_of_interest,
        "map_rows": staged_map.map_rows.astype(np.float32),
        "feature_lengths": staged_map.feature_lengths,
        "feature_ids": staged_map.feature_ids,
        "feature_is_interpolating": staged_map.feature_is_interpolating,
        "polyline_signal_histories": signal_histories.astype(np.float32),
        "frame_origin": origin.astype(np.float32),
        "frame_heading": np.float32(heading),
        "scenario_id": scenario.scenario_id,
        "provenance": contract.artifact_provenance("stage.py",
                                                   scenario.scenario_id),
    }

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    with open(partial_path, "wb") as partial_file:
        np.savez_compressed(partial_file, **staged_arrays)
    # Renaming from a .partial path makes the write atomic, so a killed
    # process never leaves a half-written file in place.
    partial_path.replace(output_path)
    return worst_spacing_deviation


# ------------------------------------------------------------------
# ONE STAGED .npz, in the scene frame. T tracks, F map features,
# D map dots, 91 steps (11 history + 80 future).
#
#   track_rows                 (T, 91, 13)  float32  agent rows
#   track_valid                (T, 91)      bool     step observed
#   track_ids                  (T,)         int64
#   is_designated_target       (T,)         bool     Waymo scores it
#   is_object_of_interest      (T,)         bool
#   map_rows                   (D, 32)      float32  dots, by feature
#   feature_lengths            (F,)         int64    dots per feature
#   feature_ids                (F,)         int64
#   feature_is_interpolating   (F,)         bool
#   polyline_signal_histories  (F, 11, 9)   float32  traffic light
#   frame_origin               (2,)         float32  scene in world
#   frame_heading              ()           float32
#   scenario_id, provenance    ()           str
#
# The 13 and 32 columns are drawn at the end of womd/loader.py.
# ------------------------------------------------------------------
