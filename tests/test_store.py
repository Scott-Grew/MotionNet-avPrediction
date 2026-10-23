"""Tests for the map rows and the scene-frame resampling in womd/store.py."""

import numpy as np

from womd import layout, store
from womd.frames import Frame
from womd_protos import map_pb2

WORLD_FRAME = Frame(np.zeros(2), 0.0)
NO_SIGNALS = store.TrafficSignals({}, {})


def polyline_feature(kind, points, feature_id=1):
    """A map feature of the given polyline kind through the given points."""
    feature = map_pb2.MapFeature()
    feature.id = feature_id
    for x, y in points:
        getattr(feature, kind).polyline.add(x=x, y=y)
    return feature


def polygon_feature(kind, corners, feature_id=1):
    """A map feature of the given polygon kind with the given corners."""
    feature = map_pb2.MapFeature()
    feature.id = feature_id
    for x, y in corners:
        getattr(feature, kind).polygon.add(x=x, y=y)
    return feature


def test_map_rows_layout_for_all_seven_kinds():
    """Every map kind fills its own one-hot column and only the detail columns
    that belong to it.
    """
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    lane = polyline_feature("lane", [(0.0, 0.0), (10.0, 0.0)], feature_id=7)
    lane.lane.type = 2
    lane.lane.speed_limit_mph = 45.0
    road_line = polyline_feature("road_line", [(0.0, 1.0), (10.0, 1.0)])
    road_line.road_line.type = 2
    road_edge = polyline_feature("road_edge", [(0.0, 2.0), (10.0, 2.0)])
    road_edge.road_edge.type = 1
    stop_sign = map_pb2.MapFeature()
    stop_sign.id = 2
    stop_sign.stop_sign.position.x = 5.0
    stop_sign.stop_sign.position.y = 5.0
    features = [
        lane,
        road_line,
        road_edge,
        stop_sign,
        polygon_feature("crosswalk", square),
        polygon_feature("speed_bump", square),
        polygon_feature("driveway", square),
    ]

    lane_signal_history = np.zeros(
        (layout.HISTORY_STEPS, layout.NUM_TRAFFIC_SIGNAL_STATES)
    )
    lane_signal_history[:, 6] = 1.0
    signals = store.TrafficSignals(
        {7: lane_signal_history}, {7: np.array([9.0, 1.0])}
    )
    stager = store.MapStager(WORLD_FRAME, signals)

    for kind_index, feature in enumerate(features):
        rows = stager.rows(feature)
        kind = layout.MAP_POLYLINE_KINDS[kind_index]

        assert rows is not None, kind
        assert rows.shape[1] == layout.MAP_FEATURE_DIM
        assert np.all(np.isfinite(rows))
        kind_block = rows[:, layout.MAP_KIND]
        assert np.all(kind_block.sum(axis=1) == 1.0)
        assert np.all(kind_block[:, kind_index] == 1.0)

        signal_history = stager.signal_history(feature)
        detail_block = rows[:, layout.MAP_KIND.stop :]
        hot_columns = (
            np.flatnonzero(detail_block.any(axis=0)) + layout.MAP_KIND.stop
        )
        if kind == "lane":
            assert np.all(signal_history == lane_signal_history)
            assert np.all(rows[:, layout.MAP_LANE_TYPE.start + 2] == 1.0)
            assert np.all(rows[:, layout.MAP_SPEED_LIMIT] == 45.0)
            assert np.all(rows[:, layout.MAP_BOUNDARY_TYPE] == 0.0)
            assert np.all(
                rows[:, layout.MAP_STOP_POINT] == signals.stop_points[7]
            )
        elif kind == "road_line":
            assert list(hot_columns) == [layout.MAP_BOUNDARY_TYPE.start + 2]
        elif kind == "road_edge":
            edge_column = (
                layout.MAP_BOUNDARY_TYPE.start + len(layout.ROAD_LINE_TYPES) + 1
            )
            assert list(hot_columns) == [edge_column]
        else:
            assert np.all(signal_history == 0.0)
            assert np.all(detail_block == 0.0)
            if kind == "stop_sign":
                assert len(rows) == 1
                assert np.all(rows[:, layout.MAP_DIRECTION] == 0.0)


def test_fill_bridges_measured_worst_case_shapes():
    """A 159.8 m two-point road edge and a 62 m corners-only crosswalk are both
    filled so no gap exceeds the dot spacing.
    """
    spacing = layout.MAP_POINT_SPACING_METRES
    endpoint_pair_edge = polyline_feature(
        "road_edge", [(0.0, 0.0), (159.8, 0.0)]
    )
    edge = store.feature_in_scene(endpoint_pair_edge, WORLD_FRAME)
    assert len(edge.points) == int(np.ceil(159.8 / spacing)) + 1
    edge_gaps = np.linalg.norm(np.diff(edge.points, axis=0), axis=1)
    assert np.all(edge_gaps <= spacing + 1e-9)
    assert np.allclose(np.linalg.norm(edge.directions, axis=1), 1.0)

    corners = [(0.0, 0.0), (62.0, 0.0), (62.0, 62.0), (0.0, 62.0)]
    corners_only_crosswalk = polygon_feature("crosswalk", corners)
    ring = store.feature_in_scene(corners_only_crosswalk, WORLD_FRAME)
    ring_gaps = np.linalg.norm(np.diff(ring.points, axis=0), axis=1)
    assert np.all(ring_gaps <= spacing + 1e-9)
    assert np.allclose(ring.points[0], ring.points[-1])


def test_crop_keeps_polyline_crossing_the_boundary():
    """A polyline that crosses the staging radius keeps its inside part and
    drops the rest.
    """
    crossing_edge = polyline_feature("road_edge", [(-300.0, 1.0), (300.0, 1.0)])
    scene = store.feature_in_scene(crossing_edge, WORLD_FRAME)
    assert len(scene.points) > 600.0 / layout.MAP_POINT_SPACING_METRES * 0.9
    assert len(scene.points) == len(scene.directions)
    distances = np.linalg.norm(scene.points, axis=1)
    assert np.all(distances <= layout.STAGING_CROP_RADIUS_METRES)


def test_crossing_codes_survive_resampling():
    """Boundary ranges arrive as raw point indices and must map onto the
    resampled dots.
    """
    lane = polyline_feature(
        "lane",
        [(0.0, 0.0), (10.0, 0.0), (30.0, 0.0), (60.0, 0.0)],
        feature_id=21,
    )
    left_boundary = lane.lane.left_boundaries.add()
    left_boundary.lane_start_index = 1
    left_boundary.lane_end_index = 2
    left_boundary.boundary_feature_id = 31
    left_boundary.boundary_type = 3

    rows = store.MapStager(WORLD_FRAME, NO_SIGNALS).rows(lane)
    left_codes = rows[:, layout.MAP_LEFT_BOUNDARY_CROSSING]

    assert len(rows) == int(60.0 / layout.MAP_POINT_SPACING_METRES) + 1
    assert np.flatnonzero(left_codes).tolist() == list(range(10, 31))
    assert np.all(left_codes[10:31] == 1 + 3)
    assert np.all(rows[:, layout.MAP_RIGHT_BOUNDARY_CROSSING] == 0.0)


def test_feature_row_layout_is_pinned():
    """The row widths the staged data was written with."""
    assert layout.MAP_FEATURE_DIM == 32
    assert layout.AGENT_FEATURE_DIM == 13
