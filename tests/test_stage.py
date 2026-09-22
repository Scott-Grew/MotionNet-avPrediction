"""Tests for staging, covering file naming, duplicate scenario ids and the
pinned layout of a staged file.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow

import stage
from womd import contract
from womd_protos import map_pb2, scenario_pb2

STAGING_PIN_PATH = Path(__file__).parent / "staging_pinned.json"


def add_track(scenario, track_id, object_type, base_x, base_y):
    """Adds a fully valid track that moves 0.1 m per step along x."""
    track = scenario.tracks.add()
    track.id = track_id
    track.object_type = object_type
    for step_index in range(contract.TOTAL_STEPS):
        state = track.states.add()
        state.center_x = base_x + 0.1 * step_index
        state.center_y = base_y
        state.heading = 0.0
        state.velocity_x = 1.0
        state.velocity_y = 0.0
        state.length = 4.0
        state.width = 2.0
        state.valid = True


def build_fixture_scenario(scenario_id):
    """A small scenario with three tracks, two neighbouring lanes, a road edge,
    a stop sign and one traffic signal.
    """
    scenario = scenario_pb2.Scenario()
    scenario.scenario_id = scenario_id
    scenario.current_time_index = contract.CURRENT_STEP_INDEX
    scenario.sdc_track_index = 0
    add_track(scenario, 100, scenario_pb2.Track.TYPE_VEHICLE, 0.0, 0.0)
    add_track(scenario, 200, scenario_pb2.Track.TYPE_PEDESTRIAN, 5.0, 5.0)
    add_track(scenario, 300, scenario_pb2.Track.TYPE_OTHER, -5.0, 2.0)
    scenario.tracks_to_predict.add().track_index = 1
    scenario.objects_of_interest.append(300)

    lane = scenario.map_features.add()
    lane.id = 7
    for x in (0.0, 10.0):
        lane.lane.polyline.add(x=x, y=1.0)
    lane.lane.type = 2
    lane.lane.speed_limit_mph = 45.0
    lane.lane.interpolating = True
    lane.lane.entry_lanes.append(6)
    lane.lane.exit_lanes.append(9)
    left_boundary = lane.lane.left_boundaries.add()
    left_boundary.lane_start_index = 0
    left_boundary.lane_end_index = 1
    left_boundary.boundary_feature_id = 8
    left_boundary.boundary_type = 2

    neighbour_lane = scenario.map_features.add()
    neighbour_lane.id = 9
    for x in (0.0, 10.0):
        neighbour_lane.lane.polyline.add(x=x, y=4.0)
    neighbour_lane.lane.type = 2
    right_neighbour = lane.lane.right_neighbors.add()
    right_neighbour.feature_id = 9
    right_neighbour.self_start_index = 0
    right_neighbour.self_end_index = 1
    right_neighbour.neighbor_start_index = 0
    right_neighbour.neighbor_end_index = 1
    shared_boundary = right_neighbour.boundaries.add()
    shared_boundary.lane_start_index = 0
    shared_boundary.lane_end_index = 1
    shared_boundary.boundary_feature_id = 8
    shared_boundary.boundary_type = 1

    road_edge = scenario.map_features.add()
    road_edge.id = 8
    for x in (0.0, 50.0):
        road_edge.road_edge.polyline.add(x=x, y=-5.0)
    road_edge.road_edge.type = 1

    stop_sign = scenario.map_features.add()
    stop_sign.id = 10
    stop_sign.stop_sign.position.x = 8.0
    stop_sign.stop_sign.position.y = 1.0
    stop_sign.stop_sign.lane.append(7)

    lane_state = scenario.dynamic_map_states.add().lane_states.add()
    lane_state.lane = 7
    lane_state.state = map_pb2.TrafficSignalLaneState.LANE_STATE_GO
    lane_state.stop_point.x = 9.0
    lane_state.stop_point.y = 1.0
    return scenario


def write_shard(shard_path, scenarios):
    """Writes scenarios to a TFRecord shard the way Waymo ships them."""
    with tensorflow.io.TFRecordWriter(str(shard_path)) as writer:
        for scenario in scenarios:
            writer.write(scenario.SerializeToString())


def stage_fixture(tmp_path):
    shard_path = tmp_path / "training.tfrecord-00000-of-01000"
    write_shard(shard_path, [build_fixture_scenario("pin00001")])
    output_directory = tmp_path / "staged"
    output_directory.mkdir()
    written_paths, _ = stage.stage_shards([shard_path], output_directory)
    return np.load(written_paths[0])


def pinned_dtype(array):
    """The dtype as the pin file records it; every string width counts as str.
    """
    if array.dtype.kind == "U":
        return "str"
    return str(array.dtype)


def staged_structure(scenario_file):
    """The shape and dtype of every array in a staged file."""
    return {
        key: {
            "shape": list(scenario_file[key].shape),
            "dtype": pinned_dtype(scenario_file[key]),
        } for key in sorted(scenario_file.files)
    }


def write_staging_pin():
    """Rewrites the pin file from the current staging code. Run by hand after a
    deliberate layout change.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temporary_directory:
        structure = staged_structure(stage_fixture(Path(temporary_directory)))
    STAGING_PIN_PATH.write_text(json.dumps(structure, indent=2) + "\n")


def test_staging_names_files_by_scenario_id(tmp_path,):
    """Each scenario becomes one file named by its scenario id, across shards.
    """
    first_shard = tmp_path / "training.tfrecord-00000-of-01000"
    second_shard = tmp_path / "training.tfrecord-00001-of-01000"
    write_shard(
        first_shard,
        [
            build_fixture_scenario("aaa111"),
            build_fixture_scenario("bbb222"),
        ],
    )
    write_shard(second_shard, [build_fixture_scenario("ccc333")])
    output_directory = tmp_path / "staged"
    output_directory.mkdir()

    written_paths, spacing_deviations = stage.stage_shards(
        [first_shard, second_shard], output_directory)

    assert sorted(path.name for path in written_paths) == [
        "aaa111.npz",
        "bbb222.npz",
        "ccc333.npz",
    ]
    assert all(path.exists() for path in written_paths)
    assert len(spacing_deviations) == len(written_paths)


def test_duplicate_scenario_ids_are_refused(tmp_path):
    """Two scenarios with the same id would overwrite each other, so staging
    must stop.
    """
    first_shard = tmp_path / "training.tfrecord-00000-of-01000"
    second_shard = tmp_path / "training.tfrecord-00001-of-01000"
    write_shard(first_shard, [build_fixture_scenario("same0001")])
    write_shard(second_shard, [build_fixture_scenario("same0001")])
    output_directory = tmp_path / "staged"
    output_directory.mkdir()

    with pytest.raises(AssertionError):
        stage.stage_shards([first_shard, second_shard], output_directory)


def test_staged_structure_matches_pin(tmp_path):
    """The staged layout must equal the pinned one, because the Kaggle datasets
    were staged with it.
    """
    structure = staged_structure(stage_fixture(tmp_path))
    pinned_structure = json.loads(STAGING_PIN_PATH.read_text())
    assert structure == pinned_structure
