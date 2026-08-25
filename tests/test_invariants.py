import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import fit_anchors
import submit
import train
from womd_protos import scenario_pb2
from womd import baseline, contract, frame_ops, loader, loss, metrics, model, pipeline, store, tfrecord

STAGED_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "staged"


def lane_polyline_rows(first_dot_x, dot_count):
    rows = np.zeros((dot_count, contract.MAP_FEATURE_DIM), dtype=np.float32)
    rows[:, contract.MAP_POSITION] = np.stack(
        [first_dot_x + np.arange(dot_count, dtype=np.float64), np.zeros(dot_count)], axis=1
    )
    rows[:, contract.MAP_DIRECTION] = np.array([1.0, 0.0])
    rows[:, contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")] = 1.0
    return rows


def synthetic_scene_batch(sample_count, neighbour_count, polyline_count, dots_per_polyline):
    agent_history = torch.randn(sample_count, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, :, contract.AGENT_TYPE] = 0.0
    for sample_index in range(sample_count):
        agent_history[
            sample_index, :, contract.AGENT_TYPE.start + sample_index % contract.NUM_OBJECT_TYPES
        ] = 1.0
    dot_count = polyline_count * dots_per_polyline
    map_rows = torch.randn(dot_count, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (dot_count, 2), dtype=torch.float32
    )
    return {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(sample_count, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(
            sample_count, neighbour_count, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM
        ),
        "neighbour_history_mask": torch.ones(
            sample_count, neighbour_count, contract.HISTORY_STEPS, dtype=torch.bool
        ),
        "agent_signal_history": torch.rand(
            sample_count, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.rand(
            sample_count, neighbour_count, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.arange(dot_count) // dots_per_polyline,
        "map_chunk_signal_history": torch.rand(
            sample_count, polyline_count, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "max_polylines_in_batch": torch.tensor(polyline_count),
        "future_positions": torch.randn(sample_count, contract.FUTURE_STEPS, 2),
        "future_mask": torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool),
    }


def test_crc32c_matches_known_answer():
    assert tfrecord.crc32c(b"123456789") == 0xE3069283


def test_tfrecord_round_trips_payloads():
    payloads = [b"", b"a", b"scenario-bytes" * 37]
    buffer = io.BytesIO()
    for payload in payloads:
        tfrecord.write_record(buffer, payload)
    buffer.seek(0)
    assert list(tfrecord.read_records(buffer)) == payloads


def test_tfrecord_rejects_corrupt_payload():
    buffer = io.BytesIO()
    tfrecord.write_record(buffer, b"intact-payload")
    corrupted = bytearray(buffer.getvalue())
    corrupted[14] ^= 0xFF
    with pytest.raises(tfrecord.CorruptRecordError):
        list(tfrecord.read_records(io.BytesIO(bytes(corrupted))))


def test_v1_agent_frame_transform_inverts():
    random_generator = np.random.default_rng(3)
    world_positions = random_generator.uniform(-120.0, 120.0, size=(64, 2))
    origin = np.array([13.5, -42.25])
    heading = 0.9137

    local = frame_ops.positions_to_agent_frame(world_positions, origin, heading)
    recovered = frame_ops.positions_to_world_frame(local, origin, heading)
    assert np.allclose(recovered, world_positions, atol=1e-9)


def test_v1_the_agent_frame_puts_a_point_straight_ahead_on_its_own_forward_axis():
    origin = np.array([13.5, -42.25])
    heading = 0.9137
    forward = np.array([np.cos(heading), np.sin(heading)])
    leftward = np.array([-np.sin(heading), np.cos(heading)])
    distance = 17.0

    straight_ahead = frame_ops.positions_to_agent_frame(
        origin + distance * forward, origin, heading
    )
    off_to_the_left = frame_ops.positions_to_agent_frame(
        origin + distance * leftward, origin, heading
    )

    assert np.allclose(straight_ahead, np.array([distance, 0.0]), atol=1e-9)
    assert np.allclose(off_to_the_left, np.array([0.0, distance]), atol=1e-9)

    transposed_convention = (distance * forward) @ frame_ops.rotation_matrix(heading).T
    assert not np.allclose(transposed_convention, straight_ahead, atol=1e-9)


def test_v1_heading_wrap_stays_in_range():
    angles = np.array([-7.0, -np.pi, 0.0, np.pi, 7.0, 100.0])
    wrapped = frame_ops.wrap_to_pi(angles)
    assert (wrapped >= -np.pi).all() and (wrapped < np.pi).all()


def test_v6_storage_then_agent_frame_matches_direct_world_to_agent():
    track = scenario_pb2.Track()
    track.object_type = scenario_pb2.Track.TYPE_VEHICLE
    for step_index in range(contract.TOTAL_STEPS):
        turn_angle = 0.03 * step_index
        state = track.states.add()
        state.center_x = 40.0 + 1.5 * step_index * np.cos(turn_angle)
        state.center_y = -25.0 + 1.5 * step_index * np.sin(turn_angle)
        state.heading = turn_angle + 0.4
        state.velocity_x = 15.0 * np.cos(turn_angle + 0.4)
        state.velocity_y = 15.0 * np.sin(turn_angle + 0.4)
        state.length = 4.5
        state.width = 2.0
        state.valid = True

    sdc_origin = np.array([-12.0, 31.0])
    sdc_heading = -1.2
    stored_rows, stored_valid = store.track_to_feature_rows(track, sdc_origin, sdc_heading, False)
    track_rows = stored_rows.astype(np.float32)[np.newaxis]

    origin, heading = loader.sample_frame(track_rows, 0)
    two_step = loader.track_rows_to_agent_frame(track_rows[0], origin, heading)

    world_positions = np.array([[state.center_x, state.center_y] for state in track.states])
    world_headings = np.array([state.heading for state in track.states])
    world_velocities = np.array([[state.velocity_x, state.velocity_y] for state in track.states])
    now_state = track.states[contract.CURRENT_STEP_INDEX]
    now_position = np.array([now_state.center_x, now_state.center_y])

    direct_positions = frame_ops.positions_to_agent_frame(world_positions, now_position, now_state.heading)
    direct_headings = world_headings - now_state.heading
    direct_velocities = frame_ops.directions_to_agent_frame(world_velocities, now_state.heading)

    assert np.allclose(two_step[:, contract.AGENT_POSITION], direct_positions, atol=1e-3)
    assert np.allclose(two_step[:, contract.AGENT_HEADING_COSINE], np.cos(direct_headings), atol=1e-3)
    assert np.allclose(two_step[:, contract.AGENT_HEADING_SINE], np.sin(direct_headings), atol=1e-3)
    assert np.allclose(two_step[:, contract.AGENT_VELOCITY], direct_velocities, atol=1e-3)


def test_polyline_pooling_isolates_groups_and_leaves_empty_slots_absent():
    torch.manual_seed(5)
    dot_embeddings = torch.randn(9, 8)
    dot_polyline_slot = torch.tensor([0, 0, 0, 1, 1, 1, 3, 3, 4])

    tokens, present = model.pool_dots_to_polyline_tokens(dot_embeddings, dot_polyline_slot, 2, 3)
    assert tokens.shape == (2, 3, 8)
    assert present.tolist() == [[True, True, False], [True, True, False]]
    assert torch.all(tokens[:, 2] == 0.0)

    poisoned = dot_embeddings.clone()
    poisoned[:3] = 999.0
    poisoned_tokens, _ = model.pool_dots_to_polyline_tokens(poisoned, dot_polyline_slot, 2, 3)
    assert torch.equal(poisoned_tokens[0, 1:], tokens[0, 1:])
    assert torch.equal(poisoned_tokens[1], tokens[1])
    assert not torch.equal(poisoned_tokens[0, 0], tokens[0, 0])


def test_v4_null_baselines_reproduce_the_motion_each_one_assumes():
    step_offsets = np.arange(-contract.CURRENT_STEP_INDEX, contract.FUTURE_STEPS + 1)
    elapsed = step_offsets * baseline.TIMESTEP_SECONDS
    speeds = np.array([8.0, 0.0])
    yaw_rates = np.array([0.3, 0.0])

    turned = yaw_rates[:, None] * elapsed[None, :]
    radii = np.where(yaw_rates == 0.0, 0.0, speeds / np.where(yaw_rates == 0.0, 1.0, yaw_rates))
    arc_positions = np.stack(
        [radii[:, None] * np.sin(turned), radii[:, None] * (1.0 - np.cos(turned))], axis=-1
    )

    agent_track = np.zeros((2, len(step_offsets), contract.AGENT_FEATURE_DIM), dtype=np.float32)
    agent_track[..., contract.AGENT_POSITION] = arc_positions
    agent_track[..., contract.AGENT_HEADING_COSINE] = np.cos(turned)
    agent_track[..., contract.AGENT_HEADING_SINE] = np.sin(turned)
    agent_track[..., contract.AGENT_VELOCITY] = speeds[:, None, None] * np.stack(
        [np.cos(turned), np.sin(turned)], axis=-1
    )

    batch = {
        "agent_history": torch.from_numpy(agent_track[:, :contract.HISTORY_STEPS]),
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
    }
    logged_future = torch.from_numpy(arc_positions[:, contract.HISTORY_STEPS:].astype(np.float32))

    turning_trajectories, turning_logits = baseline.constant_turn_rate_and_velocity(batch)
    straight_trajectories, _ = baseline.constant_velocity(batch)
    assert turning_trajectories.shape == (2, 1, contract.FUTURE_STEPS, 2)
    assert turning_logits.shape == (2, 1)

    assert torch.allclose(turning_trajectories[:, 0], logged_future, atol=1e-3)
    assert (straight_trajectories[0, 0, -1] - logged_future[0, -1]).norm() > 1.0
    assert torch.all(turning_trajectories[1] == 0.0)
    assert torch.all(straight_trajectories[1] == 0.0)

    pruned, _ = model.prune_modes_batched(turning_trajectories, turning_logits)
    assert torch.equal(pruned, turning_trajectories.expand(-1, contract.NUM_PREDICTED_MODES, -1, -1))


def test_v4_anchor_null_drives_straight_to_each_most_used_anchor_in_metres_regardless_of_speed():
    unit_anchors = model.unit_anchor_offsets_per_type() * torch.tensor(
        [40.0, 24.0, 12.0]
    )[:, None, None]
    driven_types = torch.tensor([0, contract.NUM_OBJECT_TYPES - 1])
    current_speeds = torch.tensor([8.0, 0.0])
    agent_history = torch.zeros(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.stack(
        [current_speeds, torch.zeros(2)], dim=-1
    )
    agent_history[
        torch.arange(2), contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE.start + driven_types
    ] = 1.0
    batch = {"agent_history": agent_history}

    trajectories, confidence_logits = baseline.straight_lines_to_most_used_anchors(
        batch, unit_anchors
    )

    assert trajectories.shape == (2, contract.NUM_PREDICTED_MODES, contract.FUTURE_STEPS, 2)
    assert confidence_logits.shape == (2, contract.NUM_PREDICTED_MODES)
    assert torch.allclose(
        trajectories[:, :, -1],
        unit_anchors[driven_types][:, : contract.NUM_PREDICTED_MODES],
        atol=1e-4,
    )

    steps = torch.cat([trajectories[..., :1, :], trajectories.diff(dim=-2)], dim=-2)
    assert torch.allclose(steps, steps[..., :1, :].expand_as(steps), atol=1e-5)


def test_anchor_fitting_recovers_tight_well_separated_clusters_and_leaves_no_anchor_empty():
    random_generator = np.random.default_rng(17)
    cluster_centres = model.unit_anchor_offsets() * 10.0 + torch.tensor(
        random_generator.uniform(-0.08, 0.08, (model.QUERY_COUNT, 2)), dtype=torch.float32
    )
    endpoints = (
        cluster_centres[:, None, :]
        + torch.tensor(
            random_generator.normal(0.0, 0.005, (model.QUERY_COUNT, 50, 2)), dtype=torch.float32
        )
    ).reshape(-1, 2)

    fitted, assignment, _, _, stopped_by_convergence = fit_anchors.fit_unit_anchors(endpoints)
    counts = fit_anchors.endpoints_per_centre(assignment, model.QUERY_COUNT)

    assert stopped_by_convergence
    assert fitted.shape == (model.QUERY_COUNT, 2)
    nearest_cluster = torch.cdist(fitted, cluster_centres).argmin(dim=1)
    assert sorted(nearest_cluster.tolist()) == list(range(model.QUERY_COUNT))
    assert torch.allclose(fitted, cluster_centres[nearest_cluster], atol=0.01)
    assert int(counts.min()) > 0


def test_batched_pruning_walk_matches_the_single_sample_walk():
    torch.manual_seed(7)
    trajectories = model.PRUNE_DISTANCE_METRES * torch.randn(
        2, model.QUERY_COUNT, contract.FUTURE_STEPS, 2
    )
    confidence_logits = torch.randn(2, model.QUERY_COUNT)
    trajectories[1] = trajectories[1, :1]

    batched_trajectories, batched_logits, kept_counts = model.prune_modes_batched_with_kept_count(
        trajectories, confidence_logits
    )
    assert kept_counts[0] > 1 and kept_counts[0] <= contract.NUM_PREDICTED_MODES
    assert kept_counts[1] == 1
    for sample_index in range(len(kept_counts)):
        walked_trajectories, walked_logits = model.prune_modes(
            trajectories[sample_index], confidence_logits[sample_index]
        )
        assert torch.equal(walked_trajectories, batched_trajectories[sample_index])
        assert torch.equal(walked_logits, batched_logits[sample_index])


def test_opposed_logged_endpoints_are_assigned_to_opposed_anchor_territories():
    unit_anchors = model.unit_anchor_offsets() * 40.0
    future_positions = torch.zeros(2, contract.FUTURE_STEPS, 2)
    future_positions[0, :, 0] = torch.linspace(0.5, 40.0, contract.FUTURE_STEPS)
    future_positions[1, :, 0] = torch.linspace(-0.5, -40.0, contract.FUTURE_STEPS)
    future_mask = torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool)

    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(2, -1, -1), future_positions, future_mask
    )

    assert assigned_mode[0] != assigned_mode[1]
    assert unit_anchors[assigned_mode[0], 0] > 0
    assert unit_anchors[assigned_mode[1], 0] < 0
    assert (unit_anchors[assigned_mode[0]] - unit_anchors[assigned_mode[1]]).norm() > 20.0


def test_assignment_reads_the_last_valid_future_step_and_not_the_padded_tail():
    unit_anchors = model.unit_anchor_offsets() * 40.0
    future_positions = torch.zeros(1, contract.FUTURE_STEPS, 2)
    future_positions[0, :10, 0] = 40.0
    future_mask = torch.zeros(1, contract.FUTURE_STEPS, dtype=torch.bool)
    future_mask[0, :10] = True

    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(1, -1, -1), future_positions, future_mask
    )
    padded_tail_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(1, -1, -1), future_positions,
        torch.ones_like(future_mask),
    )

    assert torch.allclose(unit_anchors[assigned_mode[0]], torch.tensor([40.0, 0.0]), atol=1e-4)
    assert padded_tail_mode[0] != assigned_mode[0]
    assert float(unit_anchors[padded_tail_mode[0]].norm()) == pytest.approx(
        40.0 / model.ANCHOR_DISTANCE_COUNT
    )


def test_submission_world_frame_returns_the_logged_future_to_its_logged_place():
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:1]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")

    scenario_array = loader.read_scenario(scenario_paths[0])
    track_index = int(loader.eligible_track_indices(
        scenario_array["track_rows"], scenario_array["track_valid"],
        scenario_array["is_designated_target"], True,
    )[0])
    sample = loader.build_sample(scenario_array, track_index)
    agent_frame_future = sample["future_positions"]

    world_future = submit.agent_frame_to_world_frame(agent_frame_future, sample, scenario_array)
    world_coordinate_resolution = np.finfo(np.float32).eps * np.abs(world_future).max()

    back_to_storage_frame = frame_ops.positions_to_agent_frame(
        world_future, scenario_array["frame_origin"], scenario_array["frame_heading"]
    )
    back_to_agent_frame = frame_ops.positions_to_agent_frame(
        back_to_storage_frame, sample["frame_origin"], sample["frame_heading"]
    )
    assert np.allclose(back_to_agent_frame, agent_frame_future, atol=world_coordinate_resolution)

    logged_world_future = frame_ops.positions_to_world_frame(
        scenario_array["track_rows"][
            track_index, contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION
        ],
        scenario_array["frame_origin"], scenario_array["frame_heading"],
    )
    logged = scenario_array["track_valid"][track_index, contract.CURRENT_STEP_INDEX + 1:]
    assert logged.any()
    assert np.allclose(
        world_future[logged], logged_world_future[logged], atol=world_coordinate_resolution
    )


def test_the_backfill_monitor_tells_six_distinct_futures_from_one_future_repeated():
    torch.manual_seed(101)
    sample_count = 2
    confidence_logits = torch.randn(sample_count, model.QUERY_COUNT)
    future_positions = torch.zeros(sample_count, contract.FUTURE_STEPS, 2)
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)

    separated_trajectories = torch.zeros(
        sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS, 2
    )
    separated_trajectories[:, :, -1, 0] = 2.0 * model.PRUNE_DISTANCE_METRES * torch.arange(
        1, model.QUERY_COUNT + 1, dtype=torch.float32
    )
    collapsed_trajectories = torch.zeros_like(separated_trajectories)

    separated_accumulator = metrics.MetricAccumulator()
    separated_accumulator.update(
        separated_trajectories, confidence_logits, future_positions, future_mask
    )
    collapsed_accumulator = metrics.MetricAccumulator()
    collapsed_accumulator.update(
        collapsed_trajectories, confidence_logits, future_positions, future_mask
    )

    assert separated_accumulator.results()["mean_kept_modes"] == float(
        contract.NUM_PREDICTED_MODES
    )
    assert separated_accumulator.results()["backfill_rate"] == 0.0
    assert collapsed_accumulator.results()["mean_kept_modes"] == 1.0
    assert collapsed_accumulator.results()["backfill_rate"] == 1.0


def test_v4_the_constant_velocity_null_carries_a_straight_agent_at_exactly_its_logged_speed():
    logged_velocity = torch.tensor([[11.0, -4.0], [0.0, 0.0]])
    agent_history = torch.zeros(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, :, contract.AGENT_HEADING_COSINE] = 1.0
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = logged_velocity
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_POSITION] = torch.tensor(
        [[3.0, 5.0], [3.0, 5.0]]
    )
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
    }

    trajectories, _ = baseline.constant_velocity(batch)
    elapsed_seconds = baseline.TIMESTEP_SECONDS * torch.arange(
        1, contract.FUTURE_STEPS + 1, dtype=torch.float32
    )
    straight_line = torch.tensor([3.0, 5.0]) + logged_velocity[0] * elapsed_seconds[:, None]

    assert torch.allclose(trajectories[0, 0], straight_line, atol=1e-4)
    assert torch.allclose(
        trajectories[0, 0].diff(dim=0).norm(dim=-1),
        torch.full((contract.FUTURE_STEPS - 1,), float(
            logged_velocity[0].norm() * baseline.TIMESTEP_SECONDS
        )),
        atol=1e-4,
    )
    assert torch.allclose(
        trajectories[1, 0], torch.tensor([3.0, 5.0]).expand(contract.FUTURE_STEPS, 2), atol=1e-4
    )


def test_artifact_provenance_round_trips_and_refuses_missing_or_mismatched_stamps():
    stamp = contract.artifact_provenance("test_invariants.py", "unit-test-source")
    checked = contract.check_artifact_provenance(stamp, "in-memory", "Regenerate.")
    assert checked["code_version"] == contract.STAGING_CODE_VERSION
    assert checked["producer"] == "test_invariants.py"

    with pytest.raises(AssertionError):
        contract.check_artifact_provenance(None, "in-memory", "Regenerate.")

    import json as json_module
    stale = json_module.dumps({
        "code_version": "not-the-working-tree", "producer": "x", "source": "y"
    })
    with pytest.raises(AssertionError):
        contract.check_artifact_provenance(stale, "in-memory", "Regenerate.")


def test_scorer_tensors_group_per_scenario_with_every_agent_in_the_ground_truth(tmp_path):
    import importlib.util
    runner_spec = importlib.util.spec_from_file_location(
        "container_runner", Path(__file__).parent.parent / "container" / "runner.py"
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)

    scenario_agent_counts = {"scn_a": 3, "scn_b": 2}
    scenario_target_track_ids = {"scn_a": [20, 30], "scn_b": [10]}
    for scenario_id, agent_count in scenario_agent_counts.items():
        track_rows = np.zeros(
            (agent_count, contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32
        )
        track_rows[:, :, contract.AGENT_HEADING_COSINE] = 1.0
        track_rows[:, :, contract.AGENT_TYPE.start] = 1.0
        for track_index in range(agent_count):
            track_rows[track_index, :, contract.AGENT_POSITION.start] = float(track_index)
        np.savez(
            tmp_path / f"{scenario_id}.npz",
            track_rows=track_rows,
            track_valid=np.ones((agent_count, contract.TOTAL_STEPS), dtype=bool),
            track_ids=np.arange(10, 10 * (agent_count + 1), 10, dtype=np.int64),
            is_designated_target=np.ones(agent_count, dtype=bool),
            is_object_of_interest=np.zeros(agent_count, dtype=bool),
            map_rows=np.zeros((0, contract.MAP_FEATURE_DIM), dtype=np.float32),
            feature_lengths=np.zeros(0, dtype=np.int64),
            feature_ids=np.zeros(0, dtype=np.int64),
            feature_is_interpolating=np.zeros(0, dtype=bool),
            polyline_signal_histories=np.zeros(
                (0, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES), dtype=np.float32
            ),
            frame_origin=np.zeros(2, dtype=np.float32),
            frame_heading=np.float32(0.0),
            scenario_id=scenario_id,
        )

    prediction_scenario_ids, prediction_track_ids = [], []
    for scenario_id, target_track_ids in scenario_target_track_ids.items():
        for track_id in target_track_ids:
            prediction_scenario_ids.append(scenario_id)
            prediction_track_ids.append(track_id)
    target_count = len(prediction_track_ids)
    predictions = {
        "scenario_id": np.array(prediction_scenario_ids),
        "track_id": np.array(prediction_track_ids, dtype=np.int64),
        "world_trajectories": np.random.default_rng(0).normal(
            size=(target_count, contract.NUM_PREDICTED_MODES, contract.SUBMISSION_STEPS, 2)
        ),
        "confidences": np.full((target_count, contract.NUM_PREDICTED_MODES), 1.0 / 6.0),
    }

    tensors = runner.build_motion_metric_tensors(predictions, tmp_path)

    assert tensors["ground_truth_trajectory"].shape == (2, 3, contract.TOTAL_STEPS, 7)
    assert tensors["prediction_trajectory"].shape == (
        2, 2, contract.NUM_PREDICTED_MODES, 1, contract.SUBMISSION_STEPS, 2
    )
    assert int(tensors["prediction_ground_truth_indices_mask"].sum()) == target_count

    scenario_row = {scenario_id: row for row, scenario_id in enumerate(tensors["scenario_id"])}
    a_row, b_row = scenario_row["scn_a"], scenario_row["scn_b"]
    assert tensors["ground_truth_is_valid"][a_row].all()
    assert tensors["ground_truth_is_valid"][b_row, :2].all()
    assert not tensors["ground_truth_is_valid"][b_row, 2:].any()
    assert not tensors["prediction_ground_truth_indices_mask"][b_row, 1:].any()

    for slot_index, expected_track_id in enumerate(scenario_target_track_ids["scn_a"]):
        agent_row = int(tensors["prediction_ground_truth_indices"][a_row, slot_index, 0])
        assert tensors["object_id"][a_row, agent_row] == expected_track_id
        assert tensors["ground_truth_trajectory"][
            a_row, agent_row, 0, 0
        ] == pytest.approx(agent_row * 1.0)


def test_permuting_neighbours_and_map_dots_leaves_trajectories_unchanged():
    torch.manual_seed(11)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(2, 6, 5, 16)
    batch["neighbour_history_mask"][:, 5] = False
    neighbour_order = torch.randperm(6)
    map_order = torch.randperm(batch["map_rows"].shape[0])
    permuted = dict(batch)
    permuted["neighbour_history"] = batch["neighbour_history"][:, neighbour_order]
    permuted["neighbour_history_mask"] = batch["neighbour_history_mask"][:, neighbour_order]
    permuted["neighbour_signal_history"] = batch["neighbour_signal_history"][:, neighbour_order]
    permuted["map_rows"] = batch["map_rows"][map_order]
    permuted["map_dot_polyline_slot"] = batch["map_dot_polyline_slot"][map_order]
    with torch.no_grad():
        base_trajectories, base_logits = predictor(batch)
        permuted_trajectories, permuted_logits = predictor(permuted)
    assert torch.allclose(base_trajectories, permuted_trajectories, atol=1e-4)
    assert torch.allclose(base_logits, permuted_logits, atol=1e-5)


def two_lane_signal_scenario(track_count, signalled_lane_history):
    signalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane[:, contract.MAP_POSITION.start + 1] = 40.0
    feature_lengths = np.array([11, 11], dtype=np.int64)
    polyline_signal_histories = np.zeros(
        (2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES), dtype=np.float32
    )
    polyline_signal_histories[0] = signalled_lane_history
    track_rows = np.zeros((track_count, contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32)
    track_valid = np.ones((track_count, contract.TOTAL_STEPS), dtype=bool)
    track_rows[:, :, contract.AGENT_HEADING_COSINE] = 1.0
    track_rows[0, :, contract.AGENT_POSITION] = np.array([5.0, 0.0])
    track_rows[1, :, contract.AGENT_POSITION] = np.array([5.0, 40.0])
    if track_count > 2:
        track_rows[2, :, contract.AGENT_POSITION] = np.array([7.0, 0.0])
        track_valid[2, contract.CURRENT_STEP_INDEX] = False
    scenario_array = {
        "map_rows": np.concatenate([signalled_lane, unsignalled_lane]),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102], dtype=np.int64),
        "polyline_signal_histories": polyline_signal_histories,
        "track_rows": track_rows,
        "track_valid": track_valid,
        "scenario_id": np.array("two-lane"),
        "track_ids": np.arange(track_count),
        "is_designated_target": np.ones(track_count, dtype=bool),
        "is_object_of_interest": np.zeros(track_count, dtype=bool),
    }
    return loader.with_derived_arrays(scenario_array)


def test_each_agent_carries_the_signal_history_of_the_lane_it_is_assigned_to():
    signalled_lane_history = np.zeros(
        (contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES), dtype=np.float32
    )
    signalled_lane_history[:8, contract.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_STOP")] = 1.0
    signalled_lane_history[8:, contract.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_GO")] = 1.0
    three_agent_sample = loader.build_sample(two_lane_signal_scenario(3, signalled_lane_history), 0)
    two_agent_sample = loader.build_sample(two_lane_signal_scenario(2, signalled_lane_history), 0)
    assert np.array_equal(three_agent_sample["agent_signal_history"], signalled_lane_history)
    assert np.all(three_agent_sample["neighbour_signal_history"] == 0.0)
    batch = loader.build_batch([three_agent_sample, two_agent_sample])
    assert batch["neighbour_signal_history"].shape == (
        2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
    )
    assert np.all(batch["neighbour_signal_history"][1, 1] == 0.0)
    torch_batch = {name: torch.from_numpy(array) for name, array in batch.items()}
    silenced_batch = dict(torch_batch)
    silenced_batch["agent_signal_history"] = torch.zeros_like(torch_batch["agent_signal_history"])
    torch.manual_seed(47)
    encoder = model.SceneEncoder()
    with torch.no_grad():
        tokens, _ = encoder(torch_batch)
        silenced_tokens, _ = encoder(silenced_batch)
    assert not torch.allclose(tokens[0, 0], silenced_tokens[0, 0])


def test_a_mode_endpoint_is_its_anchor_when_the_head_is_zero_and_the_loss_assigns_by_that_anchor():
    torch.manual_seed(37)
    unit_anchors = model.unit_anchor_offsets_per_type() * torch.tensor([40.0, 24.0, 12.0])[:, None, None]
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.zero_()
        predictor.mode_decoder.trajectory_head[-1].bias.zero_()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)
    with torch.no_grad():
        trajectories, _, _, selected_unit_anchors = predictor.predict(batch)
    assert torch.allclose(trajectories[:, :, -1], unit_anchors, atol=1e-4)
    assert torch.allclose(selected_unit_anchors, unit_anchors)
    ramp = torch.arange(1, contract.FUTURE_STEPS + 1) / contract.FUTURE_STEPS
    logged_futures = unit_anchors[0][:, None, :] * ramp[None, :, None]
    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors[0].expand(model.QUERY_COUNT, -1, -1), logged_futures,
        torch.ones(model.QUERY_COUNT, contract.FUTURE_STEPS, dtype=torch.bool),
    )
    assert torch.equal(assigned_mode, torch.arange(model.QUERY_COUNT))


def test_the_regression_term_is_the_gaussian_negative_log_likelihood_at_the_stated_uncertainty():
    torch.manual_seed(53)
    sample_count = 4
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    confidence_logits = torch.zeros(sample_count, model.QUERY_COUNT)
    unit_anchors = model.unit_anchor_offsets().expand(sample_count, -1, -1)
    displacement = torch.tensor([0.3, -0.4])

    def regression_at(log_standard_deviation, error):
        trajectories = future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1) + error
        _, regression, _ = loss.prediction_loss(
            trajectories, torch.full_like(trajectories, log_standard_deviation), confidence_logits,
            future_positions, future_mask, unit_anchors,
        )
        return float(regression)

    floor = model.MINIMUM_LOG_STANDARD_DEVIATION
    exact_form = contract.FUTURE_STEPS * (
        2.0 * floor + 2.0 * loss.HALF_LOG_TWO_PI
        + 0.5 * float((displacement ** 2).sum()) / math.exp(floor) ** 2
    )
    assert regression_at(floor, displacement) == pytest.approx(exact_form, rel=1e-5)
    for log_standard_deviation in (floor, 0.0, model.MAXIMUM_LOG_STANDARD_DEVIATION):
        assert regression_at(log_standard_deviation, torch.zeros(2)) < regression_at(
            log_standard_deviation, displacement
        )


def test_classification_is_softmax_cross_entropy_on_the_assigned_mode():
    sample_count = 3
    future_positions = torch.zeros(sample_count, contract.FUTURE_STEPS, 2)
    future_positions[:, -1, 0] = 1.0
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    trajectories = future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    unit_anchors = torch.zeros(sample_count, model.QUERY_COUNT, 2)
    unit_anchors[:, 0, 0] = 1.0
    _, _, classification = loss.prediction_loss(
        trajectories, torch.zeros_like(trajectories), torch.zeros(sample_count, model.QUERY_COUNT),
        future_positions, future_mask, unit_anchors,
    )
    assert float(classification) == pytest.approx(math.log(model.QUERY_COUNT), rel=1e-5)


def test_a_masked_future_step_moves_no_term_of_the_loss():
    torch.manual_seed(103)
    sample_count = 4
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    future_mask[1, contract.FUTURE_STEPS // 2:] = False
    future_mask[2] = False
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    trajectories = torch.randn(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS, 2)
    log_standard_deviation = torch.randn_like(trajectories)
    confidence_logits = torch.randn(sample_count, model.QUERY_COUNT)
    unit_anchors = model.unit_anchor_offsets().expand(sample_count, -1, -1) * 40.0

    def components(logged_positions, predicted_positions):
        return torch.stack(loss.prediction_loss(
            predicted_positions, log_standard_deviation, confidence_logits,
            logged_positions, future_mask, unit_anchors,
        ))

    masked_steps = ~future_mask
    poisoned_positions = future_positions.clone()
    poisoned_positions[masked_steps] = 1e6
    poisoned_trajectories = trajectories.clone()
    poisoned_trajectories[masked_steps.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1)] = 1e6
    assert torch.equal(
        components(future_positions, trajectories), components(poisoned_positions, poisoned_trajectories)
    )


def test_every_quantity_the_model_emits_is_pinned_by_the_loss_that_trains_it():
    torch.manual_seed(83)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 3, 2, 10)
    with torch.no_grad():
        emitted = list(predictor.predict(batch))
    names = ("trajectories", "log_standard_deviation", "confidence_logits", "unit_anchors")
    assert len(emitted) == len(names)
    perturbations = {
        "trajectories": lambda value: value + 1.0,
        "log_standard_deviation": lambda value: value + 1.0,
        "confidence_logits": lambda value: torch.cat([value[..., :1] + 1.0, value[..., 1:]], dim=-1),
        "unit_anchors": lambda value: -value,
    }

    def total(quantities):
        trajectories, log_standard_deviation, confidence_logits, unit_anchors = quantities
        return loss.prediction_loss(
            trajectories, log_standard_deviation, confidence_logits,
            batch["future_positions"], batch["future_mask"], unit_anchors,
        )[0]

    unperturbed = total(emitted)
    for position, name in enumerate(names):
        perturbed = list(emitted)
        perturbed[position] = perturbations[name](emitted[position])
        assert not torch.equal(total(perturbed), unperturbed), name


def test_no_step_uncertainty_starts_saturated_and_untrained_modes_are_distinct():
    torch.manual_seed(97)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)
    with torch.no_grad():
        trajectories, log_standard_deviation, confidence_logits, _ = predictor.predict(batch)
    assert trajectories.isfinite().all() and confidence_logits.isfinite().all()
    assert not (log_standard_deviation == model.MINIMUM_LOG_STANDARD_DEVIATION).any()
    assert not (log_standard_deviation == model.MAXIMUM_LOG_STANDARD_DEVIATION).any()
    endpoints = trajectories[:, :, -1]
    assert not torch.allclose(endpoints[:, :1], endpoints, atol=1e-4)


def test_anchors_are_frozen_buffers_carried_by_the_checkpoint():
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type())
    assert "mode_decoder.unit_anchors" not in {name for name, _ in predictor.named_parameters()}
    assert "mode_decoder.unit_anchors" in predictor.state_dict()
    assert not predictor.unit_anchors.requires_grad


def test_one_training_step_runs_the_whole_path_over_staged_scenarios(tmp_path):
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:2]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")
    torch.manual_seed(0)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    batch = next(iter(pipeline.batches(scenario_paths, 0, 2, 0, 0, True)))
    total, regression, classification, trajectories, confidence_logits = train.training_losses(
        predictor, batch
    )
    assert torch.stack([total, regression, classification]).isfinite().all()
    total.backward()
    starved = [
        name for name, parameter in predictor.named_parameters()
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    ]
    assert not starved, starved
    optimizer.step()
    accumulator = metrics.MetricAccumulator()
    accumulator.update(
        trajectories.detach(), confidence_logits.detach(), batch["future_positions"], batch["future_mask"]
    )
    assert all(math.isfinite(value) for value in accumulator.results().values())
    checkpoint_path = tmp_path / "predictor.pt"
    torch.save(predictor.state_dict(), checkpoint_path)
    reloaded_state = torch.load(checkpoint_path)
    assert all(torch.equal(reloaded_state[name], parameter) for name, parameter in predictor.state_dict().items())


def test_submitted_confidences_conserve_probability_mass_and_hand_a_duplicates_share_to_its_keeper():
    torch.manual_seed(131)
    trajectories = torch.zeros(1, model.QUERY_COUNT, contract.FUTURE_STEPS, 2)
    trajectories[0, :, -1, 0] = 10.0 * torch.arange(model.QUERY_COUNT, dtype=torch.float32)
    trajectories[0, 1, -1, 0] = 0.5
    confidence_logits = torch.zeros(1, model.QUERY_COUNT)
    confidence_logits[0, 0] = 3.0
    confidence_logits[0, 1] = 2.0
    kept_trajectories, kept_logits = model.prune_modes_batched(trajectories, confidence_logits)
    confidences = model.aggregated_confidences(trajectories, confidence_logits, kept_trajectories)
    probabilities = torch.softmax(confidence_logits, dim=-1)[0]
    assert float(confidences.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(kept_trajectories[0, 0, -1, 0]) == 0.0
    assert not (kept_trajectories[0, :, -1, 0] == 0.5).any()
    assert float(confidences[0, 0]) == pytest.approx(float(probabilities[0] + probabilities[1]), rel=1e-6)
    assert float(confidences[0, 1:].sum()) == pytest.approx(1.0 - float(probabilities[0] + probabilities[1]), rel=1e-5)


def test_a_constant_step_head_walks_a_straight_line_and_a_rotating_step_head_walks_an_arc():
    torch.manual_seed(137)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type() * 0.0).eval()
    tokens = torch.randn(1, 5, model.HIDDEN_DIM)
    token_present = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        bias = decoder.trajectory_head[-1].bias.view(2, contract.FUTURE_STEPS, 2)
        bias.zero_()
        bias[0, :, 0] = 0.5
        straight, _, _, _ = decoder(tokens, token_present, torch.zeros(1, dtype=torch.long))
        angles = 0.02 * torch.arange(contract.FUTURE_STEPS, dtype=torch.float32)
        bias[0, :, 0] = 0.5 * angles.cos()
        bias[0, :, 1] = 0.5 * angles.sin()
        arc, _, _, _ = decoder(tokens, token_present, torch.zeros(1, dtype=torch.long))
    expected_straight = torch.stack(
        [0.5 * torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32), torch.zeros(contract.FUTURE_STEPS)],
        dim=-1,
    )
    assert torch.allclose(straight[0, 0], expected_straight, atol=1e-5)
    steps = arc[0, 0].diff(dim=0)
    assert torch.allclose(steps.norm(dim=-1), torch.full((contract.FUTURE_STEPS - 1,), 0.5), atol=1e-5)
    turning = torch.atan2(steps[:, 1], steps[:, 0]).diff()
    assert torch.allclose(turning, torch.full_like(turning, 0.02), atol=1e-5)
