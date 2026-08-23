import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import fit_anchors
import measure_sensitivity
import submit
import train
from womd_protos import scenario_pb2
from womd import baseline, contract, frame_ops, loader, loss, metrics, model, pipeline, store, tfrecord

STAGED_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "staged"


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


def test_v_permuting_neighbours_and_map_dots_leaves_trajectories_unchanged():
    torch.manual_seed(11)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = {
        "agent_history": torch.randn(2, 11, 13),
        "agent_history_mask": torch.ones(2, 11, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 6, 11, 13),
        "neighbour_history_mask": torch.ones(2, 6, 11, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 6, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": torch.randn(80, contract.MAP_FEATURE_DIM),
        "map_dot_polyline_slot": torch.arange(80) // 8,
        "map_chunk_signal_history": torch.zeros(
            2, 5, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(2, 5, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(5),
    }
    batch["neighbour_history_mask"][:, 5] = False
    batch["map_chunk_signal_history"][0, 1:3, :, 6] = 1.0
    batch["map_chunk_lane_context"][0, 1:3, contract.LANE_CONTEXT_REACHABLE] = 1.0
    batch["map_rows"][:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (80, 2), dtype=torch.float32
    )

    neighbour_order = torch.randperm(6)
    map_order = torch.randperm(80)
    permuted = dict(batch)
    permuted["neighbour_history"] = batch["neighbour_history"][:, neighbour_order]
    permuted["neighbour_history_mask"] = batch["neighbour_history_mask"][:, neighbour_order]
    permuted["map_rows"] = batch["map_rows"][map_order]
    permuted["map_dot_polyline_slot"] = batch["map_dot_polyline_slot"][map_order]

    with torch.no_grad():
        base_trajectories, base_logits = predictor(batch)
        permuted_trajectories, permuted_logits = predictor(permuted)

    assert torch.allclose(base_trajectories, permuted_trajectories, atol=1e-4)
    assert torch.allclose(base_logits, permuted_logits, atol=1e-5)


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


def test_later_decoder_rounds_move_when_the_previous_draft_changes():
    torch.manual_seed(41)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    assert model.DECODER_ROUNDS > 1
    tokens = torch.randn(2, 7, model.HIDDEN_DIM)
    token_present = torch.ones(2, 7, dtype=torch.bool)
    unit_anchors = model.unit_anchor_offsets().expand(2, -1, -1) * 40.0
    mode_valid = torch.ones(2, model.QUERY_COUNT, dtype=torch.bool)
    normed = decoder.scene_norm(tokens)
    with torch.no_grad():
        following_draft = decoder.decode_from_anchors(
            normed, token_present, unit_anchors, mode_valid, 2
        )[-1][0]
        decoder.draft_projection.weight.zero_()
        ignoring_draft = decoder.decode_from_anchors(
            normed, token_present, unit_anchors, mode_valid, 2
        )[-1][0]
    assert following_draft.shape[1] == model.QUERY_COUNT
    assert not torch.allclose(following_draft, ignoring_draft, atol=1e-4)


def test_queries_read_each_other_inside_every_decoder_round():
    torch.manual_seed(42)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    tokens = torch.randn(2, 7, model.HIDDEN_DIM)
    token_present = torch.ones(2, 7, dtype=torch.bool)
    unit_anchors = model.unit_anchor_offsets().expand(2, -1, -1) * 40.0
    mode_valid = torch.ones(2, model.QUERY_COUNT, dtype=torch.bool)
    normed = decoder.scene_norm(tokens)
    with torch.no_grad():
        talking = decoder.decode_from_anchors(normed, token_present, unit_anchors, mode_valid, 2)
        for self_attention in decoder.round_self_attention:
            self_attention.output_projection.weight.zero_()
            self_attention.output_projection.bias.zero_()
        silent = decoder.decode_from_anchors(normed, token_present, unit_anchors, mode_valid, 2)
    assert len(talking) == model.DECODER_ROUNDS
    for round_index in range(model.DECODER_ROUNDS):
        assert not torch.allclose(talking[round_index][0], silent[round_index][0], atol=1e-4)


def test_every_decoder_round_is_scored_by_the_training_loss():
    torch.manual_seed(44)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)
    with torch.no_grad():
        round_outputs, selected_unit_anchors, mode_valid, _, _ = predictor.predict_every_round(batch)
    assert len(round_outputs) == model.DECODER_ROUNDS

    def summed_total(outputs):
        return train.round_summed_prediction_loss(
            outputs, batch, selected_unit_anchors, mode_valid, 1.0, 1.0, 1.0
        )[0]

    unperturbed = summed_total(round_outputs)
    for round_index in range(model.DECODER_ROUNDS):
        perturbed = list(round_outputs)
        trajectories, *rest = round_outputs[round_index]
        perturbed[round_index] = (trajectories + 1.0, *rest)
        assert not torch.equal(summed_total(perturbed), unperturbed), (
            f"round {round_index}'s trajectories moved without moving the loss"
        )


def test_padded_modes_are_never_assigned_kept_or_believed():
    torch.manual_seed(46)
    anchor_counts = torch.tensor([model.QUERY_COUNT, contract.NUM_PREDICTED_MODES + 2, model.QUERY_COUNT])
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type() * 40.0, anchor_counts).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)
    pedestrian_row = contract.PREDICTED_OBJECT_TYPES.index("TYPE_PEDESTRIAN")
    with torch.no_grad():
        (
            trajectories, _, _, _, confidence_logits, _, selected_unit_anchors, mode_valid, _, _,
        ) = predictor.predict_with_heading(batch)
    valid_count = int(anchor_counts[pedestrian_row])
    assert mode_valid[pedestrian_row].tolist() == [True] * valid_count + [False] * (
        model.QUERY_COUNT - valid_count
    )
    assert mode_valid[0].all() and mode_valid[2].all()
    assert torch.isinf(confidence_logits[pedestrian_row, valid_count:]).all()
    assert torch.isfinite(confidence_logits[pedestrian_row, :valid_count]).all()

    far_future = torch.zeros(contract.NUM_OBJECT_TYPES, contract.FUTURE_STEPS, 2)
    far_future[..., :] = selected_unit_anchors[:, -1][:, None, :]
    assigned = loss.anchor_assigned_mode(
        selected_unit_anchors, far_future,
        torch.ones(contract.NUM_OBJECT_TYPES, contract.FUTURE_STEPS, dtype=torch.bool), mode_valid,
    )
    assert int(assigned[pedestrian_row]) < valid_count
    assert int(assigned[0]) == model.QUERY_COUNT - 1

    crowded = trajectories.clone()
    crowded[pedestrian_row, :valid_count, -1] = 0.0
    kept_trajectories, kept_logits, kept_count = model.prune_modes_batched_with_kept_count(
        crowded, confidence_logits, mode_valid
    )
    assert int(kept_count[pedestrian_row]) == 1
    assert torch.isfinite(kept_logits[pedestrian_row]).all()
    assert torch.equal(kept_trajectories[pedestrian_row, :, -1], torch.zeros(contract.NUM_PREDICTED_MODES, 2))
    walked_trajectories, walked_logits = model.prune_modes(
        crowded[pedestrian_row], confidence_logits[pedestrian_row], mode_valid[pedestrian_row]
    )
    assert torch.equal(walked_trajectories, kept_trajectories[pedestrian_row])
    assert torch.equal(walked_logits, kept_logits[pedestrian_row])


def lane_polyline_rows(first_dot_x, dot_count):
    rows = np.zeros((dot_count, contract.MAP_FEATURE_DIM), dtype=np.float32)
    rows[:, contract.MAP_POSITION] = np.stack(
        [first_dot_x + np.arange(dot_count, dtype=np.float64), np.zeros(dot_count)], axis=1
    )
    rows[:, contract.MAP_DIRECTION] = np.array([1.0, 0.0])
    rows[:, contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")] = 1.0
    return rows


def test_lane_context_walks_connections_forward_only_and_charges_the_lane_left_behind():
    feature_lengths = np.array([11, 11, 11], dtype=np.int64)
    scenario_array = {
        "map_rows": np.concatenate(
            [lane_polyline_rows(0.0, 11), lane_polyline_rows(20.0, 11), lane_polyline_rows(-30.0, 11)]
        ),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102, 103], dtype=np.int64),
        "map_dot_polyline_index": np.repeat(np.arange(3), feature_lengths),
        "lane_connections": np.array([[101, 102, 1], [103, 101, 1]], dtype=np.int64),
        "lane_neighbour_ids": np.zeros((0, contract.LANE_NEIGHBOUR_ID_WIDTH), dtype=np.int64),
        "stop_sign_controlled_lanes": np.zeros((0, contract.STOP_SIGN_LANE_WIDTH), dtype=np.int64),
    }

    lane_context = loader.lane_context_per_polyline(
        loader.lane_graph_of_scenario(scenario_array), np.array([5.0, 2.0]), np.array([1.0, 0.0])
    )
    own_lane, one_hop_downstream, upstream_only = lane_context

    assert np.all(
        lane_context[:, contract.LANE_CONTEXT_AGENT_LANE_DISTANCE]
        == pytest.approx(2.0 / contract.DISTANCE_NORMALISER_METRES, rel=1e-6)
    )
    assert own_lane[contract.LANE_CONTEXT_REACHABLE] == 1.0
    assert own_lane[contract.LANE_CONTEXT_GRAPH_DISTANCE] == 0.0
    assert one_hop_downstream[contract.LANE_CONTEXT_REACHABLE] == 1.0
    assert one_hop_downstream[contract.LANE_CONTEXT_GRAPH_DISTANCE] == pytest.approx(
        10.0 / contract.LANE_CONTEXT_GRAPH_DISTANCE_NORMALISER_METRES, rel=1e-6
    )
    assert upstream_only[contract.LANE_CONTEXT_REACHABLE] == 0.0
    assert upstream_only[contract.LANE_CONTEXT_GRAPH_DISTANCE] == 0.0


def test_a_straight_chunk_has_zero_curvature_and_a_circular_arc_matches_one_over_radius():
    straight = lane_polyline_rows(0.0, 11)
    radius = 20.0
    angles = np.linspace(0.0, np.pi / 2, 11)
    circular = np.zeros((11, contract.MAP_FEATURE_DIM), dtype=np.float32)
    circular[:, contract.MAP_POSITION] = np.stack(
        [radius * np.sin(angles), radius * (1.0 - np.cos(angles))], axis=1
    )
    circular[:, contract.MAP_DIRECTION] = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    circular[:, contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")] = 1.0
    map_rows = np.concatenate([straight, circular])
    feature_lengths = np.array([11, 11], dtype=np.int64)
    dot_polyline_index = np.repeat(np.arange(2), feature_lengths)
    polyline_lane_context = np.zeros((2, contract.LANE_CONTEXT_DIM), dtype=np.float32)
    polyline_signal = np.zeros(
        (2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES), dtype=np.float32
    )

    _, _, _, chunk_lane_context = loader.crop_and_reframe_map(
        map_rows, dot_polyline_index, polyline_signal, polyline_lane_context,
        np.array([0.0, 0.0]), 0.0, 0.0,
    )
    curvatures = chunk_lane_context[:, contract.LANE_CONTEXT_CURVATURE]
    assert curvatures[0] == pytest.approx(0.0, abs=1e-5)
    assert curvatures[1] == pytest.approx(1.0 / radius, rel=0.05)
    assert float(np.std(curvatures)) > 0.0


def test_a_lane_reached_only_sideways_is_flagged_a_lane_change_at_the_distance_it_changed_from():
    parallel_lane_rows = lane_polyline_rows(20.0, 11)
    parallel_lane_rows[:, contract.MAP_POSITION.start + 1] = 3.5
    feature_lengths = np.array([11, 11, 11], dtype=np.int64)
    scenario_array = {
        "map_rows": np.concatenate(
            [lane_polyline_rows(0.0, 11), lane_polyline_rows(20.0, 11), parallel_lane_rows]
        ),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102, 103], dtype=np.int64),
        "map_dot_polyline_index": np.repeat(np.arange(3), feature_lengths),
        "lane_connections": np.array([[101, 102, 1]], dtype=np.int64),
        "lane_neighbour_ids": np.array(
            [[102, 103, contract.LANE_SIDES.index("left")]], dtype=np.int64
        ),
        "stop_sign_controlled_lanes": np.zeros((0, contract.STOP_SIGN_LANE_WIDTH), dtype=np.int64),
    }

    own_lane, one_hop_downstream, sideways_only = loader.lane_context_per_polyline(
        loader.lane_graph_of_scenario(scenario_array), np.array([5.0, 0.0]), np.array([1.0, 0.0])
    )

    assert one_hop_downstream[contract.LANE_CONTEXT_REACHABLE] == 1.0
    assert one_hop_downstream[contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] == 0.0
    assert sideways_only[contract.LANE_CONTEXT_REACHABLE] == 0.0
    assert sideways_only[contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] == 1.0
    assert own_lane[contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] == 0.0
    assert sideways_only[contract.LANE_CONTEXT_GRAPH_DISTANCE] == pytest.approx(
        one_hop_downstream[contract.LANE_CONTEXT_GRAPH_DISTANCE], rel=1e-6
    )
    assert sideways_only[contract.LANE_CONTEXT_GRAPH_DISTANCE] == pytest.approx(
        10.0 / contract.LANE_CONTEXT_GRAPH_DISTANCE_NORMALISER_METRES, rel=1e-6
    )


def two_lane_signal_scenario(track_count, signalled_lane_history):
    signalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane[:, contract.MAP_POSITION.start + 1] = 40.0
    feature_lengths = np.array([11, 11], dtype=np.int64)
    polyline_signal_histories = np.zeros(
        (2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES), dtype=np.float32
    )
    polyline_signal_histories[0] = signalled_lane_history

    track_rows = np.zeros(
        (track_count, contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32
    )
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
        "lane_connections": np.zeros((0, contract.LANE_CONNECTION_WIDTH), dtype=np.int64),
        "lane_neighbour_ids": np.zeros((0, contract.LANE_NEIGHBOUR_ID_WIDTH), dtype=np.int64),
        "stop_sign_controlled_lanes": np.zeros((0, contract.STOP_SIGN_LANE_WIDTH), dtype=np.int64),
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

    three_agent_sample = loader.build_sample(
        two_lane_signal_scenario(3, signalled_lane_history), 0
    )
    two_agent_sample = loader.build_sample(two_lane_signal_scenario(2, signalled_lane_history), 0)

    assert np.array_equal(three_agent_sample["agent_signal_history"], signalled_lane_history)
    assert np.all(three_agent_sample["neighbour_signal_history"][0] == 0.0)
    assert np.all(three_agent_sample["neighbour_signal_history"][1] == 0.0)

    batch = loader.build_batch([three_agent_sample, two_agent_sample])
    assert batch["agent_signal_history"].shape == (
        2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
    )
    assert batch["neighbour_signal_history"].shape == (
        2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
    )
    assert np.all(batch["neighbour_signal_history"][1, 1] == 0.0)

    torch.manual_seed(47)
    encoder = model.SceneEncoder()
    with torch.no_grad():
        projected_agent = encoder.signal_projection(
            torch.from_numpy(batch["agent_signal_history"]).flatten(start_dim=-2)
        )
        projected_neighbours = encoder.signal_projection(
            torch.from_numpy(batch["neighbour_signal_history"]).flatten(start_dim=-2)
        )

    assert torch.any(projected_agent != 0.0)
    assert torch.all(projected_neighbours[1, 1] == 0.0)

    torch_batch = {name: torch.from_numpy(array) for name, array in batch.items()}
    silenced_batch = dict(torch_batch)
    silenced_batch["agent_signal_history"] = torch.zeros_like(
        torch_batch["agent_signal_history"]
    )
    with torch.no_grad():
        tokens, _ = encoder(torch_batch)
        silenced_tokens, _ = encoder(silenced_batch)

    assert not torch.allclose(tokens[0, 0], silenced_tokens[0, 0])


def test_neighbour_future_loss_scores_only_the_neighbour_steps_that_were_logged():
    torch.manual_seed(23)
    logged_positions = torch.randn(2, 3, contract.FUTURE_STEPS, 2)
    neighbour_future_mask = torch.ones(2, 3, contract.FUTURE_STEPS, dtype=torch.bool)
    neighbour_future_mask[:, 2] = False
    log_standard_deviation = torch.zeros_like(logged_positions)
    exact_nll = 2.0 * loss.HALF_LOG_TWO_PI * contract.FUTURE_STEPS

    neighbour_readable = torch.ones(2, 3, dtype=torch.bool)
    predicted_positions = logged_positions.clone()
    exact = loss.neighbour_future_loss(
        predicted_positions, log_standard_deviation, logged_positions,
        neighbour_future_mask, neighbour_readable,
    )
    assert torch.isfinite(exact) and float(exact) == pytest.approx(exact_nll, rel=1e-5)

    predicted_positions[:, 2] = 1e6
    assert float(
        loss.neighbour_future_loss(
            predicted_positions, log_standard_deviation, logged_positions,
            neighbour_future_mask, neighbour_readable,
        )
    ) == pytest.approx(exact_nll, rel=1e-5)

    predicted_positions[:, 0] = logged_positions[:, 0] + torch.tensor([3.0, 4.0])
    scored = loss.neighbour_future_loss(
        predicted_positions, log_standard_deviation, logged_positions,
        neighbour_future_mask, neighbour_readable,
    )
    assert torch.isfinite(scored)
    assert float(scored) == pytest.approx(6.25 * contract.FUTURE_STEPS + exact_nll, rel=1e-5)

    unreadable = neighbour_readable.clone()
    unreadable[:, 0] = False
    assert float(
        loss.neighbour_future_loss(
            predicted_positions, log_standard_deviation, logged_positions,
            neighbour_future_mask, unreadable,
        )
    ) == pytest.approx(exact_nll, rel=1e-5)


def test_padded_chunk_slot_stays_exactly_zero_through_both_map_projections():
    torch.manual_seed(19)
    encoder = model.SceneEncoder()
    map_rows = torch.randn(24, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (24, 2), dtype=torch.float32
    )
    signal_history = torch.zeros(2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)
    lane_context = torch.zeros(2, 2, contract.LANE_CONTEXT_DIM)
    signal_history[0, 0, :, 6] = 1.0
    lane_context[0, 0, contract.LANE_CONTEXT_REACHABLE] = 1.0

    with torch.no_grad():
        map_tokens, map_present = model.pool_dots_to_polyline_tokens(
            encoder.map_encoder(map_rows), torch.arange(24) // 12, 2, 2
        )
        map_tokens = (
            map_tokens
            + encoder.signal_projection(signal_history.flatten(start_dim=-2))
            + encoder.lane_context_projection(lane_context)
        )

    assert map_present.tolist() == [[True, True], [False, False]]
    assert torch.all(map_tokens[1] == 0.0)
    assert torch.any(map_tokens[0] != 0.0)


def test_an_absent_neighbours_token_is_unchanged_by_the_predicted_future_feedback():
    torch.manual_seed(29)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = {
        "agent_history": torch.randn(1, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "agent_history_mask": torch.ones(1, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(1, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(1, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            1, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            1, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": torch.randn(10, contract.MAP_FEATURE_DIM),
        "map_dot_polyline_slot": torch.arange(10) // 5,
        "map_chunk_signal_history": torch.zeros(
            1, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(1, 2, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(2),
    }
    batch["map_rows"][:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (10, 2), dtype=torch.float32
    )
    batch["neighbour_history_mask"][:, 1] = False

    with torch.no_grad():
        tokens_before_feedback, _ = predictor.scene_encoder(batch)
        tokens_after_feedback, *_ = predictor.encode_scene_and_modes(batch)

    assert torch.equal(tokens_after_feedback[:, 2], tokens_before_feedback[:, 2])
    assert not torch.equal(tokens_after_feedback[:, 1], tokens_before_feedback[:, 1])


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


def test_v4_lane_null_drives_the_centreline_at_the_logged_speed_and_forks_into_separate_routes():
    turning_lane = lane_polyline_rows(0.0, 21)
    turning_lane[:, contract.MAP_POSITION] = np.stack([np.full(21, 11.0), np.arange(21.0)], axis=1)
    turning_lane[:, contract.MAP_DIRECTION] = np.array([0.0, 1.0])
    feature_lengths = np.array([11, 20, 21], dtype=np.int64)

    speed = 10.0
    track_rows = np.zeros((1, contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32)
    track_rows[0, :, contract.AGENT_POSITION] = np.array([2.0, 0.0])
    track_rows[0, :, contract.AGENT_HEADING_COSINE] = 1.0
    track_rows[0, :, contract.AGENT_VELOCITY] = np.array([speed, 0.0])
    scenario_array = {
        "map_rows": np.concatenate(
            [lane_polyline_rows(0.0, 11), lane_polyline_rows(11.0, 20), turning_lane]
        ),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102, 103], dtype=np.int64),
        "map_dot_polyline_index": np.repeat(np.arange(3), feature_lengths),
        "lane_connections": np.array([[101, 102, 1], [101, 103, 1]], dtype=np.int64),
        "track_rows": track_rows,
    }

    trajectories, followed_a_lane = baseline.follow_the_lane_predictions(scenario_array, 0)
    step_metres = speed * baseline.TIMESTEP_SECONDS
    driven_metres = step_metres * np.arange(1, contract.FUTURE_STEPS + 1)

    assert followed_a_lane
    assert trajectories.shape == (contract.NUM_PREDICTED_MODES, contract.FUTURE_STEPS, 2)
    assert np.allclose(
        trajectories[0], np.stack([driven_metres, np.zeros(contract.FUTURE_STEPS)], axis=1), atol=1e-4
    )
    assert np.allclose(
        np.linalg.norm(np.diff(trajectories[:2], axis=1), axis=-1), step_metres, atol=1e-4
    )
    assert np.allclose(trajectories[1, -1], np.array([9.0, 71.0]), atol=1e-4)
    assert np.array_equal(trajectories[2], trajectories[0])


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


def total_turning_degrees(polyline):
    steps = polyline.diff(dim=-2)
    leading, trailing = steps[..., :-1, :], steps[..., 1:, :]
    cross = leading[..., 0] * trailing[..., 1] - leading[..., 1] * trailing[..., 0]
    return torch.atan2(cross.abs(), (leading * trailing).sum(dim=-1)).rad2deg().sum(dim=-1)


def test_a_trajectory_cannot_outrun_or_outturn_the_control_polygon_it_is_drawn_through():
    torch.manual_seed(23)
    assert model.TRAJECTORY_CONTROL_POINTS < contract.FUTURE_STEPS

    curve_basis = model.bernstein_curve_basis(
        model.TRAJECTORY_CONTROL_POINTS, contract.FUTURE_STEPS
    )
    control_points = 20.0 * torch.randn(512, model.TRAJECTORY_CONTROL_POINTS, 2)
    curve = torch.matmul(curve_basis[:, 1:], control_points)
    walked = torch.cat([torch.zeros(512, 1, 2), curve], dim=1)
    control_polygon = torch.cat([torch.zeros(512, 1, 2), control_points], dim=1)

    assert curve_basis.shape == (contract.FUTURE_STEPS, model.TRAJECTORY_CONTROL_POINTS + 1)
    assert torch.allclose(curve[:, -1], control_points[:, -1], atol=1e-4)
    assert (
        walked.diff(dim=1).norm(dim=-1).sum(dim=-1)
        <= control_polygon.diff(dim=1).norm(dim=-1).sum(dim=-1)
    ).all()
    assert (total_turning_degrees(walked) <= total_turning_degrees(control_polygon)).all()


def test_every_emitted_trajectory_stays_in_the_curve_family_after_the_anchor_ramp_is_added():
    torch.manual_seed(31)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.mul_(50.0)

    agent_history = torch.randn(3, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor(
        [[0.0, 0.0], [12.0, 0.0], [0.0, -25.0]]
    )
    map_rows = torch.randn(30, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(3, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(3, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(3, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            3, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            3, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.arange(30) // 10,
        "map_chunk_signal_history": torch.zeros(
            3, 1, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(3, 1, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(1),
    }

    with torch.no_grad():
        trajectories, _ = predictor(batch)
    flattened = trajectories.permute(2, 0, 1, 3).reshape(contract.FUTURE_STEPS, -1)
    curve_basis = predictor.mode_decoder.curve_basis
    projected = curve_basis @ torch.linalg.lstsq(curve_basis, flattened).solution

    independent_positions = trajectories.std() * torch.randn_like(flattened)
    independent_residual = (
        independent_positions - curve_basis @ torch.linalg.lstsq(
            curve_basis, independent_positions
        ).solution
    ).norm()

    assert trajectories.abs().max() > 1.0
    assert (projected - flattened).norm() < 1e-3 * flattened.norm()
    assert independent_residual > 0.1 * independent_positions.norm()


def test_swapping_two_anchors_swaps_the_modes_that_carry_them():
    torch.manual_seed(29)
    unit_anchors = model.unit_anchor_offsets_per_type() * 40.0
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.queries.zero_()

    agent_history = torch.randn(1, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor([10.0, 0.0])
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE] = 0.0
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE.start] = 1.0
    map_rows = torch.randn(12, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(1, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(1, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(1, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            1, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            1, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.zeros(12, dtype=torch.long),
        "map_chunk_signal_history": torch.zeros(
            1, 1, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(1, 1, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(1),
    }

    forward_mode = 0
    backward_mode = model.ANCHOR_DISTANCE_COUNT * (model.ANCHOR_DIRECTION_COUNT // 2)
    swapped_order = torch.arange(model.QUERY_COUNT)
    swapped_order[[forward_mode, backward_mode]] = swapped_order[[backward_mode, forward_mode]]
    with torch.no_grad():
        endpoints = predictor(batch)[0][0, :, -1]
        predictor.mode_decoder.anchor_offsets.copy_(unit_anchors[:, swapped_order])
        swapped_endpoints = predictor(batch)[0][0, :, -1]

    assert (endpoints[forward_mode] - endpoints[backward_mode]).norm() > 1.0
    assert torch.allclose(swapped_endpoints[forward_mode], endpoints[backward_mode], atol=1e-4)
    assert torch.allclose(swapped_endpoints[backward_mode], endpoints[forward_mode], atol=1e-4)
    untouched_modes = [
        mode for mode in range(model.QUERY_COUNT) if mode not in (forward_mode, backward_mode)
    ]
    assert torch.allclose(swapped_endpoints[untouched_modes], endpoints[untouched_modes], atol=1e-4)


def test_a_mode_endpoint_carries_its_anchor_at_the_distance_the_loss_assigns_by():
    torch.manual_seed(37)
    unit_anchors = model.unit_anchor_offsets_per_type() * torch.tensor(
        [40.0, 24.0, 12.0]
    )[:, None, None]
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.zero_()
        predictor.mode_decoder.trajectory_head[-1].bias.zero_()

    driven_types = torch.tensor([0, contract.NUM_OBJECT_TYPES - 1])
    agent_history = torch.randn(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor(
        [[7.0, 0.0], [0.0, 0.0]]
    )
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE] = 0.0
    agent_history[
        torch.arange(2), contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE.start + driven_types
    ] = 1.0
    map_rows = torch.randn(20, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(2, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.arange(20) // 10,
        "map_chunk_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(2, 2, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(2),
    }

    with torch.no_grad():
        trajectories, _ = predictor(batch)
    endpoints = trajectories[:, :, -1]

    assert torch.allclose(endpoints, unit_anchors[driven_types], atol=1e-4)

    ramp = torch.arange(1, contract.FUTURE_STEPS + 1) / contract.FUTURE_STEPS
    logged_futures = endpoints[0][:, None, :] * ramp[None, :, None]
    assigned_mode = loss.anchor_assigned_mode(
        predictor.unit_anchors[driven_types[0]].expand(model.QUERY_COUNT, -1, -1),
        logged_futures,
        torch.ones(model.QUERY_COUNT, contract.FUTURE_STEPS, dtype=torch.bool),
    )

    assert torch.equal(assigned_mode, torch.arange(model.QUERY_COUNT))


def test_two_agents_alike_but_for_their_object_type_get_their_own_types_anchor_set():
    torch.manual_seed(71)
    geometric_fan = model.unit_anchor_offsets() * 40.0
    vehicle_index = contract.PREDICTED_OBJECT_TYPES.index("TYPE_VEHICLE")
    pedestrian_index = contract.PREDICTED_OBJECT_TYPES.index("TYPE_PEDESTRIAN")
    unit_anchors = model.unit_anchor_offsets_per_type() * 40.0
    unit_anchors[pedestrian_index] = 0.6 * geometric_fan.flip(0)
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.zero_()
        predictor.mode_decoder.trajectory_head[-1].bias.zero_()

    agent_history = torch.randn(
        1, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM
    ).repeat(2, 1, 1)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor([9.0, 0.0])
    agent_history[:, :, contract.AGENT_TYPE] = 0.0
    agent_history[0, :, contract.AGENT_TYPE.start + vehicle_index] = 1.0
    agent_history[1, :, contract.AGENT_TYPE.start + pedestrian_index] = 1.0
    map_rows = torch.randn(10, contract.MAP_FEATURE_DIM).repeat(2, 1)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(
            1, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM
        ).repeat(2, 1, 1, 1),
        "neighbour_history_mask": torch.ones(2, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.repeat_interleave(torch.tensor([0, 2]), 10),
        "map_chunk_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(2, 2, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(2),
    }

    with torch.no_grad():
        (
            trajectories, _, _, _, _, _, selected_unit_anchors, _, _, _,
        ) = predictor.predict_with_heading(batch)
    endpoints = trajectories[:, :, -1]

    assert torch.allclose(selected_unit_anchors[0], unit_anchors[vehicle_index], atol=1e-6)
    assert torch.allclose(selected_unit_anchors[1], unit_anchors[pedestrian_index], atol=1e-6)
    assert torch.allclose(endpoints[0], unit_anchors[vehicle_index], atol=1e-4)
    assert torch.allclose(endpoints[1], unit_anchors[pedestrian_index], atol=1e-4)
    assert float((endpoints[0] - endpoints[1]).norm(dim=-1).min()) > 1.0

    shared_target_mode = 3
    logged_endpoint = unit_anchors[vehicle_index, shared_target_mode]
    ramp = torch.arange(1, contract.FUTURE_STEPS + 1) / contract.FUTURE_STEPS
    future_positions = (logged_endpoint[None, :] * ramp[:, None]).expand(2, -1, -1)
    assigned_mode = loss.anchor_assigned_mode(
        selected_unit_anchors, future_positions,
        torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
    )

    assert int(assigned_mode[0]) == shared_target_mode
    assert int(assigned_mode[1]) != shared_target_mode
    assert int(assigned_mode[1]) == int(
        (unit_anchors[pedestrian_index] - logged_endpoint).norm(dim=-1).argmin()
    )


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


def test_one_training_step_runs_the_whole_path_over_staged_scenarios(tmp_path):
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:2]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")

    torch.manual_seed(0)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    batch = next(iter(pipeline.batches(scenario_paths, 0, 2, 0, 0, True)))

    (
        trajectories, heading_cosine_sine, position_log_standard_deviation,
        heading_log_standard_deviation, confidence_logits, predicted_speed,
        selected_unit_anchors, mode_valid, neighbour_future_positions,
        neighbour_log_standard_deviation,
    ) = predictor.predict_with_heading(batch)
    assert neighbour_future_positions.shape == (
        batch["neighbour_future_positions"].shape[:2] + (contract.FUTURE_STEPS, 2)
    )
    assert selected_unit_anchors.shape == (
        batch["agent_history"].shape[0], model.QUERY_COUNT, 2
    )
    components = loss.prediction_loss(
        trajectories, heading_cosine_sine, position_log_standard_deviation,
        heading_log_standard_deviation, confidence_logits, predicted_speed,
        batch["future_positions"], batch["future_headings"], batch["future_mask"],
        selected_unit_anchors, 1.0, 1.0, 1.0, mode_valid,
    )
    neighbour_future = loss.neighbour_future_loss(
        neighbour_future_positions, neighbour_log_standard_deviation,
        batch["neighbour_future_positions"],
        batch["neighbour_future_mask"],
        batch["neighbour_history_mask"].any(dim=-1),
    )
    assert torch.stack(components).isfinite().all()
    assert torch.isfinite(neighbour_future)
    (components[0] + neighbour_future).backward()
    starved_parameters = [
        name for name, parameter in predictor.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0)
    ]
    assert not starved_parameters, (
        f"{len(starved_parameters)} parameters took no gradient from one step over staged"
        f" scenarios: {starved_parameters}. A parameter the loss cannot reach is an unwired head,"
        f" and an unwired head is exactly what a per-parameter check exists to expose."
    )
    optimizer.step()

    accumulator = metrics.MetricAccumulator()
    accumulator.update(
        trajectories.detach(), confidence_logits.detach(),
        batch["future_positions"], batch["future_mask"],
    )
    assert all(math.isfinite(value) for value in accumulator.results().values())

    checkpoint_path = tmp_path / "predictor.pt"
    torch.save(predictor.state_dict(), checkpoint_path)
    reloaded_state = torch.load(checkpoint_path)
    assert all(
        torch.equal(reloaded_state[name], parameter)
        for name, parameter in predictor.state_dict().items()
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


def test_the_heading_rides_one_bernstein_curve_of_its_own_control_pairs():
    torch.manual_seed(43)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    with torch.no_grad():
        decoder.trajectory_head[-1].weight.mul_(20.0)
    tokens = torch.randn(2, 9, model.HIDDEN_DIM)
    token_present = torch.ones(2, 9, dtype=torch.bool)

    with torch.no_grad():
        round_outputs, _, _ = decoder(tokens, token_present, torch.zeros(2, dtype=torch.long))
        heading_cosine_sine = round_outputs[-1][1]

    time_fractions = (
        torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32) / contract.FUTURE_STEPS
    )
    degree = model.TRAJECTORY_CONTROL_POINTS
    independent_basis = torch.stack(
        [
            math.comb(degree, index)
            * time_fractions ** index
            * (1.0 - time_fractions) ** (degree - index)
            for index in range(1, degree + 1)
        ],
        dim=1,
    )
    assert heading_cosine_sine.shape == (2, model.QUERY_COUNT, contract.FUTURE_STEPS, 2)

    heading_at_now = torch.tensor(model.HEADING_AT_NOW)
    flattened = (heading_cosine_sine - heading_at_now).permute(2, 0, 1, 3).reshape(
        contract.FUTURE_STEPS, -1
    )
    fitted_control_pairs = torch.linalg.lstsq(independent_basis, flattened).solution
    reconstructed = independent_basis @ fitted_control_pairs
    assert flattened.abs().max() > 0.1
    assert (reconstructed - flattened).norm() < 1e-4 * flattened.norm()

    control_polygon = torch.cat(
        [
            torch.zeros(2, model.QUERY_COUNT, 1, 2),
            fitted_control_pairs.view(degree, 2, model.QUERY_COUNT, 2).permute(1, 2, 0, 3),
        ],
        dim=2,
    ) + heading_at_now
    walked_heading = torch.cat(
        [heading_at_now.expand(2, model.QUERY_COUNT, 1, 2), heading_cosine_sine], dim=2
    )
    assert (
        total_turning_degrees(walked_heading) <= total_turning_degrees(control_polygon) + 1e-2
    ).all()

    smooth_control_angles = torch.tensor([0.05, 0.20, 0.30])
    smooth_curve = independent_basis @ torch.stack(
        [smooth_control_angles.cos(), smooth_control_angles.sin()], dim=-1
    )
    step_angle_changes = torch.atan2(smooth_curve[:, 1], smooth_curve[:, 0]).diff().abs()
    assert step_angle_changes.max() <= smooth_control_angles.max() - smooth_control_angles.min()


def test_the_step_uncertainty_rides_one_bernstein_curve_and_never_leaves_the_clamped_range():
    torch.manual_seed(59)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    tokens = torch.randn(2, 7, model.HIDDEN_DIM)
    token_present = torch.ones(2, 7, dtype=torch.bool)
    log_standard_deviation_start = model.POSITION_CONTROL_VALUES + model.HEADING_CONTROL_VALUES
    log_standard_deviation_end = (
        log_standard_deviation_start + model.LOG_STANDARD_DEVIATION_CONTROL_VALUES
    )
    moderate_control_pairs = torch.tensor([[-1.0, 0.5], [0.4, -0.2], [1.2, 0.9]])

    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        decoder.trajectory_head[-1].bias.zero_()
        decoder.trajectory_head[-1].bias[log_standard_deviation_start:log_standard_deviation_end] = (
            moderate_control_pairs.flatten()
        )
        round_outputs, _, _ = decoder(tokens, token_present, torch.zeros(2, dtype=torch.long))
        moderate_log_standard_deviation = round_outputs[-1][2]

    time_fractions = (
        torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32) / contract.FUTURE_STEPS
    )
    degree = model.TRAJECTORY_CONTROL_POINTS
    independent_basis = torch.stack(
        [
            math.comb(degree, index)
            * time_fractions ** index
            * (1.0 - time_fractions) ** (degree - index)
            for index in range(1, degree + 1)
        ],
        dim=1,
    )

    assert moderate_log_standard_deviation.shape == (
        2, model.QUERY_COUNT, contract.FUTURE_STEPS, 2
    )
    floor_at_now = model.MINIMUM_LOG_STANDARD_DEVIATION * (1.0 - time_fractions) ** degree
    assert torch.allclose(
        moderate_log_standard_deviation,
        floor_at_now[:, None] + independent_basis @ moderate_control_pairs,
        atol=1e-5,
    )
    assert float(moderate_log_standard_deviation[..., 0, :].min()) < -1.5

    with torch.no_grad():
        decoder.trajectory_head[-1].bias[log_standard_deviation_start:log_standard_deviation_end] = (
            torch.tensor([-1e3, 1e3, -1e3, 1e3, -1e3, 1e3])
        )
        round_outputs, _, _ = decoder(tokens, token_present, torch.zeros(2, dtype=torch.long))
        extreme_log_standard_deviation = round_outputs[-1][2]
    extreme_standard_deviation = extreme_log_standard_deviation.exp()
    floor_metres = math.exp(model.MINIMUM_LOG_STANDARD_DEVIATION)
    ceiling_metres = math.exp(model.MAXIMUM_LOG_STANDARD_DEVIATION)

    assert (extreme_standard_deviation >= floor_metres).all()
    assert (extreme_standard_deviation <= ceiling_metres).all()
    assert float(extreme_standard_deviation.min()) == pytest.approx(floor_metres, rel=1e-6)
    assert float(extreme_standard_deviation.max()) == pytest.approx(ceiling_metres, rel=1e-6)
    assert floor_metres == pytest.approx(0.2, abs=1e-3)


def test_the_regression_term_is_the_gaussian_negative_log_likelihood_at_the_stated_uncertainty():
    torch.manual_seed(53)
    sample_count = 4
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    future_headings = torch.nn.functional.normalize(
        torch.randn(sample_count, contract.FUTURE_STEPS, 2), dim=-1
    )
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    heading_cosine_sine = future_headings.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    confidence_logits = torch.zeros(sample_count, model.QUERY_COUNT)
    unit_anchors = model.unit_anchor_offsets().expand(sample_count, -1, -1)
    displacement = torch.tensor([0.3, -0.4])
    predicted_speed = torch.zeros(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS)

    def regression_at(log_standard_deviation, error):
        trajectories = (
            future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1) + error
        )
        _, regression, _, _, _ = loss.prediction_loss(
            trajectories, heading_cosine_sine,
            torch.full_like(trajectories, log_standard_deviation),
            torch.zeros_like(trajectories), confidence_logits,
            predicted_speed,
            future_positions, future_headings, future_mask,
            unit_anchors, 1.0, 1.0, 1.0,
        )
        return float(regression)

    floor = model.MINIMUM_LOG_STANDARD_DEVIATION
    floor_standard_deviation = math.exp(floor)
    squared_distance_form = contract.FUTURE_STEPS * (
        2.0 * floor
        + 2.0 * loss.HALF_LOG_TWO_PI
        + 0.5 * float((displacement ** 2).sum()) / floor_standard_deviation ** 2
    )

    assert regression_at(floor, displacement) == pytest.approx(squared_distance_form, rel=1e-5)
    for log_standard_deviation in (floor, 0.0, model.MAXIMUM_LOG_STANDARD_DEVIATION):
        assert regression_at(log_standard_deviation, torch.zeros(2)) < regression_at(
            log_standard_deviation, displacement
        )


def test_a_straight_constant_speed_control_polygon_predicts_that_constant_speed_analytically():
    torch.manual_seed(61)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    control_point_spacing = torch.tensor([3.0, 4.0])
    equally_spaced_control_points = control_point_spacing * torch.arange(
        1, model.TRAJECTORY_CONTROL_POINTS + 1, dtype=torch.float32
    )[:, None]

    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        decoder.trajectory_head[-1].bias.zero_()
        decoder.trajectory_head[-1].bias[: model.POSITION_CONTROL_VALUES] = (
            equally_spaced_control_points.flatten()
        )
        normed_tokens = decoder.scene_norm(torch.randn(2, 7, model.HIDDEN_DIM))
        token_present = torch.ones(2, 7, dtype=torch.bool)
        predicted_speed = decoder.decode_from_anchors(
            normed_tokens, token_present, torch.zeros(2, model.QUERY_COUNT, 2),
            torch.ones(2, model.QUERY_COUNT, dtype=torch.bool), 2,
        )[-1][5]

    expected_speed = float(
        (model.TRAJECTORY_CONTROL_POINTS * control_point_spacing).norm()
        / contract.FUTURE_HORIZON_SECONDS
    )
    assert torch.allclose(predicted_speed, torch.full_like(predicted_speed, expected_speed), atol=1e-4)


def test_predicted_speed_carries_the_anchor_motion_the_emitted_trajectory_carries():
    torch.manual_seed(67)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    anchor_endpoint = torch.tensor([24.0, -7.0])

    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        decoder.trajectory_head[-1].bias.zero_()
        normed_tokens = decoder.scene_norm(torch.randn(2, 7, model.HIDDEN_DIM))
        token_present = torch.ones(2, 7, dtype=torch.bool)
        anchored_position, _, _, _, _, predicted_speed = decoder.decode_from_anchors(
            normed_tokens, token_present,
            anchor_endpoint.expand(2, model.QUERY_COUNT, 2),
            torch.ones(2, model.QUERY_COUNT, dtype=torch.bool), 2,
        )[-1]

    assert torch.allclose(
        anchored_position[:, :, -1], anchor_endpoint.expand(2, model.QUERY_COUNT, 2), atol=1e-4
    )
    anchor_only_speed = float(anchor_endpoint.norm() / contract.FUTURE_HORIZON_SECONDS)
    assert torch.allclose(
        predicted_speed, torch.full_like(predicted_speed, anchor_only_speed), atol=1e-4
    )


def test_every_feature_column_the_contract_declares_has_a_sensitivity_perturbation():
    measure_sensitivity.assert_selectors_cover_every_column(
        measure_sensitivity.AGENT_FEATURE_SELECTORS, contract.AGENT_FEATURE_DIM, "agent"
    )
    measure_sensitivity.assert_selectors_cover_every_column(
        measure_sensitivity.MAP_FEATURE_SELECTORS, contract.MAP_FEATURE_DIM, "map"
    )
    measure_sensitivity.assert_selectors_cover_every_column(
        measure_sensitivity.LANE_CONTEXT_SELECTORS, contract.LANE_CONTEXT_DIM, "lane context"
    )


def test_heading_term_is_the_gaussian_negative_log_likelihood_on_the_unit_pair():
    torch.manual_seed(73)
    sample_count = 3
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    future_headings = torch.nn.functional.normalize(
        torch.randn(sample_count, contract.FUTURE_STEPS, 2), dim=-1
    )
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    trajectories = future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    log_standard_deviation = torch.zeros_like(trajectories)
    confidence_logits = torch.zeros(sample_count, model.QUERY_COUNT)
    unit_anchors = model.unit_anchor_offsets().expand(sample_count, -1, -1)
    predicted_speed = torch.zeros(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS)

    def heading_term(heading_cosine_sine):
        _, _, heading, _, _ = loss.prediction_loss(
            trajectories, heading_cosine_sine, log_standard_deviation,
            torch.zeros_like(trajectories), confidence_logits,
            predicted_speed,
            future_positions, future_headings, future_mask,
            unit_anchors, 1.0, 1.0, 1.0,
        )
        return float(heading)

    exact = future_headings.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    quarter_turn = torch.stack([-exact[..., 1], exact[..., 0]], dim=-1)
    shrunken_exact = 1e-6 * exact
    exact_nll = 2.0 * loss.HALF_LOG_TWO_PI * contract.FUTURE_STEPS

    assert heading_term(exact) == pytest.approx(exact_nll, rel=1e-5)
    assert heading_term(quarter_turn) > heading_term(exact)
    assert heading_term(shrunken_exact) == pytest.approx(exact_nll, rel=1e-4)


def test_heading_log_sigma_cannot_exceed_the_unit_pair_support():
    torch.manual_seed(59)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    heading_log_start = (
        model.POSITION_CONTROL_VALUES
        + model.HEADING_CONTROL_VALUES
        + model.LOG_STANDARD_DEVIATION_CONTROL_VALUES
    )
    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        decoder.trajectory_head[-1].bias.zero_()
        decoder.trajectory_head[-1].bias[heading_log_start:] = 1e3
        round_outputs, _, _ = decoder(
            torch.randn(1, 5, model.HIDDEN_DIM),
            torch.ones(1, 5, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
        )
        heading_log_standard_deviation = round_outputs[-1][3]
    assert heading_log_standard_deviation.max() == pytest.approx(
        model.HEADING_MAXIMUM_LOG_STANDARD_DEVIATION, abs=1e-5
    )
    assert math.exp(model.HEADING_MAXIMUM_LOG_STANDARD_DEVIATION) == pytest.approx(2.0)


def test_classification_is_softmax_cross_entropy_on_the_assigned_mode():
    sample_count = 3
    future_positions = torch.zeros(sample_count, contract.FUTURE_STEPS, 2)
    future_positions[:, -1, 0] = 1.0
    future_headings = torch.zeros(sample_count, contract.FUTURE_STEPS, 2)
    future_headings[..., 0] = 1.0
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    trajectories = future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    heading_cosine_sine = future_headings.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    unit_anchors = torch.zeros(sample_count, model.QUERY_COUNT, 2)
    unit_anchors[:, 0, 0] = 1.0
    confidence_logits = torch.zeros(sample_count, model.QUERY_COUNT)
    _, _, _, classification, _ = loss.prediction_loss(
        trajectories, heading_cosine_sine, torch.zeros_like(trajectories),
        torch.zeros_like(trajectories), confidence_logits,
        torch.zeros(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS),
        future_positions, future_headings, future_mask,
        unit_anchors, 1.0, 1.0, 1.0,
    )
    assert float(classification) == pytest.approx(math.log(model.QUERY_COUNT), rel=1e-5)


def test_speed_term_is_gaussian_nll_at_the_measured_velocity_scale():
    sample_count = 2
    step_speed = contract.VELOCITY_NORMALISER_METRES_PER_SECOND
    future_positions = torch.zeros(sample_count, contract.FUTURE_STEPS, 2)
    future_positions[..., 0] = step_speed * contract.TIMESTEP_SECONDS * (
        torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32)
    )
    future_headings = torch.zeros(sample_count, contract.FUTURE_STEPS, 2)
    future_headings[..., 0] = 1.0
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    trajectories = future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    heading_cosine_sine = future_headings.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    unit_anchors = torch.zeros(sample_count, model.QUERY_COUNT, 2)
    unit_anchors[:, 0] = future_positions[:, -1]
    exact_speed = torch.full(
        (sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS), step_speed
    )
    doubled_speed = exact_speed * 2.0

    def speed_term(predicted_speed):
        _, _, _, _, speed = loss.prediction_loss(
            trajectories, heading_cosine_sine, torch.zeros_like(trajectories),
            torch.zeros_like(trajectories), torch.zeros(sample_count, model.QUERY_COUNT),
            predicted_speed, future_positions, future_headings, future_mask,
            unit_anchors, 1.0, 1.0, 1.0,
        )
        return float(speed)

    exact_nll = (math.log(step_speed) + loss.HALF_LOG_TWO_PI) * contract.FUTURE_STEPS
    one_sigma_nll = exact_nll + 0.5 * contract.FUTURE_STEPS
    assert speed_term(exact_speed) == pytest.approx(exact_nll, rel=1e-5)
    assert speed_term(doubled_speed) == pytest.approx(one_sigma_nll, rel=1e-5)


def synthetic_scene_batch(sample_count, neighbour_count, polyline_count, dots_per_polyline):
    agent_history = torch.randn(sample_count, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, :, contract.AGENT_TYPE] = 0.0
    for sample_index in range(sample_count):
        agent_history[
            sample_index, :,
            contract.AGENT_TYPE.start + sample_index % contract.NUM_OBJECT_TYPES,
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
        "neighbour_future_positions": torch.randn(
            sample_count, neighbour_count, contract.FUTURE_STEPS, 2
        ),
        "neighbour_future_mask": torch.ones(
            sample_count, neighbour_count, contract.FUTURE_STEPS, dtype=torch.bool
        ),
        "agent_signal_history": torch.rand(
            sample_count, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.rand(
            sample_count, neighbour_count, contract.HISTORY_STEPS,
            contract.NUM_TRAFFIC_SIGNAL_STATES,
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.arange(dot_count) // dots_per_polyline,
        "map_chunk_signal_history": torch.rand(
            sample_count, polyline_count, contract.HISTORY_STEPS,
            contract.NUM_TRAFFIC_SIGNAL_STATES,
        ),
        "map_chunk_lane_context": torch.randn(
            sample_count, polyline_count, contract.LANE_CONTEXT_DIM
        ),
        "max_polylines_in_batch": torch.tensor(polyline_count),
        "future_positions": torch.randn(sample_count, contract.FUTURE_STEPS, 2),
        "future_headings": torch.nn.functional.normalize(
            torch.randn(sample_count, contract.FUTURE_STEPS, 2), dim=-1
        ),
        "future_mask": torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool),
    }


EMITTED_QUANTITY_NAMES = (
    "trajectories",
    "heading_cosine_sine",
    "position_log_standard_deviation",
    "heading_log_standard_deviation",
    "confidence_logits",
    "predicted_speed",
    "selected_unit_anchors",
    "mode_valid",
    "neighbour_future_positions",
    "neighbour_log_standard_deviation",
)


def turned_a_quarter_circle(cosine_sine):
    return torch.stack([-cosine_sine[..., 1], cosine_sine[..., 0]], dim=-1)


def combined_training_loss(emitted_quantities, batch):
    (
        trajectories, heading_cosine_sine, position_log_standard_deviation,
        heading_log_standard_deviation, confidence_logits, predicted_speed,
        selected_unit_anchors, mode_valid, neighbour_future_positions,
        neighbour_log_standard_deviation,
    ) = emitted_quantities
    total, _, _, _, _ = loss.prediction_loss(
        trajectories, heading_cosine_sine, position_log_standard_deviation,
        heading_log_standard_deviation, confidence_logits, predicted_speed,
        batch["future_positions"], batch["future_headings"], batch["future_mask"],
        selected_unit_anchors, 1.0, 1.0, 1.0, mode_valid,
    )
    return total + loss.neighbour_future_loss(
        neighbour_future_positions, neighbour_log_standard_deviation,
        batch["neighbour_future_positions"],
        batch["neighbour_future_mask"],
        batch["neighbour_history_mask"].any(dim=-1),
    )


def test_every_quantity_the_model_emits_is_pinned_by_the_loss_that_trains_it():
    torch.manual_seed(83)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 3, 2, 10)
    with torch.no_grad():
        emitted_quantities = list(predictor.predict_with_heading(batch))

    assert len(emitted_quantities) == len(EMITTED_QUANTITY_NAMES), (
        f"predict_with_heading emits {len(emitted_quantities)} quantities but the free-knob audit"
        f" names {len(EMITTED_QUANTITY_NAMES)}: {EMITTED_QUANTITY_NAMES}. Every emitted quantity"
        f" needs something that pins it to reality, so a new one is a decision to make here, not"
        f" an addition that slides through."
    )

    perturbation_of_quantity = {
        "trajectories": lambda value: value + 1.0,
        "heading_cosine_sine": turned_a_quarter_circle,
        "position_log_standard_deviation": lambda value: value + 1.0,
        "heading_log_standard_deviation": lambda value: value + 1.0,
        "confidence_logits": lambda value: torch.cat(
            [value[..., :1] + 1.0, value[..., 1:]], dim=-1
        ),
        "predicted_speed": lambda value: value + 1.0,
        "selected_unit_anchors": lambda value: -value,
        "mode_valid": lambda value: torch.zeros_like(value),
        "neighbour_future_positions": lambda value: value + 1.0,
        "neighbour_log_standard_deviation": lambda value: value + 1.0,
    }
    assert set(perturbation_of_quantity) == set(EMITTED_QUANTITY_NAMES)
    by_name = dict(zip(EMITTED_QUANTITY_NAMES, emitted_quantities))
    assert by_name["mode_valid"].dtype == torch.bool
    assert by_name["confidence_logits"].shape == (contract.NUM_OBJECT_TYPES, model.QUERY_COUNT)
    assert by_name["selected_unit_anchors"].shape == (contract.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2)
    assert by_name["predicted_speed"].shape == (
        contract.NUM_OBJECT_TYPES, model.QUERY_COUNT, contract.FUTURE_STEPS
    )
    assert by_name["neighbour_log_standard_deviation"].shape == by_name["neighbour_future_positions"].shape
    assert by_name["heading_log_standard_deviation"].shape == by_name["heading_cosine_sine"].shape

    unperturbed_total = combined_training_loss(emitted_quantities, batch)
    for position, quantity_name in enumerate(EMITTED_QUANTITY_NAMES):
        perturbed_quantities = list(emitted_quantities)
        perturbed_quantities[position] = perturbation_of_quantity[quantity_name](
            emitted_quantities[position]
        )
        assert not torch.equal(
            combined_training_loss(perturbed_quantities, batch), unperturbed_total
        ), (
            f"perturbing {quantity_name} on its own left the training loss bit-identical at"
            f" {float(unperturbed_total)}: nothing the optimiser answers to pins it, so it is a"
            f" free knob the optimiser will turn instead of solving the problem."
        )


def test_the_heading_pairs_length_is_the_one_emitted_quantity_the_loss_cannot_see():
    torch.manual_seed(83)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 3, 2, 10)
    with torch.no_grad():
        emitted_quantities = list(predictor.predict_with_heading(batch))

    heading_position = EMITTED_QUANTITY_NAMES.index("heading_cosine_sine")
    lengthened_quantities = list(emitted_quantities)
    lengthened_quantities[heading_position] = emitted_quantities[heading_position] * 2.0

    assert torch.equal(
        combined_training_loss(lengthened_quantities, batch),
        combined_training_loss(emitted_quantities, batch),
    )
    assert not torch.equal(
        combined_training_loss(
            [
                turned_a_quarter_circle(quantity) if position == heading_position else quantity
                for position, quantity in enumerate(emitted_quantities)
            ],
            batch,
        ),
        combined_training_loss(emitted_quantities, batch),
    )


def test_the_position_likelihood_never_reads_the_predicted_heading():
    torch.manual_seed(89)
    sample_count = 4
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    future_headings = torch.nn.functional.normalize(
        torch.randn(sample_count, contract.FUTURE_STEPS, 2), dim=-1
    )
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    trajectories = torch.randn(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS, 2)
    position_log_standard_deviation = torch.randn_like(trajectories)
    confidence_logits = torch.randn(sample_count, model.QUERY_COUNT)
    predicted_speed = torch.rand(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS)
    unit_anchors = model.unit_anchor_offsets().expand(sample_count, -1, -1) * 40.0
    heading_cosine_sine = torch.randn_like(trajectories)

    def regression_and_heading_terms(predicted_pair):
        _, regression, heading, _, _ = loss.prediction_loss(
            trajectories, predicted_pair, position_log_standard_deviation,
            torch.zeros_like(trajectories), confidence_logits,
            predicted_speed,
            future_positions, future_headings, future_mask,
            unit_anchors, 1.0, 1.0, 1.0,
        )
        return regression, heading

    regression, heading = regression_and_heading_terms(heading_cosine_sine)
    turned_regression, turned_heading = regression_and_heading_terms(
        turned_a_quarter_circle(heading_cosine_sine)
    )

    assert torch.equal(regression, turned_regression)
    assert not torch.equal(heading, turned_heading)


def test_no_step_uncertainty_starts_saturated_and_both_axes_start_carrying_spread():
    torch.manual_seed(97)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)

    with torch.no_grad():
        (
            _, _, position_log_standard_deviation, heading_log_standard_deviation,
            _, _, _, _, _, _,
        ) = predictor.predict_with_heading(batch)

    saturated_low = position_log_standard_deviation == model.MINIMUM_LOG_STANDARD_DEVIATION
    saturated_high = position_log_standard_deviation == model.MAXIMUM_LOG_STANDARD_DEVIATION
    heading_at_ceiling = (
        heading_log_standard_deviation == model.HEADING_MAXIMUM_LOG_STANDARD_DEVIATION
    )

    assert not saturated_low.any()
    assert not saturated_high.any()
    assert not heading_at_ceiling.any()
    for axis in range(2):
        axis_values = position_log_standard_deviation[..., axis]
        assert float(axis_values.min()) < float(axis_values.max())
    assert not torch.equal(
        position_log_standard_deviation[..., 0], position_log_standard_deviation[..., 1]
    )


def test_v2_a_masked_future_step_moves_no_term_of_the_prediction_loss():
    torch.manual_seed(103)
    sample_count = 4
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    future_mask[1, contract.FUTURE_STEPS // 2:] = False
    future_mask[2] = False
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    future_headings = torch.nn.functional.normalize(
        torch.randn(sample_count, contract.FUTURE_STEPS, 2), dim=-1
    )
    trajectories = torch.randn(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS, 2)
    heading_cosine_sine = torch.randn_like(trajectories)
    position_log_standard_deviation = torch.randn_like(trajectories)
    heading_log_standard_deviation = torch.randn_like(trajectories)
    confidence_logits = torch.randn(sample_count, model.QUERY_COUNT)
    predicted_speed = torch.rand(sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS)
    unit_anchors = model.unit_anchor_offsets().expand(sample_count, -1, -1) * 40.0

    def loss_components(logged_positions, logged_headings, predicted_positions, predicted_speeds):
        return torch.stack(loss.prediction_loss(
            predicted_positions, heading_cosine_sine, position_log_standard_deviation,
            heading_log_standard_deviation, confidence_logits, predicted_speeds,
            logged_positions, logged_headings, future_mask,
            unit_anchors, 1.0, 1.0, 1.0,
        ))

    masked_steps = ~future_mask
    masked_mode_steps = masked_steps.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1)
    poisoned_positions = future_positions.clone()
    poisoned_positions[masked_steps] = 1e6
    poisoned_headings = future_headings.clone()
    poisoned_headings[masked_steps] = 1e6
    poisoned_trajectories = trajectories.clone()
    poisoned_trajectories[masked_mode_steps] = 1e6
    poisoned_speed = predicted_speed.clone()
    poisoned_speed[masked_mode_steps] = 1e6

    assert future_mask[2].sum() == 0
    assert masked_steps.any() and future_mask.any()
    assert torch.equal(
        loss_components(future_positions, future_headings, trajectories, predicted_speed),
        loss_components(
            poisoned_positions, poisoned_headings, poisoned_trajectories, poisoned_speed
        ),
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
            lane_connections=np.zeros((0, contract.LANE_CONNECTION_WIDTH), dtype=np.int64),
            lane_neighbour_ids=np.zeros((0, contract.LANE_NEIGHBOUR_ID_WIDTH), dtype=np.int64),
            lane_neighbour_bounds=np.zeros(
                (0, contract.LANE_NEIGHBOUR_BOUND_WIDTH), dtype=np.float32
            ),
            lane_boundary_ids=np.zeros((0, contract.LANE_BOUNDARY_ID_WIDTH), dtype=np.int64),
            lane_boundary_bounds=np.zeros(
                (0, contract.LANE_BOUNDARY_BOUND_WIDTH), dtype=np.float32
            ),
            stop_sign_controlled_lanes=np.zeros((0, contract.STOP_SIGN_LANE_WIDTH), dtype=np.int64),
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


def test_untrained_modes_are_finite_and_not_the_same_point():
    torch.manual_seed(107)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)
    with torch.no_grad():
        trajectories, heading_cosine_sine, _, _, _, predicted_speed, _, _, _, _ = (
            predictor.predict_with_heading(batch)
        )
    assert trajectories.isfinite().all()
    assert heading_cosine_sine.isfinite().all()
    assert predicted_speed.isfinite().all()
    endpoints = trajectories[:, :, -1]
    assert not torch.allclose(endpoints[:, :1], endpoints, atol=1e-4)


def test_constant_velocity_and_the_model_are_both_finite_on_the_same_batch():
    torch.manual_seed(109)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)
    null_trajectories, null_logits = baseline.constant_velocity(batch)
    with torch.no_grad():
        model_trajectories, model_logits = predictor(batch)
    assert null_trajectories.isfinite().all() and null_logits.isfinite().all()
    assert model_trajectories.isfinite().all() and model_logits.isfinite().all()


def test_autocast_forward_keeps_every_likelihood_input_in_float32_and_steps_stay_finite():
    torch.manual_seed(113)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type() * 300.0)
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    scaler = train.GradScaler(enabled=False)
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 2, 2, 10)
    batch["neighbour_history"][..., contract.AGENT_POSITION] += 300.0
    for _ in range(3):
        with torch.amp.autocast(device_type="cpu", enabled=True):
            (
                round_outputs, selected_unit_anchors, mode_valid,
                neighbour_future_positions, neighbour_log_standard_deviation,
            ) = predictor.predict_every_round(batch)
            for round_output in round_outputs:
                trajectories, _, position_log_standard_deviation, heading_log_standard_deviation, _, _ = round_output
                assert trajectories.dtype == torch.float32
                assert position_log_standard_deviation.dtype == torch.float32
                assert heading_log_standard_deviation.dtype == torch.float32
            assert neighbour_future_positions.dtype == torch.float32
            total, *_ = train.round_summed_prediction_loss(
                round_outputs, batch, selected_unit_anchors, mode_valid,
                train.HEADING_LOSS_WEIGHT,
                train.CLASSIFICATION_LOSS_WEIGHT,
                train.SPEED_LOSS_WEIGHT,
            )
            total = total + train.NEIGHBOUR_FUTURE_LOSS_WEIGHT * loss.neighbour_future_loss(
                neighbour_future_positions, neighbour_log_standard_deviation,
                batch["neighbour_future_positions"],
                batch["neighbour_future_mask"],
                batch["neighbour_history_mask"].any(dim=-1),
            )
        assert total.dtype == torch.float32
        assert torch.isfinite(total)
        optimizer.zero_grad()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(predictor.parameters(), float("inf")))
        assert math.isfinite(gradient_norm)
        scaler.step(optimizer)
        scaler.update()


def test_a_zeroed_head_emits_the_known_now_boundary_heading_straight_ahead_at_the_sigma_floor():
    torch.manual_seed(115)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type() * 40.0).eval()
    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        decoder.trajectory_head[-1].bias.zero_()
        round_outputs, _, _ = decoder(
            torch.randn(2, 5, model.HIDDEN_DIM),
            torch.ones(2, 5, dtype=torch.bool),
            torch.zeros(2, dtype=torch.long),
        )
    _, heading_cosine_sine, position_log_standard_deviation, heading_log_standard_deviation, _, _ = (
        round_outputs[-1]
    )
    assert torch.allclose(
        heading_cosine_sine, torch.tensor(model.HEADING_AT_NOW).expand_as(heading_cosine_sine)
    )
    first_step_weight = (1.0 - 1.0 / contract.FUTURE_STEPS) ** model.TRAJECTORY_CONTROL_POINTS
    for log_standard_deviation in (position_log_standard_deviation, heading_log_standard_deviation):
        assert float(log_standard_deviation[..., 0, :].min()) == pytest.approx(
            model.MINIMUM_LOG_STANDARD_DEVIATION * first_step_weight, rel=1e-5
        )
        assert float(log_standard_deviation[..., -1, :].max()) == pytest.approx(0.0, abs=1e-6)
        assert (log_standard_deviation.diff(dim=-2) >= 0.0).all()


def test_neighbour_futures_are_displacements_from_the_neighbours_last_seen_position():
    torch.manual_seed(117)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = synthetic_scene_batch(contract.NUM_OBJECT_TYPES, 3, 2, 10)
    batch["neighbour_history_mask"][:, 1, contract.CURRENT_STEP_INDEX - 2:] = False
    with torch.no_grad():
        predictor.neighbour_future_head.network[-1].weight.zero_()
        predictor.neighbour_future_head.network[-1].bias.zero_()
        _, _, _, _, _, _, _, _, neighbour_future_positions, _ = predictor.predict_with_heading(batch)
    now_positions = batch["neighbour_history"][:, :, contract.CURRENT_STEP_INDEX, contract.AGENT_POSITION]
    last_seen_positions = batch["neighbour_history"][
        :, :, contract.CURRENT_STEP_INDEX - 3, contract.AGENT_POSITION
    ]
    assert torch.allclose(
        neighbour_future_positions[:, 0], now_positions[:, 0, None, :].expand(-1, contract.FUTURE_STEPS, -1)
    )
    assert torch.allclose(
        neighbour_future_positions[:, 1],
        last_seen_positions[:, 1, None, :].expand(-1, contract.FUTURE_STEPS, -1),
    )


def test_anchors_are_frozen_buffers_carried_by_the_checkpoint():
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type())
    parameter_names = {name for name, _ in predictor.named_parameters()}
    assert "mode_decoder.anchor_offsets" not in parameter_names
    assert "mode_decoder.anchor_offsets" in predictor.state_dict()
    assert "mode_decoder.anchor_counts" in predictor.state_dict()
    assert not predictor.unit_anchors.requires_grad


def test_the_anchor_count_search_keeps_every_anchor_clear_of_the_prune_radius():
    random_generator = np.random.default_rng(19)
    spread_centres = 3.0 * torch.stack(
        [torch.arange(model.QUERY_COUNT) % 9, torch.arange(model.QUERY_COUNT) // 9], dim=1
    ).float()
    spread_endpoints = (
        spread_centres[:, None, :]
        + torch.tensor(random_generator.normal(0.0, 0.01, (model.QUERY_COUNT, 20, 2)), dtype=torch.float32)
    ).reshape(-1, 2)
    spread_count, (spread_fitted, *_) = fit_anchors.largest_count_kept_apart_by_the_prune(
        spread_endpoints, "spread"
    )
    assert spread_count == model.QUERY_COUNT
    assert fit_anchors.minimum_pairwise_distance(spread_fitted) >= model.PRUNE_DISTANCE_METRES

    crowded_line = 1.3 * torch.stack([torch.arange(18, dtype=torch.float32), torch.zeros(18)], dim=1)
    crowded_endpoints = (
        crowded_line[:, None, :]
        + torch.tensor(random_generator.normal(0.0, 0.01, (18, 60, 2)), dtype=torch.float32)
    ).reshape(-1, 2)
    crowded_count, (crowded_fitted, *_) = fit_anchors.largest_count_kept_apart_by_the_prune(
        crowded_endpoints, "crowded"
    )
    assert contract.NUM_PREDICTED_MODES <= crowded_count < 18
    assert fit_anchors.minimum_pairwise_distance(crowded_fitted) >= model.PRUNE_DISTANCE_METRES


def test_map_chunk_centres_are_the_mean_dot_position_per_slot():
    map_rows = torch.zeros(6, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_POSITION] = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [30.0, 0.0], [30.0, 4.0], [30.0, 8.0]]
    )
    centres = model.map_chunk_centres(map_rows, torch.tensor([0, 0, 1, 3, 3, 3]), 2, 2)
    assert centres.shape == (2, 2, 2)
    assert torch.allclose(centres[0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(centres[0, 1], torch.tensor([10.0, 10.0]))
    assert torch.allclose(centres[1, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(centres[1, 1], torch.tensor([30.0, 4.0]))


def test_a_focused_decoder_reads_the_map_chunk_near_its_draft_more_than_a_far_one():
    torch.manual_seed(121)
    unit_anchors = model.unit_anchor_offsets_per_type() * 40.0
    decoder = model.ModeDecoder(unit_anchors).eval()
    with torch.no_grad():
        decoder.map_focus_scale.fill_(20.0)
    neighbour_count = 2
    tokens = torch.randn(1, 1 + neighbour_count + 2, model.HIDDEN_DIM)
    token_present = torch.ones(1, tokens.shape[1], dtype=torch.bool)
    chunk_centres = torch.tensor([[[40.0, 0.0], [-380.0, 0.0]]])
    near_token, far_token = 1 + neighbour_count, 2 + neighbour_count
    forward_mode = model.ANCHOR_DISTANCE_COUNT - 1
    vehicle = torch.zeros(1, dtype=torch.long)

    def endpoint(token_set):
        with torch.no_grad():
            round_outputs, _, _ = decoder(token_set, token_present, vehicle, chunk_centres)
        return round_outputs[-1][0][0, forward_mode, -1]

    base = endpoint(tokens)
    near_perturbed = tokens.clone()
    near_perturbed[0, near_token] += 3.0
    far_perturbed = tokens.clone()
    far_perturbed[0, far_token] += 3.0
    near_shift = float((endpoint(near_perturbed) - base).norm())
    far_shift = float((endpoint(far_perturbed) - base).norm())
    assert near_shift > 10.0 * far_shift
    with torch.no_grad():
        decoder.map_focus_scale.fill_(-20.0)
    unfocused_far_shift = float((endpoint(far_perturbed) - endpoint(tokens)).norm())
    assert unfocused_far_shift > far_shift
