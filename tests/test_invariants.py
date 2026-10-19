"""One test per correctness property of the frames, staging, loader, model,
loss, pruning, metrics and submission.
"""

import importlib.util
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import reference_implementations
import torch

import fit_anchors
import submit
import train
from womd import (
    baseline,
    contract,
    frame_ops,
    loader,
    loss,
    metrics,
    model,
    pipeline,
    pruning,
    store,
)
from womd.frame_ops import Frame
from womd.loader import MapArrays, SceneArrays, SceneBatch, TargetArrays
from womd.model import ModePredictions
from womd.pipeline import BatchSettings
from womd_protos import scenario_pb2

STAGED_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "staged"
RUNNER_PATH = Path(__file__).parent.parent / "container" / "runner.py"
TWO_STAGED_SCENARIOS = BatchSettings(
    batch_size=2,
    worker_count=0,
    prefetch_batches=0,
    seed=0,
    designated_only=True,
)


class SyntheticSize(NamedTuple):
    """How large a synthetic scene batch is."""

    samples: int
    agents: int
    polylines: int
    dots_per_polyline: int


def lane_polyline_rows(first_dot_x, dot_count):
    """Map rows for a straight lane along x with one dot per metre."""
    rows = np.zeros((dot_count, contract.MAP_FEATURE_DIM), dtype=np.float32)
    along_x = first_dot_x + np.arange(dot_count, dtype=np.float64)
    rows[:, contract.MAP_POSITION] = np.stack(
        [along_x, np.zeros(dot_count)], axis=1
    )
    rows[:, contract.MAP_DIRECTION] = np.array([1.0, 0.0])
    lane_column = contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index(
        "lane"
    )
    rows[:, lane_column] = 1.0
    return rows


def synthetic_scene_batch(size):
    """A random scene batch of the given size, one object type per sample in
    turn.
    """
    history = contract.HISTORY_STEPS
    signals = contract.NUM_TRAFFIC_SIGNAL_STATES
    agent_history = torch.randn(
        size.samples, history, contract.AGENT_FEATURE_DIM
    )
    agent_history[:, :, contract.AGENT_TYPE] = 0.0
    for sample_index in range(size.samples):
        type_column = (
            contract.AGENT_TYPE.start + sample_index % contract.NUM_OBJECT_TYPES
        )
        agent_history[sample_index, :, type_column] = 1.0
    dot_count = size.polylines * size.dots_per_polyline
    map_rows = torch.randn(dot_count, contract.MAP_FEATURE_DIM)
    codes = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (dot_count, 2)
    )
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING :] = codes.float()
    token_count = size.agents + size.polylines
    return SceneBatch(
        scene=SceneArrays(
            agent_history=torch.randn(
                size.samples, size.agents, history, contract.AGENT_FEATURE_DIM
            ),
            agent_history_mask=torch.ones(
                size.samples, size.agents, history, dtype=torch.bool
            ),
            agent_signal_history=torch.rand(
                size.samples, size.agents, history, signals
            ),
        ),
        map=MapArrays(
            rows=map_rows,
            dot_chunk_slot=torch.arange(dot_count) // size.dots_per_polyline,
            chunk_signal_history=torch.rand(
                size.samples, size.polylines, history, signals
            ),
        ),
        targets=TargetArrays(
            scene_index=torch.arange(size.samples),
            agent_history=agent_history,
            agent_history_mask=torch.ones(
                size.samples, history, dtype=torch.bool
            ),
            agent_signal_history=torch.rand(size.samples, history, signals),
            token_visible=torch.ones(
                size.samples, token_count, dtype=torch.bool
            ),
            token_pose=torch.randn(size.samples, token_count, 4),
            future_positions=torch.randn(
                size.samples, contract.FUTURE_STEPS, 2
            ),
            future_mask=torch.ones(
                size.samples, contract.FUTURE_STEPS, dtype=torch.bool
            ),
        ),
    )


def future_targets(future_positions, future_mask):
    """A TargetArrays holding only a future, for the loss and monitor tests."""
    blanks = [None] * (len(TargetArrays._fields) - 2)
    return TargetArrays(*blanks, future_positions, future_mask)


def test_agent_frame_transform_inverts():
    """Into an agent's frame and back out returns the same world positions."""
    random_generator = np.random.default_rng(3)
    world_positions = random_generator.uniform(-120.0, 120.0, size=(64, 2))
    frame = Frame(np.array([13.5, -42.25]), 0.9137)

    local = frame_ops.positions_to_frame(world_positions, frame)
    recovered = frame_ops.positions_from_frame(local, frame)
    assert np.allclose(recovered, world_positions, atol=1e-9)


def test_agent_frame_x_axis_points_ahead():
    """A point straight ahead of the agent lands on its forward axis."""
    origin = np.array([13.5, -42.25])
    heading = 0.9137
    frame = Frame(origin, heading)
    forward = np.array([np.cos(heading), np.sin(heading)])
    leftward = np.array([-np.sin(heading), np.cos(heading)])
    distance = 17.0

    straight_ahead = frame_ops.positions_to_frame(
        origin + distance * forward, frame
    )
    off_to_the_left = frame_ops.positions_to_frame(
        origin + distance * leftward, frame
    )

    assert np.allclose(straight_ahead, np.array([distance, 0.0]), atol=1e-9)
    assert np.allclose(off_to_the_left, np.array([0.0, distance]), atol=1e-9)

    transposed_convention = (distance * forward) @ frame_ops.rotation_matrix(
        heading
    ).T
    assert not np.allclose(transposed_convention, straight_ahead, atol=1e-9)


def test_heading_wrap_stays_in_range():
    """Wrapped angles land in the half-open range from -pi to pi."""
    angles = np.array([-7.0, -np.pi, 0.0, np.pi, 7.0, 100.0])
    wrapped = frame_ops.wrap_to_pi(angles)
    assert (wrapped >= -np.pi).all() and (wrapped < np.pi).all()


def turning_track():
    """A vehicle track that drives an arc, every step valid."""
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
    return track


def test_frame_transforms_compose():
    """World to scene to agent must equal world to agent directly."""
    track = turning_track()
    sdc_frame = Frame(np.array([-12.0, 31.0]), -1.2)
    stored_rows = store.track_rows(track, sdc_frame, is_sdc=False)
    track_rows = stored_rows.astype(np.float32)

    frame = loader.agent_frame(track_rows)
    two_step = loader.rows_to_frame(track_rows, frame)

    world_positions = np.array(
        [[state.center_x, state.center_y] for state in track.states]
    )
    world_headings = np.array([state.heading for state in track.states])
    world_velocities = np.array(
        [[state.velocity_x, state.velocity_y] for state in track.states]
    )
    now_state = track.states[contract.CURRENT_STEP_INDEX]
    now_frame = Frame(
        np.array([now_state.center_x, now_state.center_y]), now_state.heading
    )

    direct_positions = frame_ops.positions_to_frame(world_positions, now_frame)
    direct_headings = world_headings - now_state.heading
    direct_velocities = frame_ops.directions_to_frame(
        world_velocities, now_state.heading
    )

    assert np.allclose(
        two_step[:, contract.AGENT_POSITION], direct_positions, atol=1e-3
    )
    assert np.allclose(
        two_step[:, contract.AGENT_HEADING_COSINE],
        np.cos(direct_headings),
        atol=1e-3,
    )
    assert np.allclose(
        two_step[:, contract.AGENT_HEADING_SINE],
        np.sin(direct_headings),
        atol=1e-3,
    )
    assert np.allclose(
        two_step[:, contract.AGENT_VELOCITY], direct_velocities, atol=1e-3
    )


def test_chunk_pooling_isolates_groups():
    """Empty chunk slots must come back marked absent."""
    torch.manual_seed(5)
    dot_embeddings = torch.randn(9, 8)
    dot_polyline_slot = torch.tensor([0, 0, 0, 1, 1, 1, 3, 3, 4])

    pooled = model.pool_chunks(dot_embeddings, dot_polyline_slot, (2, 3))
    assert pooled.tokens.shape == (2, 3, 8)
    assert pooled.present.tolist() == [[True, True, False], [True, True, False]]
    assert torch.all(pooled.tokens[:, 2] == 0.0)

    poisoned = dot_embeddings.clone()
    poisoned[:3] = 999.0
    poisoned_pooled = model.pool_chunks(poisoned, dot_polyline_slot, (2, 3))
    assert torch.equal(poisoned_pooled.tokens[0, 1:], pooled.tokens[0, 1:])
    assert torch.equal(poisoned_pooled.tokens[1], pooled.tokens[1])
    assert not torch.equal(poisoned_pooled.tokens[0, 0], pooled.tokens[0, 0])


def test_null_baselines_reproduce_their_motion():
    """The constant-turn baseline retraces an arc exactly, the constant-velocity
    one leaves it, and an invalid agent gets zeros.
    """
    step_offsets = np.arange(
        -contract.CURRENT_STEP_INDEX, contract.FUTURE_STEPS + 1
    )
    elapsed = step_offsets * contract.TIMESTEP_SECONDS
    speeds = np.array([8.0, 0.0])
    yaw_rates = np.array([0.3, 0.0])

    turned = yaw_rates[:, None] * elapsed[None, :]
    safe_yaw_rates = np.where(yaw_rates == 0.0, 1.0, yaw_rates)
    radii = np.where(yaw_rates == 0.0, 0.0, speeds / safe_yaw_rates)
    arc_x = radii[:, None] * np.sin(turned)
    arc_y = radii[:, None] * (1.0 - np.cos(turned))
    arc_positions = np.stack([arc_x, arc_y], axis=-1)

    agent_track = np.zeros(
        (2, len(step_offsets), contract.AGENT_FEATURE_DIM), dtype=np.float32
    )
    agent_track[..., contract.AGENT_POSITION] = arc_positions
    agent_track[..., contract.AGENT_HEADING_COSINE] = np.cos(turned)
    agent_track[..., contract.AGENT_HEADING_SINE] = np.sin(turned)
    heading_directions = np.stack([np.cos(turned), np.sin(turned)], axis=-1)
    agent_track[..., contract.AGENT_VELOCITY] = (
        speeds[:, None, None] * heading_directions
    )

    agent_history = torch.from_numpy(agent_track[:, : contract.HISTORY_STEPS])
    agent_history_mask = torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool)
    logged_future = torch.from_numpy(
        arc_positions[:, contract.HISTORY_STEPS :].astype(np.float32)
    )

    turning_trajectories, turning_logits = baseline.constant_turn_rate(
        agent_history, agent_history_mask
    )
    straight_trajectories, _ = baseline.constant_velocity(agent_history)
    assert turning_trajectories.shape == (2, 1, contract.FUTURE_STEPS, 2)
    assert turning_logits.shape == (2, 1)

    assert torch.allclose(turning_trajectories[:, 0], logged_future, atol=1e-3)
    assert (straight_trajectories[0, 0, -1] - logged_future[0, -1]).norm() > 1.0
    assert torch.all(turning_trajectories[1] == 0.0)
    assert torch.all(straight_trajectories[1] == 0.0)

    pruned = pruning.prune_modes(turning_trajectories, turning_logits)
    expanded = turning_trajectories.expand(
        -1, contract.NUM_PREDICTED_MODES, -1, -1
    )
    assert torch.equal(pruned.trajectories, expanded)


def test_anchor_null_ends_on_the_most_used_anchors():
    """Anchors are in metres, so the agent's speed must not change where the
    null's paths end.
    """
    reach = torch.tensor([40.0, 24.0, 12.0])[:, None, None]
    unit_anchors = (
        reference_implementations.unit_anchor_offsets_per_type() * reach
    )
    driven_types = torch.tensor([0, contract.NUM_OBJECT_TYPES - 1])
    current_speeds = torch.tensor([8.0, 0.0])
    agent_history = torch.zeros(
        2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM
    )
    velocities = torch.stack([current_speeds, torch.zeros(2)], dim=-1)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = (
        velocities
    )
    type_columns = contract.AGENT_TYPE.start + driven_types
    agent_history[
        torch.arange(2), contract.CURRENT_STEP_INDEX, type_columns
    ] = 1.0

    trajectories, confidence_logits = baseline.anchor_lines(
        agent_history, unit_anchors
    )

    assert trajectories.shape == (
        2,
        contract.NUM_PREDICTED_MODES,
        contract.FUTURE_STEPS,
        2,
    )
    assert confidence_logits.shape == (2, contract.NUM_PREDICTED_MODES)
    assert torch.allclose(
        trajectories[:, :, -1],
        unit_anchors[driven_types][:, : contract.NUM_PREDICTED_MODES],
        atol=1e-4,
    )

    steps = torch.cat(
        [trajectories[..., :1, :], trajectories.diff(dim=-2)], dim=-2
    )
    assert torch.allclose(steps, steps[..., :1, :].expand_as(steps), atol=1e-5)


def test_anchor_fitting_recovers_separated_clusters():
    """No anchor may be left without endpoints."""
    random_generator = np.random.default_rng(17)
    jitter = random_generator.uniform(-0.08, 0.08, (model.QUERY_COUNT, 2))
    cluster_centres = (
        reference_implementations.unit_anchor_offsets() * 10.0
        + torch.tensor(jitter, dtype=torch.float32)
    )
    spread = random_generator.normal(0.0, 0.005, (model.QUERY_COUNT, 50, 2))
    endpoints = cluster_centres[:, None, :] + torch.tensor(
        spread, dtype=torch.float32
    )
    endpoints = endpoints.reshape(-1, 2)

    fit = fit_anchors.fit_type("vehicle", endpoints)
    fitted = fit.anchors

    assert fit.stopped_by_convergence
    assert fitted.shape == (model.QUERY_COUNT, 2)
    nearest_cluster = torch.cdist(fitted, cluster_centres).argmin(dim=1)
    assert sorted(nearest_cluster.tolist()) == list(range(model.QUERY_COUNT))
    assert torch.allclose(fitted, cluster_centres[nearest_cluster], atol=0.01)
    assert int(fit.counts.min()) > 0


def test_batched_pruning_matches_the_loop():
    """The plain single-sample loop is the reference."""
    torch.manual_seed(7)
    trajectories = pruning.PRUNE_DISTANCE_METRES * torch.randn(
        2, model.QUERY_COUNT, contract.FUTURE_STEPS, 2
    )
    confidence_logits = torch.randn(2, model.QUERY_COUNT)
    trajectories[1] = trajectories[1, :1]

    pruned = pruning.prune_modes(trajectories, confidence_logits)
    kept_counts = pruned.kept_count
    assert kept_counts[0] > 1 and kept_counts[0] <= contract.NUM_PREDICTED_MODES
    assert kept_counts[1] == 1
    for sample_index in range(len(kept_counts)):
        walked = reference_implementations.prune_modes_single_sample(
            trajectories[sample_index], confidence_logits[sample_index]
        )
        walked_trajectories, walked_logits = walked
        assert torch.equal(
            walked_trajectories, pruned.trajectories[sample_index]
        )
        assert torch.equal(
            walked_logits, pruned.confidence_logits[sample_index]
        )


def test_opposite_endpoints_get_opposite_anchors():
    """A future ending ahead and one ending behind are assigned anchors on those
    sides.
    """
    unit_anchors = reference_implementations.unit_anchor_offsets() * 40.0
    future_positions = torch.zeros(2, contract.FUTURE_STEPS, 2)
    future_positions[0, :, 0] = torch.linspace(0.5, 40.0, contract.FUTURE_STEPS)
    future_positions[1, :, 0] = torch.linspace(
        -0.5, -40.0, contract.FUTURE_STEPS
    )
    future_mask = torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool)

    targets = future_targets(future_positions, future_mask)
    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(2, -1, -1), targets
    )

    assert assigned_mode[0] != assigned_mode[1]
    assert unit_anchors[assigned_mode[0], 0] > 0
    assert unit_anchors[assigned_mode[1], 0] < 0
    assert (
        unit_anchors[assigned_mode[0]] - unit_anchors[assigned_mode[1]]
    ).norm() > 20.0


def test_assignment_ignores_padded_steps():
    """Assignment scores the whole valid path, never the padded tail."""
    unit_anchors = reference_implementations.unit_anchor_offsets() * 40.0
    anchors = unit_anchors.expand(1, -1, -1)
    future_positions = torch.zeros(1, contract.FUTURE_STEPS, 2)
    future_positions[0, :10, 0] = 40.0
    future_mask = torch.zeros(1, contract.FUTURE_STEPS, dtype=torch.bool)
    future_mask[0, :10] = True

    assigned_mode = loss.anchor_assigned_mode(
        anchors, future_targets(future_positions, future_mask)
    )
    all_valid = torch.ones_like(future_mask)
    padded_tail_mode = loss.anchor_assigned_mode(
        anchors, future_targets(future_positions, all_valid)
    )

    assert torch.allclose(
        unit_anchors[assigned_mode[0]], torch.tensor([40.0, 0.0]), atol=1e-4
    )
    assert padded_tail_mode[0] != assigned_mode[0]
    assert float(unit_anchors[padded_tail_mode[0]].norm()) == pytest.approx(
        40.0 / reference_implementations.ANCHOR_DISTANCE_COUNT
    )

    waits_then_arrives = torch.zeros(1, contract.FUTURE_STEPS, 2)
    waits_then_arrives[0, -1, 0] = 40.0
    late_mode = loss.anchor_assigned_mode(
        anchors, future_targets(waits_then_arrives, all_valid)
    )
    assert float(unit_anchors[late_mode[0]].norm()) < 40.0


def test_submission_returns_to_the_world_frame():
    """The logged future, sent through the submission transform, must land where
    it was logged.
    """
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:1]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")

    scenario = loader.read_scenario(scenario_paths[0])
    track_index = int(loader.eligible_tracks(scenario, designated_only=True)[0])
    scene_sample = loader.scene_sample(scenario, [track_index])
    sample = scene_sample.targets[0]
    agent_frame_future = sample.future_positions

    world_future = submit.to_world(
        agent_frame_future, sample.frame, scenario.scene_frame
    )
    world_coordinate_resolution = (
        np.finfo(np.float32).eps * np.abs(world_future).max()
    )

    back_to_scene_frame = frame_ops.positions_to_frame(
        world_future, scenario.scene_frame
    )
    back_to_agent_frame = frame_ops.positions_to_frame(
        back_to_scene_frame, sample.frame
    )
    assert np.allclose(
        back_to_agent_frame,
        agent_frame_future,
        atol=world_coordinate_resolution,
    )

    future_steps = slice(contract.CURRENT_STEP_INDEX + 1, None)
    logged_scene_future = scenario.track_rows[
        track_index, future_steps, contract.AGENT_POSITION
    ]
    logged_world_future = frame_ops.positions_from_frame(
        logged_scene_future, scenario.scene_frame
    )
    logged = scenario.track_valid[track_index, future_steps]
    assert logged.any()
    assert np.allclose(
        world_future[logged],
        logged_world_future[logged],
        atol=world_coordinate_resolution,
    )


def test_constant_velocity_null_keeps_the_logged_speed():
    """A straight-moving agent is carried at exactly its logged velocity."""
    logged_velocity = torch.tensor([[11.0, -4.0], [0.0, 0.0]])
    start = torch.tensor([3.0, 5.0])
    agent_history = torch.zeros(
        2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM
    )
    agent_history[:, :, contract.AGENT_HEADING_COSINE] = 1.0
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = (
        logged_velocity
    )
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_POSITION] = (
        start
    )

    trajectories, _ = baseline.constant_velocity(agent_history)
    step_numbers = torch.arange(
        1, contract.FUTURE_STEPS + 1, dtype=torch.float32
    )
    elapsed_seconds = contract.TIMESTEP_SECONDS * step_numbers
    straight_line = start + logged_velocity[0] * elapsed_seconds[:, None]

    assert torch.allclose(trajectories[0, 0], straight_line, atol=1e-4)
    step_length = float(logged_velocity[0].norm() * contract.TIMESTEP_SECONDS)
    assert torch.allclose(
        trajectories[0, 0].diff(dim=0).norm(dim=-1),
        torch.full((contract.FUTURE_STEPS - 1,), step_length),
        atol=1e-4,
    )
    assert torch.allclose(
        trajectories[1, 0], start.expand(contract.FUTURE_STEPS, 2), atol=1e-4
    )


def test_artifact_provenance_round_trips():
    """A missing stamp, or one from another code version, is refused."""
    stamp = contract.artifact_provenance(
        "test_invariants.py", "unit-test-source"
    )
    checked = contract.check_provenance(stamp, "in-memory", "Regenerate.")
    assert checked["code_version"] == contract.STAGING_CODE_VERSION
    assert checked["producer"] == "test_invariants.py"

    with pytest.raises(ValueError):
        contract.check_provenance(None, "in-memory", "Regenerate.")

    stale = json.dumps(
        {"code_version": "not-the-working-tree", "producer": "x", "source": "y"}
    )
    with pytest.raises(ValueError):
        contract.check_provenance(stale, "in-memory", "Regenerate.")


def write_scorer_scenario(path, scenario_id, agent_count):
    """A staged file of agent_count agents at x = 0, 1, 2, ... with no map."""
    track_rows = np.zeros(
        (agent_count, contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM),
        np.float32,
    )
    track_rows[:, :, contract.AGENT_HEADING_COSINE] = 1.0
    track_rows[:, :, contract.AGENT_TYPE.start] = 1.0
    for track_index in range(agent_count):
        track_rows[track_index, :, contract.AGENT_POSITION.start] = float(
            track_index
        )
    signal_shape = (
        0,
        contract.HISTORY_STEPS,
        contract.NUM_TRAFFIC_SIGNAL_STATES,
    )
    np.savez(
        path,
        track_rows=track_rows,
        track_valid=np.ones((agent_count, contract.TOTAL_STEPS), dtype=bool),
        track_ids=np.arange(10, 10 * (agent_count + 1), 10, dtype=np.int64),
        is_designated_target=np.ones(agent_count, dtype=bool),
        is_object_of_interest=np.zeros(agent_count, dtype=bool),
        map_rows=np.zeros((0, contract.MAP_FEATURE_DIM), dtype=np.float32),
        feature_lengths=np.zeros(0, dtype=np.int64),
        feature_ids=np.zeros(0, dtype=np.int64),
        feature_is_interpolating=np.zeros(0, dtype=bool),
        polyline_signal_histories=np.zeros(signal_shape, dtype=np.float32),
        frame_origin=np.zeros(2, dtype=np.float32),
        frame_heading=np.float32(0.0),
        scenario_id=scenario_id,
    )


def test_scorer_tensors_group_by_scenario(tmp_path):
    """Ground truth holds every agent, not only the predicted ones."""
    runner_spec = importlib.util.spec_from_file_location(
        "container_runner", RUNNER_PATH
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)

    scenario_agent_counts = {"scn_a": 3, "scn_b": 2}
    scenario_target_track_ids = {"scn_a": [20, 30], "scn_b": [10]}
    for scenario_id, agent_count in scenario_agent_counts.items():
        write_scorer_scenario(
            tmp_path / f"{scenario_id}.npz", scenario_id, agent_count
        )

    prediction_scenario_ids = []
    prediction_track_ids = []
    for scenario_id, target_track_ids in scenario_target_track_ids.items():
        for track_id in target_track_ids:
            prediction_scenario_ids.append(scenario_id)
            prediction_track_ids.append(track_id)
    target_count = len(prediction_track_ids)
    trajectory_shape = (
        target_count,
        contract.NUM_PREDICTED_MODES,
        contract.SUBMISSION_STEPS,
        2,
    )
    predictions = contract.Submission(
        scenario_id=np.array(prediction_scenario_ids),
        track_id=np.array(prediction_track_ids, dtype=np.int64),
        world_trajectories=np.random.default_rng(0).normal(
            size=trajectory_shape
        ),
        confidences=np.full(
            (target_count, contract.NUM_PREDICTED_MODES), 1.0 / 6.0
        ),
        provenance=contract.artifact_provenance("test", "scorer-tensors"),
    )

    tensors = runner.metric_inputs(predictions, tmp_path)

    assert tensors.ground_truth_trajectory.shape == (
        2,
        3,
        contract.TOTAL_STEPS,
        7,
    )
    assert tensors.prediction_trajectory.shape == (
        2,
        2,
        contract.NUM_PREDICTED_MODES,
        1,
        contract.SUBMISSION_STEPS,
        2,
    )
    assert (
        int(tensors.prediction_ground_truth_indices_mask.sum()) == target_count
    )

    scenario_row = {
        scenario_id: row for row, scenario_id in enumerate(tensors.scenario_id)
    }
    a_row, b_row = scenario_row["scn_a"], scenario_row["scn_b"]
    assert tensors.ground_truth_is_valid[a_row].all()
    assert tensors.ground_truth_is_valid[b_row, :2].all()
    assert not tensors.ground_truth_is_valid[b_row, 2:].any()
    assert not tensors.prediction_ground_truth_indices_mask[b_row, 1:].any()

    for slot_index, expected_track_id in enumerate(
        scenario_target_track_ids["scn_a"]
    ):
        agent_row = int(
            tensors.prediction_ground_truth_indices[a_row, slot_index, 0]
        )
        assert tensors.object_id[a_row, agent_row] == expected_track_id
        first_x = tensors.ground_truth_trajectory[a_row, agent_row, 0, 0]
        assert first_x == pytest.approx(agent_row * 1.0)


def test_input_order_does_not_change_predictions():
    """Scene agents and map dots are shuffled; trajectories must not move."""
    torch.manual_seed(11)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    batch = synthetic_scene_batch(
        SyntheticSize(samples=2, agents=6, polylines=5, dots_per_polyline=16)
    )
    batch.scene.agent_history_mask[:, 5] = False
    scene_agent_order = torch.randperm(6)
    token_order = torch.cat([scene_agent_order, torch.arange(6, 11)])
    map_order = torch.randperm(batch.map.rows.shape[0])
    shuffled_scene = {}
    for name, array in batch.scene._asdict().items():
        shuffled_scene[name] = array[:, scene_agent_order]
    permuted = SceneBatch(
        scene=SceneArrays(**shuffled_scene),
        map=batch.map._replace(
            rows=batch.map.rows[map_order],
            dot_chunk_slot=batch.map.dot_chunk_slot[map_order],
        ),
        targets=batch.targets._replace(
            token_visible=batch.targets.token_visible[:, token_order],
            token_pose=batch.targets.token_pose[:, token_order],
        ),
    )
    with torch.no_grad():
        base = predictor(batch)
        permuted_output = predictor(permuted)
    base_trajectories, _, base_logits, _ = base
    permuted_trajectories, _, permuted_logits, _ = permuted_output
    assert torch.allclose(base_trajectories, permuted_trajectories, atol=1e-4)
    assert torch.allclose(base_logits, permuted_logits, atol=1e-5)


def two_lane_signal_scenario(track_count, signalled_history):
    """A staged scenario with a signalled lane under the first track and an
    unsignalled lane 40 m away.
    """
    signalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane[:, contract.MAP_POSITION.start + 1] = 40.0
    signal_shape = (
        2,
        contract.HISTORY_STEPS,
        contract.NUM_TRAFFIC_SIGNAL_STATES,
    )
    polyline_signal_histories = np.zeros(signal_shape, dtype=np.float32)
    polyline_signal_histories[0] = signalled_history
    track_shape = (
        track_count,
        contract.TOTAL_STEPS,
        contract.AGENT_FEATURE_DIM,
    )
    track_rows = np.zeros(track_shape, dtype=np.float32)
    track_valid = np.ones((track_count, contract.TOTAL_STEPS), dtype=bool)
    track_rows[:, :, contract.AGENT_HEADING_COSINE] = 1.0
    track_rows[0, :, contract.AGENT_POSITION] = np.array([5.0, 0.0])
    track_rows[1, :, contract.AGENT_POSITION] = np.array([5.0, 40.0])
    if track_count > 2:
        track_rows[2, :, contract.AGENT_POSITION] = np.array([7.0, 0.0])
        track_valid[2, contract.CURRENT_STEP_INDEX] = False
    staged = contract.StagedFile(
        track_rows=track_rows,
        track_valid=track_valid,
        track_ids=np.arange(track_count),
        is_designated_target=np.ones(track_count, dtype=bool),
        is_object_of_interest=np.zeros(track_count, dtype=bool),
        map_rows=np.concatenate([signalled_lane, unsignalled_lane]),
        feature_lengths=np.array([11, 11], dtype=np.int64),
        feature_ids=np.array([101, 102], dtype=np.int64),
        feature_is_interpolating=np.zeros(2, dtype=bool),
        polyline_signal_histories=polyline_signal_histories,
        frame_origin=np.zeros(2, dtype=np.float32),
        frame_heading=np.float32(0.0),
        scenario_id=np.array("two-lane"),
    )
    return loader.staged_scenario(staged)


def test_agents_carry_their_lane_signal_history():
    """An agent on a signalled lane receives that lane's signal history, and the
    encoder's output depends on it.
    """
    signal_shape = (contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)
    signalled_history = np.zeros(signal_shape, dtype=np.float32)
    stop_column = contract.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_STOP")
    go_column = contract.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_GO")
    signalled_history[:8, stop_column] = 1.0
    signalled_history[8:, go_column] = 1.0
    three_agent_scene = loader.scene_sample(
        two_lane_signal_scenario(3, signalled_history), [0]
    )
    two_agent_scene = loader.scene_sample(
        two_lane_signal_scenario(2, signalled_history), [0]
    )
    assert np.array_equal(
        three_agent_scene.targets[0].agent_signal_history, signalled_history
    )
    assert np.all(three_agent_scene.scene_agent_signal_history[1:] == 0.0)
    batch = loader.scene_batch([three_agent_scene, two_agent_scene])
    assert batch.scene.agent_signal_history.shape == (2, 3) + signal_shape
    assert np.all(batch.scene.agent_signal_history[1, 2] == 0.0)
    tensor_batch = pipeline.torch_batch(batch)
    silenced_targets = tensor_batch.targets._replace(
        agent_signal_history=torch.zeros_like(
            tensor_batch.targets.agent_signal_history
        )
    )
    silenced_batch = tensor_batch._replace(targets=silenced_targets)
    torch.manual_seed(47)
    encoder = model.SceneEncoder()
    with torch.no_grad():
        memory = encoder(tensor_batch)
        silenced_memory = encoder(silenced_batch)
    assert not torch.allclose(memory.tokens[0, 0], silenced_memory.tokens[0, 0])


def test_target_prediction_ignores_batch_company():
    """A target predicted alone must equal the same target batched beside other
    targets and another scene.
    """
    signal_shape = (contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)
    signalled_history = np.zeros(signal_shape, dtype=np.float32)
    three_track_scenario = two_lane_signal_scenario(3, signalled_history)
    two_track_scenario = two_lane_signal_scenario(2, signalled_history)

    torch.manual_seed(53)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    alone_sample = loader.scene_sample(three_track_scenario, [1])
    together_samples = [
        loader.scene_sample(two_track_scenario, [0]),
        loader.scene_sample(three_track_scenario, [0, 1, 2]),
    ]
    with torch.no_grad():
        alone_batch = pipeline.torch_batch(loader.scene_batch([alone_sample]))
        together_batch = pipeline.torch_batch(
            loader.scene_batch(together_samples)
        )
        alone = predictor(alone_batch).trajectories
        together = predictor(together_batch).trajectories
    assert torch.allclose(alone[0], together[2], atol=1e-4)


def test_zero_head_output_lands_on_the_anchor():
    """With a zero head a mode ends at its anchor, and the loss assigns by that
    same anchor.
    """
    torch.manual_seed(37)
    reach = torch.tensor([40.0, 24.0, 12.0])[:, None, None]
    unit_anchors = (
        reference_implementations.unit_anchor_offsets_per_type() * reach
    )
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.zero_()
        predictor.mode_decoder.trajectory_head[-1].bias.zero_()
    size = SyntheticSize(
        samples=contract.NUM_OBJECT_TYPES,
        agents=2,
        polylines=2,
        dots_per_polyline=10,
    )
    batch = synthetic_scene_batch(size)
    with torch.no_grad():
        predicted = predictor(batch)
    assert torch.allclose(
        predicted.trajectories[:, :, -1], unit_anchors, atol=1e-4
    )
    assert torch.allclose(predicted.anchors, unit_anchors)
    ramp = torch.arange(1, contract.FUTURE_STEPS + 1) / contract.FUTURE_STEPS
    logged_futures = unit_anchors[0][:, None, :] * ramp[None, :, None]
    all_valid = torch.ones(
        model.QUERY_COUNT, contract.FUTURE_STEPS, dtype=torch.bool
    )
    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors[0].expand(model.QUERY_COUNT, -1, -1),
        future_targets(logged_futures, all_valid),
    )
    assert torch.equal(assigned_mode, torch.arange(model.QUERY_COUNT))


def test_regression_term_is_gaussian_nll():
    """Checked against the closed form at the uncertainty the model states."""
    torch.manual_seed(53)
    sample_count = 4
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    future_mask = torch.ones(
        sample_count, contract.FUTURE_STEPS, dtype=torch.bool
    )
    targets = future_targets(future_positions, future_mask)
    confidence_logits = torch.zeros(sample_count, model.QUERY_COUNT)
    unit_anchors = reference_implementations.unit_anchor_offsets().expand(
        sample_count, -1, -1
    )
    displacement = torch.tensor([0.3, -0.4])

    def regression_at(log_standard_deviation, error):
        """The regression loss when every mode misses by error at the given log
        standard deviation.
        """
        trajectories = (
            future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
            + error
        )
        log_sigma = torch.full_like(trajectories, log_standard_deviation)
        predicted = ModePredictions(
            trajectories, log_sigma, confidence_logits, unit_anchors
        )
        return float(loss.loss_terms(predicted, targets).regression)

    floor = model.MINIMUM_LOG_STANDARD_DEVIATION
    squared_miss = float((displacement**2).sum())
    exact_form = (
        2.0 * floor
        + 2.0 * loss.HALF_LOG_TWO_PI
        + 0.5 * squared_miss / math.exp(floor) ** 2
    )
    assert regression_at(floor, displacement) == pytest.approx(
        exact_form, rel=1e-5
    )
    for log_standard_deviation in (
        floor,
        0.0,
        model.MAXIMUM_LOG_STANDARD_DEVIATION,
    ):
        exact_hit = regression_at(log_standard_deviation, torch.zeros(2))
        assert exact_hit < regression_at(log_standard_deviation, displacement)


def test_classification_term_is_cross_entropy():
    """Taken over anchors, against the assigned mode."""
    sample_count = 3
    future_positions = torch.zeros(sample_count, contract.FUTURE_STEPS, 2)
    future_positions[:, -1, 0] = 1.0
    future_mask = torch.ones(
        sample_count, contract.FUTURE_STEPS, dtype=torch.bool
    )
    trajectories = future_positions.unsqueeze(1).expand(
        -1, model.QUERY_COUNT, -1, -1
    )
    unit_anchors = torch.zeros(sample_count, model.QUERY_COUNT, 2)
    unit_anchors[:, 0, 0] = 1.0
    predicted = ModePredictions(
        trajectories,
        torch.zeros_like(trajectories),
        torch.zeros(sample_count, model.QUERY_COUNT),
        unit_anchors,
    )
    terms = loss.loss_terms(
        predicted, future_targets(future_positions, future_mask)
    )
    assert float(terms.classification) == pytest.approx(
        math.log(model.QUERY_COUNT), rel=1e-5
    )


def test_masked_steps_move_no_loss():
    """Values at masked future steps cannot change any loss component."""
    torch.manual_seed(103)
    sample_count = 4
    future_mask = torch.ones(
        sample_count, contract.FUTURE_STEPS, dtype=torch.bool
    )
    future_mask[1, contract.FUTURE_STEPS // 2 :] = False
    future_mask[2] = False
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    trajectories = torch.randn(
        sample_count, model.QUERY_COUNT, contract.FUTURE_STEPS, 2
    )
    log_standard_deviation = torch.randn_like(trajectories)
    confidence_logits = torch.randn(sample_count, model.QUERY_COUNT)
    unit_anchors = (
        reference_implementations.unit_anchor_offsets().expand(
            sample_count, -1, -1
        )
        * 40.0
    )

    def components(logged_positions, predicted_positions):
        predicted = ModePredictions(
            predicted_positions,
            log_standard_deviation,
            confidence_logits,
            unit_anchors,
        )
        return torch.stack(
            loss.loss_terms(
                predicted, future_targets(logged_positions, future_mask)
            )
        )

    masked_steps = ~future_mask
    poisoned_positions = future_positions.clone()
    poisoned_positions[masked_steps] = 1e6
    poisoned_trajectories = trajectories.clone()
    poisoned_trajectories[
        masked_steps.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1)
    ] = 1e6
    assert torch.equal(
        components(future_positions, trajectories),
        components(poisoned_positions, poisoned_trajectories),
    )


def test_every_model_output_moves_the_loss():
    """An output the loss ignored could drift freely, so each one is perturbed
    and the loss must change.
    """
    torch.manual_seed(83)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    size = SyntheticSize(
        samples=contract.NUM_OBJECT_TYPES,
        agents=3,
        polylines=2,
        dots_per_polyline=10,
    )
    batch = synthetic_scene_batch(size)
    with torch.no_grad():
        emitted = predictor(batch)
    perturbations = {
        "trajectories": lambda value: value + 1.0,
        "log_standard_deviation": lambda value: value + 1.0,
        "confidence_logits": lambda value: torch.cat(
            [value[..., :1] + 1.0, value[..., 1:]], dim=-1
        ),
        "anchors": lambda value: -value,
    }
    assert set(perturbations) == set(ModePredictions._fields)

    def total(predicted):
        return loss.loss_terms(predicted, batch.targets).total

    unperturbed = total(emitted)
    for name, perturb in perturbations.items():
        perturbed = emitted._replace(**{name: perturb(getattr(emitted, name))})
        assert not torch.equal(total(perturbed), unperturbed), name


def test_no_step_uncertainty_starts_saturated():
    """Untrained modes must also start out distinct from one another."""
    torch.manual_seed(97)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    size = SyntheticSize(
        samples=contract.NUM_OBJECT_TYPES,
        agents=2,
        polylines=2,
        dots_per_polyline=10,
    )
    batch = synthetic_scene_batch(size)
    with torch.no_grad():
        predicted = predictor(batch)
    assert predicted.trajectories.isfinite().all()
    assert predicted.confidence_logits.isfinite().all()
    log_sigma = predicted.log_standard_deviation
    assert not (log_sigma == model.MINIMUM_LOG_STANDARD_DEVIATION).any()
    assert not (log_sigma == model.MAXIMUM_LOG_STANDARD_DEVIATION).any()
    endpoints = predicted.trajectories[:, :, -1]
    assert not torch.allclose(endpoints[:, :1], endpoints, atol=1e-4)


def test_anchors_are_frozen_buffers():
    """They take no gradient and are saved inside the checkpoint."""
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    )
    parameter_names = {name for name, _ in predictor.named_parameters()}
    assert "mode_decoder.unit_anchors" not in parameter_names
    assert "mode_decoder.unit_anchors" in predictor.state_dict()
    assert not predictor.unit_anchors.requires_grad


def test_training_step_runs_on_staged_scenarios(tmp_path):
    """One optimiser step on two real staged scenarios gives finite losses and
    gradients. Skipped when no staged data is present.
    """
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:2]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")
    torch.manual_seed(0)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    )
    optimizer = torch.optim.AdamW(
        train.parameter_groups(predictor), lr=train.LEARNING_RATE
    )
    batch = next(iter(pipeline.batches(scenario_paths, TWO_STAGED_SCENARIOS)))
    step = train.training_step(predictor, batch)
    assert torch.stack(list(step.losses)).isfinite().all()
    step.losses.total.backward()
    starved = [
        name
        for name, parameter in predictor.named_parameters()
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    ]
    assert not starved, starved
    optimizer.step()
    accumulator = metrics.MetricAccumulator()
    detached = ModePredictions(*(tensor.detach() for tensor in step.predicted))
    accumulator.update(detached, batch.targets)
    assert math.isfinite(accumulator.min_ade())
    assert math.isfinite(accumulator.min_fde())
    checkpoint_path = tmp_path / "predictor.pt"
    torch.save(predictor.state_dict(), checkpoint_path)
    reloaded_state = torch.load(checkpoint_path)
    assert all(
        torch.equal(reloaded_state[name], parameter)
        for name, parameter in predictor.state_dict().items()
    )


def test_pruning_conserves_probability_mass():
    """A dropped duplicate's share goes to the mode kept in its place."""
    torch.manual_seed(131)
    trajectories = torch.zeros(1, model.QUERY_COUNT, contract.FUTURE_STEPS, 2)
    trajectories[0, :, -1, 0] = 10.0 * torch.arange(
        model.QUERY_COUNT, dtype=torch.float32
    )
    trajectories[0, 1, -1, 0] = 0.5
    confidence_logits = torch.zeros(1, model.QUERY_COUNT)
    confidence_logits[0, 0] = 3.0
    confidence_logits[0, 1] = 2.0
    pruned = pruning.prune_modes(trajectories, confidence_logits)
    kept_trajectories = pruned.trajectories
    confidences = pruning.kept_confidences(
        trajectories, confidence_logits, kept_trajectories
    )
    probabilities = torch.softmax(confidence_logits, dim=-1)[0]
    assert float(confidences.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(kept_trajectories[0, 0, -1, 0]) == 0.0
    assert not (kept_trajectories[0, :, -1, 0] == 0.5).any()
    merged = float(probabilities[0] + probabilities[1])
    assert float(confidences[0, 0]) == pytest.approx(merged, rel=1e-6)
    assert float(confidences[0, 1:].sum()) == pytest.approx(
        1.0 - merged, rel=1e-5
    )


def test_step_head_walks_a_line_or_an_arc():
    """Constant steps walk a straight line; steps that rotate walk an arc."""
    torch.manual_seed(137)
    decoder = model.ModeDecoder(
        reference_implementations.unit_anchor_offsets_per_type() * 0.0
    ).eval()
    memory = model.Memory(
        torch.randn(1, 5, model.HIDDEN_DIM), torch.ones(1, 5, dtype=torch.bool)
    )
    type_index = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        bias = decoder.trajectory_head[-1].bias.view(
            2, contract.FUTURE_STEPS, 2
        )
        bias.zero_()
        bias[0, :, 0] = 0.5
        straight = decoder(memory, type_index).trajectories
        angles = 0.02 * torch.arange(contract.FUTURE_STEPS, dtype=torch.float32)
        bias[0, :, 0] = 0.5 * angles.cos()
        bias[0, :, 1] = 0.5 * angles.sin()
        arc = decoder(memory, type_index).trajectories
    step_numbers = torch.arange(
        1, contract.FUTURE_STEPS + 1, dtype=torch.float32
    )
    expected_straight = torch.stack(
        [0.5 * step_numbers, torch.zeros(contract.FUTURE_STEPS)], dim=-1
    )
    assert torch.allclose(straight[0, 0], expected_straight, atol=1e-5)
    steps = arc[0, 0].diff(dim=0)
    assert torch.allclose(
        steps.norm(dim=-1),
        torch.full((contract.FUTURE_STEPS - 1,), 0.5),
        atol=1e-5,
    )
    turning = torch.atan2(steps[:, 1], steps[:, 0]).diff()
    assert torch.allclose(turning, torch.full_like(turning, 0.02), atol=1e-5)
