"""Runs inside the container, feeding predictions and ground truth to Waymo's
metrics op and checking our protos against theirs.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, NamedTuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPOSITORY_ROOT))

import womd.runtime_env
import numpy as np

from womd import contract, frame_ops, loader

# The pip package holding Waymo's metrics; its installed version is
# printed with each run.
WAYMO_OPEN_DATASET_DISTRIBUTION_NAME = "waymo-open-dataset-tf-2-12-0"
# Two decoders may differ by this much on a decimal field and still
# count as agreeing.
FLOATING_POINT_DISAGREEMENT_TOLERANCE = 1e-4

# Waymo's tutorial metrics configuration, predictions at 2 Hz scored at
# 3, 5 and 8 seconds, at most 6 per agent.
WAYMO_TUTORIAL_MOTION_METRICS_CONFIG_TEXT = """
track_steps_per_second: 10
prediction_steps_per_second: 2
track_history_samples: 10
track_future_samples: 80
speed_lower_bound: 1.4
speed_upper_bound: 11.0
speed_scale_lower: 0.5
speed_scale_upper: 1.0
step_configurations {
  measurement_step: 5
  lateral_miss_threshold: 1.0
  longitudinal_miss_threshold: 2.0
}
step_configurations {
  measurement_step: 9
  lateral_miss_threshold: 1.8
  longitudinal_miss_threshold: 3.6
}
step_configurations {
  measurement_step: 15
  lateral_miss_threshold: 3.0
  longitudinal_miss_threshold: 6.0
}
max_predictions: 6
"""


def track_row_world_frame_state(track_row: np.ndarray, frame_origin: np.ndarray,
                                frame_heading: float) -> np.ndarray:
    """One agent's scene-frame rows to the 7 world-frame columns the metrics op
    wants, x, y, length, width, heading, vx and vy.
    """
    positions_world = frame_ops.positions_from_frame(
        track_row[:, contract.AGENT_POSITION],
        frame_origin,
        frame_heading,
    )
    # A velocity rotates with the frame but has no origin to add.
    velocities_world = frame_ops.positions_from_frame(
        track_row[:, contract.AGENT_VELOCITY],
        np.zeros(2),
        frame_heading,
    )
    heading_scene_frame = np.arctan2(
        track_row[:, contract.AGENT_HEADING_SINE],
        track_row[:, contract.AGENT_HEADING_COSINE],
    )
    heading_world = frame_ops.wrap_to_pi(heading_scene_frame + frame_heading)
    dimensions = track_row[:, contract.AGENT_DIMENSIONS]
    return np.concatenate(
        [
            positions_world,
            dimensions,
            heading_world[:, np.newaxis],
            velocities_world,
        ],
        axis=1,
    )


def scenario_world_frame_ground_truth(
    scenario: loader.StagedScenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds world-frame ground truth state, validity, and object type for each
    track in a scenario, for scoring against it.
    """
    track_rows = scenario.track_rows
    frame_origin = scenario.frame_origin
    frame_heading = scenario.frame_heading
    world_frame_states = np.stack([
        track_row_world_frame_state(track_row, frame_origin, frame_heading)
        for track_row in track_rows
    ]).astype(np.float32)
    type_onehots = track_rows[:, contract.CURRENT_STEP_INDEX,
                              contract.AGENT_TYPE]
    object_types = type_onehots.argmax(axis=-1).astype(np.int64) + 1
    return (
        world_frame_states,
        scenario.track_valid,
        object_types,
    )


class SubmissionArrays(NamedTuple):
    """A submission .npz, one row per predicted target, in the world frame."""
    scenario_id: np.ndarray  # (targets,)
    track_id: np.ndarray  # (targets,)
    world_trajectories: np.ndarray  # (targets, 6, 16, 2) at 2 Hz
    confidences: np.ndarray  # (targets, 6)


class MotionMetricInputs(NamedTuple):
    """The padded scenario-by-agent tensors Waymo's motion metrics op reads,
    named as its keyword arguments. S scenarios, P predictions and A agents in
    the largest scenario, M modes, K submitted steps, 91 logged steps.
    """
    prediction_trajectory: np.ndarray  # (S, P, M, 1, K, 2)
    prediction_score: np.ndarray  # (S, P, M)
    ground_truth_trajectory: np.ndarray  # (S, A, 91, 7)
    ground_truth_is_valid: np.ndarray  # (S, A, 91)
    prediction_ground_truth_indices: np.ndarray  # (S, P, 1)
    prediction_ground_truth_indices_mask: np.ndarray  # (S, P, 1)
    object_type: np.ndarray  # (S, A)
    object_id: np.ndarray  # (S, A)
    scenario_id: np.ndarray  # (S,)


def prediction_rows_by_scenario(
        scenario_ids: np.ndarray) -> dict[str, list[int]]:
    """Groups the flat per-target prediction rows by scenario in first-seen
    order, because the metrics op wants one row per scenario.
    """
    prediction_rows_of_scenario = {}
    for prediction_index, scenario_id in enumerate(scenario_ids):
        prediction_rows_of_scenario.setdefault(scenario_id,
                                               []).append(prediction_index)
    return prediction_rows_of_scenario


def build_motion_metric_tensors(
        predictions: SubmissionArrays,
        staged_directory: Path | str) -> MotionMetricInputs:
    """Reshapes flat per-target predictions and staged ground truth into the
    padded scenario-by-agent tensors the metrics op expects.
    """
    track_ids = predictions.track_id
    world_trajectories = predictions.world_trajectories
    confidences = predictions.confidences
    prediction_rows_of_scenario = prediction_rows_by_scenario(
        predictions.scenario_id)
    ordered_scenario_ids = list(prediction_rows_of_scenario)
    scenarios = [
        loader.read_scenario(Path(staged_directory) / f"{scenario_id}.npz")
        for scenario_id in ordered_scenario_ids
    ]

    # The padded sizes give every scenario room for the most agents and
    # the most predictions any scenario has.
    scenario_count = len(ordered_scenario_ids)
    max_agents = max(len(scenario.track_ids) for scenario in scenarios)
    max_predictions = max(
        len(rows) for rows in prediction_rows_of_scenario.values())
    mode_count, submitted_steps = world_trajectories.shape[1:3]
    per_prediction = (scenario_count, max_predictions)
    per_agent = (scenario_count, max_agents)
    step_count = contract.TOTAL_STEPS

    tensors = MotionMetricInputs(
        prediction_trajectory=np.zeros(
            per_prediction + (mode_count, 1, submitted_steps, 2), np.float32),
        prediction_score=np.zeros(per_prediction + (mode_count,), np.float32),
        ground_truth_trajectory=np.zeros(per_agent + (step_count, 7),
                                         np.float32),
        ground_truth_is_valid=np.zeros(per_agent + (step_count,), bool),
        prediction_ground_truth_indices=np.zeros(per_prediction + (1,),
                                                 np.int64),
        prediction_ground_truth_indices_mask=np.zeros(per_prediction + (1,),
                                                      bool),
        object_type=np.zeros(per_agent, np.int64),
        object_id=np.zeros(per_agent, np.int64),
        scenario_id=np.array(ordered_scenario_ids),
    )

    for scenario_index, scenario_id in enumerate(ordered_scenario_ids):
        scenario = scenarios[scenario_index]
        scenario_track_ids = scenario.track_ids
        agent_count = len(scenario_track_ids)

        # Ground truth holds every agent of the scenario.
        world_frame_states, track_valid, object_types = (
            scenario_world_frame_ground_truth(scenario))
        agents = (scenario_index, slice(0, agent_count))
        tensors.ground_truth_trajectory[agents] = world_frame_states
        tensors.ground_truth_is_valid[agents] = track_valid
        tensors.object_type[agents] = object_types
        tensors.object_id[agents] = scenario_track_ids

        # Each prediction points at its agent by index, not track id.
        for slot_index, prediction_index in enumerate(
                prediction_rows_of_scenario[scenario_id]):
            track_id = track_ids[prediction_index]
            matching_track_indices = np.flatnonzero(
                scenario_track_ids == track_id)
            assert len(matching_track_indices) == 1, (
                f"track {track_id} appears"
                f" {len(matching_track_indices)} times in scenario"
                f" {scenario_id}")
            slot = (scenario_index, slot_index)
            tensors.prediction_trajectory[slot][:, 0] = (
                world_trajectories[prediction_index])
            tensors.prediction_score[slot] = confidences[prediction_index]
            tensors.prediction_ground_truth_indices[slot] = (
                matching_track_indices[0])
            tensors.prediction_ground_truth_indices_mask[slot] = True

    return tensors


def load_predictions(predictions_path: Path) -> SubmissionArrays:
    """Loads a submission .npz and refuses one this code did not produce."""
    with np.load(predictions_path) as predictions_file:
        contract.check_artifact_provenance(
            (predictions_file["provenance"]
             if "provenance" in predictions_file else None),
            predictions_path,
            "Regenerate the predictions with submit.py.",
        )
        return SubmissionArrays(
            **
            {name: predictions_file[name] for name in SubmissionArrays._fields})


def print_score_table(breakdown_names: list[str],
                      metric_columns: tuple[np.ndarray, ...]) -> None:
    """Prints one row per breakdown with minADE, minFDE, miss rate, overlap
    rate and mAP, in the order Waymo's op returns them.
    """
    min_ade, min_fde, miss_rate, overlap_rate, mean_average_precision = (
        metric_columns)
    print(f"{'breakdown':32s} {'minADE':>10s} {'minFDE':>10s} {'missRate':>10s}"
          f" {'overlapRate':>12s} {'mAP':>10s}")
    for index, name in enumerate(breakdown_names):
        print(f"{name:32s} {min_ade[index]:10.4f} {min_fde[index]:10.4f}"
              f" {miss_rate[index]:10.4f} {overlap_rate[index]:12.4f}"
              f" {mean_average_precision[index]:10.4f}")


def run_score(predictions_path: Path, staged_directory: Path) -> None:
    """Scores a submission .npz with Waymo's motion metrics op.

    The only place waymo_open_dataset is imported; runs inside the container.
    """
    import tensorflow as tf
    from google.protobuf import text_format
    from waymo_open_dataset.metrics.ops import py_metrics_ops
    from waymo_open_dataset.metrics.python import config_util_py
    from waymo_open_dataset.protos import motion_metrics_pb2

    predictions = load_predictions(predictions_path)
    tensors = build_motion_metric_tensors(predictions, staged_directory)

    config = motion_metrics_pb2.MotionMetricsConfig()
    text_format.Parse(WAYMO_TUTORIAL_MOTION_METRICS_CONFIG_TEXT, config)
    breakdown_names = config_util_py.get_breakdown_names_from_motion_config(
        config)

    tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    with graph.as_default():
        motion_metric_ops = py_metrics_ops.motion_metrics(
            config=config.SerializeToString(),
            **tensors._asdict(),
        )
    with tf.compat.v1.Session(graph=graph) as session:
        metric_columns = session.run(motion_metric_ops)

    target_count = int(tensors.prediction_ground_truth_indices_mask.sum())
    scenario_count = tensors.object_type.shape[0]
    print(f"{target_count} designated targets across"
          f" {scenario_count} scenarios scored from {predictions_path}")
    print_score_table(breakdown_names, metric_columns)


def base_environment() -> dict[str, str]:
    """Copies the current environment with PYTHONPATH stripped, as a base for
    building a child process environment from.
    """
    return {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }


def generate_container_local_protos(generated_root: Path) -> None:
    """Compiles the vendored .proto files apart from Waymo's installed protos so
    the two sets can be imported and compared separately.
    """
    generated_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "protoc",
            f"--proto_path={REPOSITORY_ROOT / 'proto'}",
            f"--python_out={generated_root}",
            str(REPOSITORY_ROOT / "proto/womd_protos/scenario.proto"),
            str(REPOSITORY_ROOT / "proto/womd_protos/map.proto"),
        ],
        check=True,
    )
    (generated_root / "womd_protos").mkdir(parents=True, exist_ok=True)
    (generated_root / "womd_protos" / "__init__.py").touch()


def fields_disagree(ours_value: Any, theirs_value: Any) -> bool:
    """Compares two decoded field values, using a tolerance for floats and exact
    equality otherwise.
    """
    if isinstance(ours_value, float) or isinstance(theirs_value, float):
        return (abs(ours_value - theirs_value)
                > FLOATING_POINT_DISAGREEMENT_TOLERANCE)
    return ours_value != theirs_value


def compare_scenario_fields(scenario_index: int,
                            ours: Any,
                            theirs: Any,
                            path: str = "") -> int:
    """Recursively compares two decoded scenarios field by field, printing each
    disagreement and returning the count found.
    """
    if isinstance(ours, dict):
        assert set(ours.keys()) == set(theirs.keys()), (
            f"scenario {scenario_index}{path}: key sets differ,"
            f" ours={sorted(ours.keys())} theirs={sorted(theirs.keys())}")
        return sum(
            compare_scenario_fields(
                scenario_index,
                ours[key],
                theirs[key],
                f"{path}.{key}",
            ) for key in ours)
    if isinstance(ours, list):
        assert len(ours) == len(theirs), (
            f"scenario {scenario_index}{path}: length differs,"
            f" ours={len(ours)} theirs={len(theirs)}")
        return sum(
            compare_scenario_fields(
                scenario_index,
                ours_item,
                theirs_item,
                f"{path}[{index}]",
            )
            for index, (ours_item, theirs_item) in enumerate(zip(ours, theirs)))
    if fields_disagree(ours, theirs):
        print(f"DISAGREEMENT scenario {scenario_index}{path}:"
              f" ours={ours!r} theirs={theirs!r}")
        return 1
    return 0


def run_probe(role: str, shard_path: Path, sample_count: int,
              environment: dict[str, str]) -> list[dict[str, Any]]:
    """Runs reader_probe.py in its own process under one set of protos and
    returns the scenarios it decoded.
    """
    probe_script = Path(__file__).resolve().with_name("reader_probe.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(probe_script),
            "--role",
            role,
            "--shard-path",
            str(shard_path),
            "--sample-count",
            str(sample_count),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def run_check_reader(shard_path: Path, sample_count: int) -> None:
    """Runs reader_probe.py under both proto sets and diffs the decoded
    scenarios field by field, exiting 1 on disagreement.
    """
    generated_root = Path("/tmp/container_local_protos")
    generate_container_local_protos(generated_root)

    # Ours runs with the freshly generated protos first on the path;
    # theirs runs with only Waymo's installed package.
    ours_environment = {
        **base_environment(),
        "PYTHONPATH": f"{generated_root}:{REPOSITORY_ROOT}",
    }
    ours_scenarios = run_probe("ours", shard_path, sample_count,
                               ours_environment)
    theirs_scenarios = run_probe("theirs", shard_path, sample_count,
                                 base_environment())
    assert len(ours_scenarios) == len(theirs_scenarios), (
        f"ours parsed {len(ours_scenarios)} scenarios, theirs"
        f" parsed {len(theirs_scenarios)}")

    disagreement_count = sum(
        compare_scenario_fields(scenario_index, ours, theirs)
        for scenario_index, (
            ours, theirs) in enumerate(zip(ours_scenarios, theirs_scenarios)))
    if disagreement_count == 0:
        print(f"{len(ours_scenarios)} scenarios decoded identically:"
              f" our protos agree with Waymo's on every field")
    else:
        print(f"{disagreement_count} field disagreements across"
              f" {len(ours_scenarios)} scenarios")
        sys.exit(1)


def main() -> None:
    """Container entry point that scores a predictions file or checks the
    vendored protos against Waymo's.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("predictions_path", type=Path)
    score_parser.add_argument("staged_directory", type=Path)

    check_reader_parser = subparsers.add_parser("check-reader")
    check_reader_parser.add_argument("shard_path", type=Path)
    check_reader_parser.add_argument("sample_count", type=int)

    arguments = parser.parse_args()

    installed_version = importlib.metadata.version(
        WAYMO_OPEN_DATASET_DISTRIBUTION_NAME)
    print(f"{WAYMO_OPEN_DATASET_DISTRIBUTION_NAME}=={installed_version}")

    if arguments.command == "score":
        run_score(arguments.predictions_path, arguments.staged_directory)
    elif arguments.command == "check-reader":
        run_check_reader(arguments.shard_path, arguments.sample_count)


if __name__ == "__main__":
    main()
