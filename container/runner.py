import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPOSITORY_ROOT))

import womd.runtime_env
import numpy as np

from womd import contract, frame_ops, loader

WAYMO_OPEN_DATASET_DISTRIBUTION_NAME = "waymo-open-dataset-tf-2-12-0"
FLOATING_POINT_DISAGREEMENT_TOLERANCE = 1e-4

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


def track_row_world_frame_state(
    track_row, frame_origin, frame_heading
):
    positions_world = frame_ops.positions_to_world_frame(
        track_row[:, contract.AGENT_POSITION],
        frame_origin,
        frame_heading,
    )
    velocities_world = frame_ops.positions_to_world_frame(
        track_row[:, contract.AGENT_VELOCITY],
        np.zeros(2),
        frame_heading,
    )
    heading_storage_frame = np.arctan2(
        track_row[:, contract.AGENT_HEADING_SINE],
        track_row[:, contract.AGENT_HEADING_COSINE],
    )
    heading_world = frame_ops.wrap_to_pi(
        heading_storage_frame + frame_heading
    )
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


def scenario_world_frame_ground_truth(scenario_array):
    track_rows = scenario_array["track_rows"]
    frame_origin = scenario_array["frame_origin"]
    frame_heading = float(scenario_array["frame_heading"])
    world_frame_states = np.stack(
        [
            track_row_world_frame_state(
                track_row, frame_origin, frame_heading
            )
            for track_row in track_rows
        ]
    ).astype(np.float32)
    type_onehots = track_rows[
        :, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE
    ]
    object_types = type_onehots.argmax(axis=-1).astype(np.int64) + 1
    return (
        world_frame_states,
        scenario_array["track_valid"],
        object_types,
    )


def build_motion_metric_tensors(predictions, staged_directory):
    scenario_ids = predictions["scenario_id"]
    track_ids = predictions["track_id"]
    world_trajectories = predictions["world_trajectories"]
    confidences = predictions["confidences"]

    prediction_rows_of_scenario = {}
    for prediction_index, scenario_id in enumerate(scenario_ids):
        prediction_rows_of_scenario.setdefault(
            scenario_id, []
        ).append(prediction_index)
    ordered_scenario_ids = list(prediction_rows_of_scenario)

    scenario_arrays = [
        loader.read_scenario(
            Path(staged_directory) / f"{scenario_id}.npz"
        )
        for scenario_id in ordered_scenario_ids
    ]
    scenario_count = len(ordered_scenario_ids)
    max_agents = max(
        len(scenario_array["track_ids"])
        for scenario_array in scenario_arrays
    )
    max_predictions = max(
        len(rows) for rows in prediction_rows_of_scenario.values()
    )
    mode_count = world_trajectories.shape[1]
    submitted_steps = world_trajectories.shape[2]

    prediction_trajectory = np.zeros(
        (
            scenario_count,
            max_predictions,
            mode_count,
            1,
            submitted_steps,
            2,
        ),
        dtype=np.float32,
    )
    prediction_score = np.zeros(
        (scenario_count, max_predictions, mode_count),
        dtype=np.float32,
    )
    ground_truth_trajectory = np.zeros(
        (scenario_count, max_agents, contract.TOTAL_STEPS, 7),
        dtype=np.float32,
    )
    ground_truth_is_valid = np.zeros(
        (scenario_count, max_agents, contract.TOTAL_STEPS), dtype=bool
    )
    prediction_ground_truth_indices = np.zeros(
        (scenario_count, max_predictions, 1), dtype=np.int64
    )
    prediction_ground_truth_indices_mask = np.zeros(
        (scenario_count, max_predictions, 1), dtype=bool
    )
    object_type = np.zeros(
        (scenario_count, max_agents), dtype=np.int64
    )
    object_id = np.zeros((scenario_count, max_agents), dtype=np.int64)

    for scenario_index, (scenario_id, scenario_array) in enumerate(
        zip(ordered_scenario_ids, scenario_arrays)
    ):
        world_frame_states, track_valid, object_types = (
            scenario_world_frame_ground_truth(scenario_array)
        )
        agent_count = len(scenario_array["track_ids"])
        ground_truth_trajectory[scenario_index, :agent_count] = (
            world_frame_states
        )
        ground_truth_is_valid[scenario_index, :agent_count] = (
            track_valid
        )
        object_type[scenario_index, :agent_count] = object_types
        object_id[scenario_index, :agent_count] = scenario_array[
            "track_ids"
        ]

        for slot_index, prediction_index in enumerate(
            prediction_rows_of_scenario[scenario_id]
        ):
            matching_track_indices = np.flatnonzero(
                scenario_array["track_ids"]
                == track_ids[prediction_index]
            )
            assert len(matching_track_indices) == 1, (
                f"track {track_ids[prediction_index]} appears {len(matching_track_indices)} times"
                f" in scenario {scenario_id}"
            )
            prediction_trajectory[
                scenario_index, slot_index, :, 0
            ] = world_trajectories[prediction_index]
            prediction_score[scenario_index, slot_index] = (
                confidences[prediction_index]
            )
            prediction_ground_truth_indices[
                scenario_index, slot_index, 0
            ] = matching_track_indices[0]
            prediction_ground_truth_indices_mask[
                scenario_index, slot_index, 0
            ] = True

    return {
        "prediction_trajectory": prediction_trajectory,
        "prediction_score": prediction_score,
        "ground_truth_trajectory": ground_truth_trajectory,
        "ground_truth_is_valid": ground_truth_is_valid,
        "prediction_ground_truth_indices": prediction_ground_truth_indices,
        "prediction_ground_truth_indices_mask": prediction_ground_truth_indices_mask,
        "object_type": object_type,
        "object_id": object_id,
        "scenario_id": np.array(ordered_scenario_ids),
    }


def run_score(predictions_path, staged_directory):
    import tensorflow as tf
    from google.protobuf import text_format
    from waymo_open_dataset.metrics.ops import py_metrics_ops
    from waymo_open_dataset.metrics.python import config_util_py
    from waymo_open_dataset.protos import motion_metrics_pb2

    with np.load(predictions_path) as predictions_file:
        predictions = {
            name: predictions_file[name]
            for name in predictions_file.files
        }
    contract.check_artifact_provenance(
        predictions.get("provenance"),
        predictions_path,
        "Regenerate the predictions with submit.py.",
    )

    tensors = build_motion_metric_tensors(
        predictions, staged_directory
    )

    config = motion_metrics_pb2.MotionMetricsConfig()
    text_format.Parse(
        WAYMO_TUTORIAL_MOTION_METRICS_CONFIG_TEXT, config
    )
    breakdown_names = (
        config_util_py.get_breakdown_names_from_motion_config(config)
    )

    tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    with graph.as_default():
        motion_metric_ops = py_metrics_ops.motion_metrics(
            config=config.SerializeToString(),
            prediction_trajectory=tensors["prediction_trajectory"],
            prediction_score=tensors["prediction_score"],
            ground_truth_trajectory=tensors[
                "ground_truth_trajectory"
            ],
            ground_truth_is_valid=tensors["ground_truth_is_valid"],
            prediction_ground_truth_indices=tensors[
                "prediction_ground_truth_indices"
            ],
            prediction_ground_truth_indices_mask=tensors[
                "prediction_ground_truth_indices_mask"
            ],
            object_type=tensors["object_type"],
            object_id=tensors["object_id"],
            scenario_id=tensors["scenario_id"],
        )
    with tf.compat.v1.Session(graph=graph) as session:
        (
            min_ade,
            min_fde,
            miss_rate,
            overlap_rate,
            mean_average_precision,
        ) = session.run(motion_metric_ops)

    target_count = int(
        tensors["prediction_ground_truth_indices_mask"].sum()
    )
    scenario_count = tensors["object_type"].shape[0]
    print(
        f"{target_count} designated targets across {scenario_count} scenarios scored"
        f" from {predictions_path}"
    )
    print(
        f"{'breakdown':32s} {'minADE':>10s} {'minFDE':>10s} {'missRate':>10s}"
        f" {'overlapRate':>12s} {'mAP':>10s}"
    )
    for index, name in enumerate(breakdown_names):
        print(
            f"{name:32s} {min_ade[index]:10.4f} {min_fde[index]:10.4f}"
            f" {miss_rate[index]:10.4f} {overlap_rate[index]:12.4f}"
            f" {mean_average_precision[index]:10.4f}"
        )


def base_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if key != "PYTHONPATH"
    }


def generate_container_local_protos(generated_root):
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
    (generated_root / "womd_protos").mkdir(
        parents=True, exist_ok=True
    )
    (generated_root / "womd_protos" / "__init__.py").touch()


def fields_disagree(ours_value, theirs_value):
    if isinstance(ours_value, float) or isinstance(
        theirs_value, float
    ):
        return (
            abs(ours_value - theirs_value)
            > FLOATING_POINT_DISAGREEMENT_TOLERANCE
        )
    return ours_value != theirs_value


def compare_scenario_fields(scenario_index, ours, theirs, path=""):
    if isinstance(ours, dict):
        assert set(ours.keys()) == set(theirs.keys()), (
            f"scenario {scenario_index}{path}: key sets differ,"
            f" ours={sorted(ours.keys())} theirs={sorted(theirs.keys())}"
        )
        return sum(
            compare_scenario_fields(
                scenario_index,
                ours[key],
                theirs[key],
                f"{path}.{key}",
            )
            for key in ours
        )
    if isinstance(ours, list):
        assert len(ours) == len(
            theirs
        ), f"scenario {scenario_index}{path}: length differs, ours={len(ours)} theirs={len(theirs)}"
        return sum(
            compare_scenario_fields(
                scenario_index,
                ours_item,
                theirs_item,
                f"{path}[{index}]",
            )
            for index, (ours_item, theirs_item) in enumerate(
                zip(ours, theirs)
            )
        )
    if fields_disagree(ours, theirs):
        print(
            f"DISAGREEMENT scenario {scenario_index}{path}: ours={ours!r} theirs={theirs!r}"
        )
        return 1
    return 0


def run_check_reader(shard_path, sample_count):
    generated_root = Path("/tmp/container_local_protos")
    generate_container_local_protos(generated_root)

    probe_script = (
        Path(__file__).resolve().with_name("reader_probe.py")
    )

    ours_environment = {
        **base_environment(),
        "PYTHONPATH": f"{generated_root}:{REPOSITORY_ROOT}",
    }
    theirs_environment = base_environment()

    ours_result = subprocess.run(
        [
            sys.executable,
            str(probe_script),
            "--role",
            "ours",
            "--shard-path",
            str(shard_path),
            "--sample-count",
            str(sample_count),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=ours_environment,
    )
    theirs_result = subprocess.run(
        [
            sys.executable,
            str(probe_script),
            "--role",
            "theirs",
            "--shard-path",
            str(shard_path),
            "--sample-count",
            str(sample_count),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=theirs_environment,
    )

    ours_scenarios = json.loads(ours_result.stdout)
    theirs_scenarios = json.loads(theirs_result.stdout)
    assert len(ours_scenarios) == len(
        theirs_scenarios
    ), f"ours parsed {len(ours_scenarios)} scenarios, theirs parsed {len(theirs_scenarios)}"

    disagreement_count = sum(
        compare_scenario_fields(scenario_index, ours, theirs)
        for scenario_index, (ours, theirs) in enumerate(
            zip(ours_scenarios, theirs_scenarios)
        )
    )
    if disagreement_count == 0:
        print(
            f"{len(ours_scenarios)} scenarios byte-exact: our reader agrees with Waymo's on every field"
        )
    else:
        print(
            f"{disagreement_count} field disagreements across {len(ours_scenarios)} scenarios"
        )
        sys.exit(1)


def main():
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
        WAYMO_OPEN_DATASET_DISTRIBUTION_NAME
    )
    print(
        f"{WAYMO_OPEN_DATASET_DISTRIBUTION_NAME}=={installed_version}"
    )

    if arguments.command == "score":
        run_score(
            arguments.predictions_path, arguments.staged_directory
        )
    elif arguments.command == "check-reader":
        run_check_reader(arguments.shard_path, arguments.sample_count)


if __name__ == "__main__":
    main()
