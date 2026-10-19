"""Runs inside the container, feeding predictions and ground truth to Waymo's
metrics op and checking our protos against theirs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPOSITORY_ROOT))

import womd.runtime_env  # noqa: E402, F401

import numpy as np  # noqa: E402

from womd import contract, frame_ops, loader  # noqa: E402
from womd.frame_ops import Frame  # noqa: E402
from womd.loader import StagedScenario  # noqa: E402

# The pip package holding Waymo's metrics; its installed version is
# printed with each run.
WAYMO_OPEN_DATASET_DISTRIBUTION_NAME = "waymo-open-dataset-tf-2-12-0"
# Two decoders may differ by this much on a decimal field and still
# count as agreeing.
FLOATING_POINT_DISAGREEMENT_TOLERANCE = 1e-4
# Where the vendored protos are compiled for the reader check.
CONTAINER_LOCAL_PROTOS = Path("/tmp/container_local_protos")

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


class SubmissionArrays(NamedTuple):
    """A submission .npz, one row per predicted target, in the world frame."""

    scenario_id: np.ndarray  # (targets,)
    track_id: np.ndarray  # (targets,)
    world_trajectories: np.ndarray  # (targets, 6, 16, 2) at 2 Hz
    confidences: np.ndarray  # (targets, 6)


class MetricInputs(NamedTuple):
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


class Padding(NamedTuple):
    """The padded tensor sizes, room for the largest scenario."""

    scenarios: int
    agents: int
    predictions: int
    modes: int
    steps: int  # submitted steps per mode


class GroundTruth(NamedTuple):
    """Every track of one scenario in the world frame."""

    states: np.ndarray  # (T, 91, 7) x, y, length, width, heading, vx, vy
    valid: np.ndarray  # (T, 91)
    object_types: np.ndarray  # (T,) Waymo's numbering, vehicle 1


def world_states(rows: np.ndarray, frame: Frame) -> np.ndarray:
    """Scene-frame agent rows (..., 13) to the 7 world-frame columns the
    metrics op wants, x, y, length, width, heading, vx and vy.
    """
    positions = frame_ops.positions_from_frame(
        rows[..., contract.AGENT_POSITION], frame
    )
    velocities = frame_ops.directions_from_frame(
        rows[..., contract.AGENT_VELOCITY], frame.heading
    )
    sine = rows[..., contract.AGENT_HEADING_SINE]
    cosine = rows[..., contract.AGENT_HEADING_COSINE]
    headings = frame_ops.wrap_to_pi(np.arctan2(sine, cosine) + frame.heading)
    dimensions = rows[..., contract.AGENT_DIMENSIONS]
    columns = [positions, dimensions, headings[..., np.newaxis], velocities]
    return np.concatenate(columns, axis=-1).astype(np.float32)


def ground_truth(scenario: StagedScenario) -> GroundTruth:
    states = world_states(scenario.track_rows, scenario.scene_frame)
    now_types = scenario.track_rows[
        :, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE
    ]
    object_types = now_types.argmax(axis=-1).astype(np.int64) + 1
    return GroundTruth(states, scenario.track_valid, object_types)


def rows_by_scenario(scenario_ids: np.ndarray) -> dict[str, list[int]]:
    """Groups the flat per-target prediction rows by scenario in first-seen
    order, because the metrics op wants one row per scenario.
    """
    rows: dict[str, list[int]] = {}
    for row, scenario_id in enumerate(scenario_ids):
        rows.setdefault(scenario_id, []).append(row)
    return rows


def empty_inputs(padding: Padding, scenario_ids: list[str]) -> MetricInputs:
    per_prediction = (padding.scenarios, padding.predictions)
    per_agent = (padding.scenarios, padding.agents)
    logged = contract.TOTAL_STEPS
    return MetricInputs(
        prediction_trajectory=np.zeros(
            per_prediction + (padding.modes, 1, padding.steps, 2), np.float32
        ),
        prediction_score=np.zeros(
            per_prediction + (padding.modes,), np.float32
        ),
        ground_truth_trajectory=np.zeros(per_agent + (logged, 7), np.float32),
        ground_truth_is_valid=np.zeros(per_agent + (logged,), bool),
        prediction_ground_truth_indices=np.zeros(
            per_prediction + (1,), np.int64
        ),
        prediction_ground_truth_indices_mask=np.zeros(
            per_prediction + (1,), bool
        ),
        object_type=np.zeros(per_agent, np.int64),
        object_id=np.zeros(per_agent, np.int64),
        scenario_id=np.array(scenario_ids),
    )


class InputsBuilder:
    """Reads each predicted scenario's staged ground truth and fills the
    metric inputs one scenario at a time.
    """

    def __init__(self, predictions: SubmissionArrays, staged: Path) -> None:
        self.predictions = predictions
        rows_of = rows_by_scenario(predictions.scenario_id)
        self.prediction_rows = list(rows_of.values())
        self.scenarios = []
        for scenario_id in rows_of:
            self.scenarios.append(
                loader.read_scenario(staged / f"{scenario_id}.npz")
            )
        modes, steps = predictions.world_trajectories.shape[1:3]
        padding = Padding(
            scenarios=len(self.scenarios),
            agents=max(len(scenario.track_ids) for scenario in self.scenarios),
            predictions=max(len(rows) for rows in self.prediction_rows),
            modes=modes,
            steps=steps,
        )
        self.tensors = empty_inputs(padding, list(rows_of))

    def add_ground_truth(self, row: int, scenario: StagedScenario) -> None:
        """Ground truth holds every agent of the scenario."""
        truth = ground_truth(scenario)
        agents = (row, slice(0, len(scenario.track_ids)))
        self.tensors.ground_truth_trajectory[agents] = truth.states
        self.tensors.ground_truth_is_valid[agents] = truth.valid
        self.tensors.object_type[agents] = truth.object_types
        self.tensors.object_id[agents] = scenario.track_ids

    def add_predictions(self, row: int, scenario: StagedScenario) -> None:
        """Each prediction points at its agent by index, not track id."""
        predictions = self.predictions
        tensors = self.tensors
        for slot_index, prediction_row in enumerate(self.prediction_rows[row]):
            track_id = predictions.track_id[prediction_row]
            matches = np.flatnonzero(scenario.track_ids == track_id)
            if len(matches) != 1:
                raise ValueError(
                    f"track {track_id} appears {len(matches)} times"
                    f" in {scenario.scenario_id}"
                )
            slot = (row, slot_index)
            tensors.prediction_trajectory[slot][:, 0] = (
                predictions.world_trajectories[prediction_row]
            )
            tensors.prediction_score[slot] = predictions.confidences[
                prediction_row
            ]
            tensors.prediction_ground_truth_indices[slot] = matches[0]
            tensors.prediction_ground_truth_indices_mask[slot] = True

    def build(self) -> MetricInputs:
        for row, scenario in enumerate(self.scenarios):
            self.add_ground_truth(row, scenario)
            self.add_predictions(row, scenario)
        return self.tensors


def metric_inputs(predictions: SubmissionArrays, staged: Path) -> MetricInputs:
    """Reshapes flat per-target predictions and staged ground truth into the
    padded scenario-by-agent tensors the metrics op expects.
    """
    return InputsBuilder(predictions, staged).build()


def load_predictions(predictions_path: Path) -> SubmissionArrays:
    """Loads a submission .npz and refuses one this code did not produce."""
    with np.load(predictions_path) as predictions_file:
        stamp = (
            predictions_file["provenance"]
            if "provenance" in predictions_file
            else None
        )
        contract.check_provenance(
            stamp,
            predictions_path,
            "Regenerate the predictions with submit.py.",
        )
        arrays = {
            name: predictions_file[name] for name in SubmissionArrays._fields
        }
    return SubmissionArrays(**arrays)


def print_score_table(names: list[str], columns: Sequence[np.ndarray]) -> None:
    """Prints one row per breakdown with minADE, minFDE, miss rate, overlap
    rate and mAP, in the order Waymo's op returns them.
    """
    min_ade, min_fde, miss_rate, overlap_rate, mean_average_precision = columns
    print(
        f"{'breakdown':32s} {'minADE':>10s} {'minFDE':>10s} {'missRate':>10s}"
        f" {'overlapRate':>12s} {'mAP':>10s}"
    )
    for index, name in enumerate(names):
        print(
            f"{name:32s} {min_ade[index]:10.4f} {min_fde[index]:10.4f}"
            f" {miss_rate[index]:10.4f} {overlap_rate[index]:12.4f}"
            f" {mean_average_precision[index]:10.4f}"
        )


def run_score(predictions_path: Path, staged: Path) -> None:
    """Scores a submission .npz with Waymo's motion metrics op. The only place
    waymo_open_dataset is imported; runs inside the container.
    """
    import tensorflow as tf
    from google.protobuf import text_format
    from waymo_open_dataset.metrics.ops import py_metrics_ops
    from waymo_open_dataset.metrics.python import config_util_py
    from waymo_open_dataset.protos import motion_metrics_pb2

    predictions = load_predictions(predictions_path)
    tensors = metric_inputs(predictions, staged)
    config = motion_metrics_pb2.MotionMetricsConfig()
    text_format.Parse(WAYMO_TUTORIAL_MOTION_METRICS_CONFIG_TEXT, config)
    names = config_util_py.get_breakdown_names_from_motion_config(config)
    serialised = config.SerializeToString()

    tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    with graph.as_default():
        metric_ops = py_metrics_ops.motion_metrics(
            config=serialised, **tensors._asdict()
        )
    with tf.compat.v1.Session(graph=graph) as session:
        columns = session.run(metric_ops)

    target_count = int(tensors.prediction_ground_truth_indices_mask.sum())
    scenario_count = tensors.object_type.shape[0]
    print(
        f"{target_count} designated targets across {scenario_count}"
        f" scenarios scored from {predictions_path}"
    )
    print_score_table(names, columns)


def base_environment() -> dict[str, str]:
    """The current environment with PYTHONPATH stripped, as a base for a
    child process environment.
    """
    return {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }


def generate_container_local_protos(generated_root: Path) -> None:
    """Compiles the vendored .proto files apart from Waymo's installed protos so
    the two sets can be imported and compared separately.
    """
    generated_root.mkdir(parents=True, exist_ok=True)
    command = [
        "protoc",
        f"--proto_path={REPOSITORY_ROOT / 'proto'}",
        f"--python_out={generated_root}",
        str(REPOSITORY_ROOT / "proto/womd_protos/scenario.proto"),
        str(REPOSITORY_ROOT / "proto/womd_protos/map.proto"),
    ]
    subprocess.run(command, check=True)
    (generated_root / "womd_protos").mkdir(parents=True, exist_ok=True)
    (generated_root / "womd_protos" / "__init__.py").touch()


def fields_disagree(ours: Any, theirs: Any) -> bool:
    """Compares two decoded field values, using a tolerance for floats and exact
    equality otherwise.
    """
    if isinstance(ours, float) or isinstance(theirs, float):
        return abs(ours - theirs) > FLOATING_POINT_DISAGREEMENT_TOLERANCE
    return ours != theirs


class FieldComparison:
    """Compares two decodings of one scenario field by field, printing each
    disagreement.
    """

    def __init__(self, scenario_index: int) -> None:
        self.scenario_index = scenario_index

    def disagreements(self, ours: Any, theirs: Any, path: str) -> int:
        """Recursively counts the disagreements under one field path."""
        where = f"scenario {self.scenario_index}{path}"
        if isinstance(ours, dict):
            if set(ours) != set(theirs):
                raise ValueError(
                    f"{where}: key sets differ,"
                    f" ours={sorted(ours)} theirs={sorted(theirs)}"
                )
            count = 0
            for key in ours:
                count += self.disagreements(
                    ours[key], theirs[key], f"{path}.{key}"
                )
            return count
        if isinstance(ours, list):
            if len(ours) != len(theirs):
                raise ValueError(
                    f"{where}: length differs,"
                    f" ours={len(ours)} theirs={len(theirs)}"
                )
            count = 0
            for index, (mine, other) in enumerate(zip(ours, theirs)):
                count += self.disagreements(mine, other, f"{path}[{index}]")
            return count
        if fields_disagree(ours, theirs):
            print(f"DISAGREEMENT {where}: ours={ours!r} theirs={theirs!r}")
            return 1
        return 0


class ReaderProbe:
    """Runs reader_probe.py in its own process under one set of protos and
    returns the scenarios it decoded.
    """

    def __init__(self, shard_path: Path, sample_count: int) -> None:
        self.shard_path = shard_path
        self.sample_count = sample_count
        self.script = Path(__file__).resolve().with_name("reader_probe.py")

    def run(self, role: str, environment: dict[str, str]) -> list[Any]:
        command = [
            sys.executable,
            str(self.script),
            "--role",
            role,
            "--shard-path",
            str(self.shard_path),
            "--sample-count",
            str(self.sample_count),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, env=environment
        )
        return json.loads(completed.stdout)


def run_check_reader(shard_path: Path, sample_count: int) -> None:
    """Runs reader_probe.py under both proto sets and diffs the decoded
    scenarios field by field, exiting 1 on disagreement.
    """
    generate_container_local_protos(CONTAINER_LOCAL_PROTOS)
    probe = ReaderProbe(shard_path, sample_count)
    # Ours runs with the freshly generated protos first on the path;
    # theirs runs with only Waymo's installed package.
    ours_environment = base_environment()
    ours_environment["PYTHONPATH"] = (
        f"{CONTAINER_LOCAL_PROTOS}:{REPOSITORY_ROOT}"
    )
    ours_scenarios = probe.run("ours", ours_environment)
    theirs_scenarios = probe.run("theirs", base_environment())
    if len(ours_scenarios) != len(theirs_scenarios):
        raise ValueError(
            f"ours parsed {len(ours_scenarios)} scenarios,"
            f" theirs parsed {len(theirs_scenarios)}"
        )
    disagreement_count = 0
    for index, (ours, theirs) in enumerate(
        zip(ours_scenarios, theirs_scenarios)
    ):
        disagreement_count += FieldComparison(index).disagreements(
            ours, theirs, ""
        )
    if disagreement_count == 0:
        print(
            f"{len(ours_scenarios)} scenarios decoded identically:"
            " our protos agree with Waymo's on every field"
        )
        return
    print(
        f"{disagreement_count} field disagreements"
        f" across {len(ours_scenarios)} scenarios"
    )
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
        WAYMO_OPEN_DATASET_DISTRIBUTION_NAME
    )
    print(f"{WAYMO_OPEN_DATASET_DISTRIBUTION_NAME}=={installed_version}")
    if arguments.command == "score":
        run_score(arguments.predictions_path, arguments.staged_directory)
    else:
        run_check_reader(arguments.shard_path, arguments.sample_count)


if __name__ == "__main__":
    main()
