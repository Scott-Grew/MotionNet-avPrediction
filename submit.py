"""Step 4 of 5.

Runs a trained model over staged scenarios and writes 6 world-frame futures per
designated target.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import functools
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from womd import baseline, contract, frame_ops, loader, pipeline, pruning
from womd.checkpoint import load_predictor
from womd.frame_ops import Frame
from womd.loader import SceneBatch, SceneSample
from womd.model import MotionPredictor

# A prediction function takes a batch and returns every target's modes,
# trajectories (B, modes, 80, 2) and confidence logits (B, modes).
Modes = tuple[Tensor, Tensor]
Predict = Callable[[SceneBatch], Modes]


class Predictions(NamedTuple):
    """One scene's kept futures, (targets, 6, 16, 2) agent-frame positions
    and (targets, 6) confidences.
    """

    trajectories: ndarray
    confidences: ndarray


class SubmissionRows(NamedTuple):
    """Every designated target's futures, as the submission file stores them."""

    scenario_ids: list[str]
    track_ids: list[int]
    world_trajectories: list[ndarray]  # each (6, 16, 2)
    confidences: list[ndarray]  # each (6,)


def model_modes(predictor: MotionPredictor, batch: SceneBatch) -> Modes:
    predicted = predictor(batch)
    return predicted.trajectories, predicted.confidence_logits


def constant_velocity_modes(batch: SceneBatch) -> Modes:
    """The control: each target carried on at its current velocity."""
    return baseline.constant_velocity(batch.targets.agent_history)


def predict_scene(predict: Predict, sample: SceneSample) -> Predictions:
    """One scene's modes at 10 Hz become Waymo's 6 kept modes at 2 Hz."""
    batch = pipeline.torch_batch(loader.scene_batch([sample]))
    with torch.no_grad():
        trajectories, logits = predict(batch)
    submitted = pruning.submitted_modes(trajectories, logits)
    kept = submitted.trajectories.numpy()
    return Predictions(kept, submitted.confidences.numpy())


def to_world(positions: ndarray, agent: Frame, scene: Frame) -> ndarray:
    """Agent-frame positions out to the world, through the scene frame."""
    scene_positions = frame_ops.positions_from_frame(positions, agent)
    return frame_ops.positions_from_frame(scene_positions, scene)


def submission_rows(predict: Predict, staged: Path) -> SubmissionRows:
    """Predicts the designated targets one scene at a time and converts each
    to the world frame.
    """
    rows = SubmissionRows([], [], [], [])
    for path in sorted(staged.glob("*.npz")):
        scenario, sample = loader.designated_scene(path)
        predicted = predict_scene(predict, sample)
        kept = zip(sample.targets, *predicted, strict=True)
        for target, paths, confidences in kept:
            world = to_world(paths, target.frame, scenario.scene_frame)
            rows.scenario_ids.append(scenario.scenario_id)
            rows.track_ids.append(target.track_id)
            rows.world_trajectories.append(world)
            rows.confidences.append(confidences)
    return rows


def write_submission(rows: SubmissionRows, staged: Path, output: Path) -> None:
    """Writes the compressed submission .npz with provenance."""
    trajectories = np.stack(rows.world_trajectories)
    expected = (contract.NUM_PREDICTED_MODES, contract.SUBMISSION_STEPS, 2)
    if trajectories.shape[1:] != expected:
        raise ValueError(
            f"submission trajectories have shape {trajectories.shape}"
        )
    submission = contract.Submission(
        scenario_id=np.array(rows.scenario_ids),
        track_id=np.array(rows.track_ids, dtype=np.int64),
        world_trajectories=trajectories,
        confidences=np.stack(rows.confidences).astype(np.float64),
        provenance=contract.artifact_provenance("submit.py", staged),
    )
    submission.save(output)


def main() -> None:
    """Writes a submission .npz from a trained model or from the constant-
    velocity baseline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--constant-velocity", action="store_true")
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    if arguments.constant_velocity:
        if len(arguments.paths) != 2:
            raise SystemExit(
                "usage: submit.py --constant-velocity STAGED OUTPUT"
            )
        staged, output = arguments.paths
        predict = constant_velocity_modes
    else:
        if len(arguments.paths) != 3:
            raise SystemExit("usage: submit.py CHECKPOINT STAGED OUTPUT")
        checkpoint_path, staged, output = arguments.paths
        predictor = load_predictor(checkpoint_path)
        predict = functools.partial(model_modes, predictor)

    rows = submission_rows(predict, staged)
    write_submission(rows, staged, output)
    size = output.stat().st_size
    print(
        f"{len(rows.track_ids)} designated targets written to {output}"
        f" ({size} bytes)"
    )


if __name__ == "__main__":
    main()
