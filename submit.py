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

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from womd import baseline, contract, frame_ops, loader, pipeline, pruning
from womd.checkpoint import load_predictor
from womd.frame_ops import Frame
from womd.loader import SceneBatch
from womd.model import MotionPredictor
from womd.pruning import SubmittedModes

# A prediction function takes a batch and returns every target's modes,
# trajectories (B, modes, 80, 2) and confidence logits (B, modes).
Modes = tuple[Tensor, Tensor]
Predict = Callable[[SceneBatch], Modes]


def model_modes(predictor: MotionPredictor, batch: SceneBatch) -> Modes:
    predicted = predictor(batch)
    return predicted.trajectories, predicted.confidence_logits


def constant_velocity_modes(batch: SceneBatch) -> Modes:
    """The control: each target carried on at its current velocity."""
    return baseline.constant_velocity(batch.targets.agent_history)


def predict_scene(predict: Predict, batch: SceneBatch) -> SubmittedModes:
    """One scene's modes at 10 Hz become Waymo's 6 kept modes at 2 Hz, in
    each target's own frame.
    """
    with torch.no_grad():
        trajectories, logits = predict(batch)
    return pruning.submitted_modes(trajectories, logits)


def to_world(positions: ndarray, agent: Frame, scene: Frame) -> ndarray:
    """Agent-frame positions out to the world, through the scene frame."""
    scene_positions = frame_ops.positions_from_frame(positions, agent)
    return frame_ops.positions_from_frame(scene_positions, scene)


def submission(predict: Predict, staged: Path) -> contract.Submission:
    """Predicts the designated targets one scene at a time and converts each
    target's 6 futures to the world frame.
    """
    scenario_ids = []
    track_ids = []
    world_trajectories = []
    confidences = []
    for path in sorted(staged.glob("*.npz")):
        scenario = loader.read_scenario(path)
        tracks = loader.designated_tracks(scenario)
        batch = pipeline.torch_batch(loader.one_scene(scenario, tracks))
        kept = predict_scene(predict, batch)
        kept_paths = kept.trajectories.numpy()
        kept_rows = zip(tracks, kept_paths, kept.confidences.numpy())
        for track_index, paths, scores in kept_rows:
            frame = loader.agent_frame(scenario.track_rows[track_index])
            world = to_world(paths, frame, scenario.scene_frame)
            scenario_ids.append(scenario.scenario_id)
            track_ids.append(scenario.track_ids[track_index])
            world_trajectories.append(world)
            confidences.append(scores)
    return contract.Submission(
        scenario_id=np.array(scenario_ids),
        track_id=np.array(track_ids, dtype=np.int64),
        world_trajectories=np.stack(world_trajectories),
        confidences=np.stack(confidences).astype(np.float64),
        provenance=contract.artifact_provenance("submit.py", staged),
    )


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

    predicted = submission(predict, staged)
    predicted.save(output)
    size = output.stat().st_size
    print(
        f"{len(predicted.track_id)} designated targets written to {output}"
        f" ({size} bytes)"
    )


if __name__ == "__main__":
    main()
