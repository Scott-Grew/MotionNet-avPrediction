"""Step 4 of 5.

Runs a trained model over staged scenarios and writes 6 world-frame futures per
designated target.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from numpy import ndarray

from womd import baseline, contract, frame_ops, loader, pipeline, pruning
from womd.checkpoint import load_anchor_file, load_checkpoint
from womd.frame_ops import Frame
from womd.loader import SceneSample, StagedScenario
from womd.model import MotionPredictor

# The 16 of the 80 predicted steps that Waymo scores, 2 Hz from 10 Hz.
SUBMISSION_STEP_SELECTOR = torch.tensor(contract.SUBMISSION_FUTURE_INDICES)

Predictor = MotionPredictor | None


class Scene(NamedTuple):
    """A staged scenario and the sample holding its designated targets."""

    scenario: StagedScenario
    sample: SceneSample


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


def designated_scenes(staged: Path) -> Iterator[Scene]:
    """Each staged scenario in file order with the targets Waymo designated
    for scoring.
    """
    for path in sorted(staged.glob("*.npz")):
        scenario = loader.read_scenario(path)
        designated_count = int(scenario.is_designated_target.sum())
        track_indices = loader.eligible_tracks(scenario, designated_only=True)
        if len(track_indices) != designated_count:
            raise ValueError(
                f"{path} designates {designated_count} targets,"
                f" {len(track_indices)} can be predicted"
            )
        yield Scene(
            scenario, loader.scene_sample(scenario, track_indices.tolist())
        )


def predict_scene(predictor: Predictor, sample: SceneSample) -> Predictions:
    """One scene's 54 modes at 10 Hz become Waymo's 6 kept modes at 2 Hz.
    predictor=None runs the constant-velocity null.
    """
    batch = pipeline.torch_batch(loader.scene_batch([sample]))
    with torch.no_grad():
        if predictor is None:
            trajectories, logits = baseline.constant_velocity(
                batch.targets.agent_history
            )
        else:
            trajectories, logits = predictor(batch)
    pruned = pruning.prune_modes(trajectories, logits)
    confidences = pruning.kept_confidences(
        trajectories, logits, pruned.trajectories
    )
    decimated = pruned.trajectories.index_select(2, SUBMISSION_STEP_SELECTOR)
    return Predictions(decimated.numpy(), confidences.numpy())


def to_world(positions: ndarray, agent: Frame, scene: Frame) -> ndarray:
    """Agent-frame positions out to the world, through the scene frame."""
    scene_positions = frame_ops.positions_from_frame(positions, agent)
    return frame_ops.positions_from_frame(scene_positions, scene)


def load_predictor(checkpoint: Path, anchors: Path) -> MotionPredictor:
    """The model built from its anchor file with trained weights, in eval
    mode.
    """
    predictor = MotionPredictor(load_anchor_file(anchors))
    loaded = load_checkpoint(checkpoint)
    predictor.load_state_dict(loaded["model_state"])
    return predictor.eval()


def submission_rows(predictor: Predictor, staged: Path) -> SubmissionRows:
    """Predicts the designated targets one scene at a time and converts each
    to the world frame.
    """
    rows = SubmissionRows([], [], [], [])
    for scenario, sample in designated_scenes(staged):
        predicted = predict_scene(predictor, sample)
        per_target = zip(
            sample.targets,
            predicted.trajectories,
            predicted.confidences,
            strict=True,
        )
        for target, paths, confidences in per_target:
            rows.scenario_ids.append(scenario.scenario_id)
            rows.track_ids.append(target.track_id)
            rows.world_trajectories.append(
                to_world(paths, target.frame, scenario.scene_frame)
            )
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
    np.savez_compressed(
        output,
        scenario_id=np.array(rows.scenario_ids),
        track_id=np.array(rows.track_ids, dtype=np.int64),
        world_trajectories=trajectories,
        confidences=np.stack(rows.confidences).astype(np.float64),
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
        predictor = None
    else:
        if len(arguments.paths) != 4:
            raise SystemExit(
                "usage: submit.py CHECKPOINT STAGED ANCHORS OUTPUT"
            )
        checkpoint_path, staged, anchors_path, output = arguments.paths
        predictor = load_predictor(checkpoint_path, anchors_path)

    rows = submission_rows(predictor, staged)
    write_submission(rows, staged, output)
    size = output.stat().st_size
    print(
        f"{len(rows.track_ids)} designated targets written to {output}"
        f" ({size} bytes)"
    )


if __name__ == "__main__":
    main()
