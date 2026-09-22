"""Step 4 of 5.

Runs a trained model over staged scenarios and writes 6 world-frame futures per
designated target.
"""
from __future__ import annotations

import womd.runtime_env
import argparse
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from womd import (
    baseline,
    contract,
    frame_ops,
    loader,
    model,
    pipeline,
    pruning,
)
from womd.checkpoint import load_anchor_file, load_checkpoint_state

# The 16 of the 80 predicted steps that Waymo scores, 2 Hz from 10 Hz.
SUBMISSION_STEP_SELECTOR = torch.tensor(contract.SUBMISSION_FUTURE_INDICES)


def agent_frame_to_world_frame(
        agent_frame_positions: np.ndarray, target: dict[str, Any],
        scenario_array: dict[str, np.ndarray]) -> np.ndarray:
    """Converts a prediction from the agent's own frame to world coordinates via
    the scene frame stored on the scenario.
    """
    scene_frame_positions = frame_ops.positions_from_frame(
        agent_frame_positions,
        target["frame_origin"],
        target["frame_heading"],
    )
    return frame_ops.positions_from_frame(
        scene_frame_positions,
        scenario_array["frame_origin"],
        scenario_array["frame_heading"],
    )


def designated_target_scenes(
    staged_directory: Path | str
) -> Iterator[tuple[dict[str, np.ndarray], dict[str, Any]]]:
    """Yields (scenario_array, scene_sample) for each staged scenario, in file
    order, holding only the targets Waymo designated for scoring.
    """
    for scenario_path in sorted(Path(staged_directory).glob("*.npz")):
        scenario_array = loader.read_scenario(scenario_path)
        designated_count = int(scenario_array["is_designated_target"].sum())
        track_indices = loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            designated_targets_only=True,
        )
        assert len(track_indices) == designated_count, (
            f"{scenario_path} designates {designated_count} targets"
            f" but {designated_count - len(track_indices)} of them"
            f" cannot be predicted")
        yield scenario_array, loader.build_scene_sample(scenario_array,
                                                        track_indices.tolist())


def predict_for_submission(
        predictor: model.MotionPredictor | None,
        scene_sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """One scene in, and its 54 modes at 10 Hz become Waymo's format of the 6
    kept modes at 2 Hz. predictor=None runs the constant-velocity null.
    """
    batch = pipeline.torch_batch(loader.build_scene_batch([scene_sample]))
    with torch.no_grad():
        if predictor is None:
            trajectories, confidence_logits = baseline.constant_velocity(
                batch.agent_history)
        else:
            trajectories, confidence_logits = predictor(batch)
    pruned_trajectories, _ = pruning.prune_modes_batched(
        trajectories, confidence_logits)
    confidences = pruning.aggregated_confidences(trajectories,
                                                 confidence_logits,
                                                 pruned_trajectories)
    decimated = pruned_trajectories.index_select(2, SUBMISSION_STEP_SELECTOR)
    return decimated.numpy(), confidences.numpy()


def load_predictor(checkpoint_path: Path | str,
                   anchors_path: Path | str) -> model.MotionPredictor:
    """Builds the model from its anchor file and loads trained weights, set to
    eval mode.
    """
    predictor = model.MotionPredictor(load_anchor_file(anchors_path))
    predictor.load_state_dict(
        load_checkpoint_state(checkpoint_path)["model_state"])
    return predictor.eval()


def write_submission_arrays(predictor: model.MotionPredictor | None,
                            staged_directory: Path | str,
                            output_path: Path | str) -> int:
    """Predicts the designated targets one scene at a time, converts to world
    frame, and writes a compressed submission .npz with provenance.
    """
    scenario_ids, track_ids, world_trajectories, confidences = (
        [],
        [],
        [],
        [],
    )
    for scenario_array, scene_sample in designated_target_scenes(
            staged_directory):
        scene_trajectories, scene_confidences = predict_for_submission(
            predictor, scene_sample)
        for target, trajectories, target_confidences in zip(
                scene_sample["targets"], scene_trajectories, scene_confidences):
            scenario_ids.append(str(scene_sample["scenario_id"]))
            track_ids.append(int(target["track_id"]))
            world_trajectories.append(
                agent_frame_to_world_frame(trajectories, target,
                                           scenario_array))
            confidences.append(target_confidences)

    stacked_trajectories = np.stack(world_trajectories)
    assert stacked_trajectories.shape[1:] == (
        contract.NUM_PREDICTED_MODES,
        contract.SUBMISSION_STEPS,
        2,
    ), (f"submission trajectories have shape"
        f" {stacked_trajectories.shape}")
    submission_arrays = {
        "scenario_id": np.array(scenario_ids),
        "track_id": np.array(track_ids, dtype=np.int64),
        "world_trajectories": stacked_trajectories,
        "confidences": np.stack(confidences).astype(np.float64),
        "provenance": contract.artifact_provenance("submit.py",
                                                   staged_directory),
    }
    np.savez_compressed(output_path, **submission_arrays)
    return len(track_ids)


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
                "usage: submit.py --constant-velocity STAGED OUTPUT")
        staged_directory, output_path = arguments.paths
        predictor = None
    else:
        if len(arguments.paths) != 4:
            raise SystemExit(
                "usage: submit.py CHECKPOINT STAGED ANCHORS OUTPUT")
        (
            checkpoint_path,
            staged_directory,
            anchors_path,
            output_path,
        ) = arguments.paths
        predictor = load_predictor(checkpoint_path, anchors_path)

    agent_count = write_submission_arrays(predictor, staged_directory,
                                          output_path)
    print(f"{agent_count} designated targets written to {output_path}"
          f" ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
