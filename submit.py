import womd.runtime_env
import argparse
from pathlib import Path

import numpy as np
import torch

from womd import (
    baseline,
    contract,
    frame_ops,
    loader,
    model,
)

SUBMISSION_STEP_SELECTOR = torch.tensor(
    contract.SUBMISSION_FUTURE_INDICES
)


# Converts a prediction from the agent's own frame to world
# coordinates via the scene frame stored on the sample.
def agent_frame_to_world_frame(
    agent_frame_positions, sample, scenario_array
):
    storage_frame_positions = frame_ops.positions_to_world_frame(
        agent_frame_positions,
        sample["frame_origin"],
        sample["frame_heading"],
    )
    return frame_ops.positions_to_world_frame(
        storage_frame_positions,
        scenario_array["frame_origin"],
        scenario_array["frame_heading"],
    )


# Yields one scene at a time, each built from only the targets
# Waymo designated for scoring in that scenario.
def designated_target_scenes(staged_directory):
    for scenario_path in sorted(Path(staged_directory).glob("*.npz")):
        scenario_array = loader.read_scenario(scenario_path)
        designated_count = int(
            scenario_array["is_designated_target"].sum()
        )
        track_indices = loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        )
        assert len(track_indices) == designated_count, (
            f"{scenario_path} designates {designated_count} targets but"
            f" {designated_count - len(track_indices)} of them cannot be predicted"
        )
        yield scenario_array, loader.build_scene_sample(
            scenario_array, track_indices.tolist()
        )


# Predicts one scene (constant velocity when predictor is None),
# prunes 54 modes to 6, and keeps the 16 submission steps.
def submission_trajectories_and_confidences(predictor, scene_sample):
    batch = {
        name: torch.from_numpy(array)
        for name, array in loader.build_scene_batch(
            [scene_sample]
        ).items()
    }
    with torch.no_grad():
        if predictor is None:
            trajectories, confidence_logits = (
                baseline.constant_velocity(batch)
            )
        else:
            trajectories, confidence_logits = predictor(batch)
    pruned_trajectories, _ = model.prune_modes_batched(
        trajectories, confidence_logits
    )
    confidences = model.aggregated_confidences(
        trajectories, confidence_logits, pruned_trajectories
    )
    decimated = pruned_trajectories.index_select(
        2, SUBMISSION_STEP_SELECTOR
    )
    return decimated.numpy(), confidences.numpy()


# Builds the model from its anchor file and loads trained weights,
# set to eval mode.
def load_predictor(checkpoint_path, anchors_path):
    predictor = model.MotionPredictor(
        model.load_anchor_file(anchors_path)
    )
    predictor.load_state_dict(
        model.load_checkpoint_state(checkpoint_path)["model_state"]
    )
    return predictor.eval()


# Predicts every designated target, converts trajectories to world
# coordinates, writes the .npz, and returns the target count.
def write_submission_arrays(predictor, staged_directory, output_path):
    scenario_ids, track_ids, world_trajectories, confidences = (
        [],
        [],
        [],
        [],
    )
    for scenario_array, scene_sample in designated_target_scenes(
        staged_directory
    ):
        scene_trajectories, scene_confidences = (
            submission_trajectories_and_confidences(
                predictor, scene_sample
            )
        )
        for target, trajectories, target_confidences in zip(
            scene_sample["targets"],
            scene_trajectories,
            scene_confidences,
        ):
            scenario_ids.append(str(scene_sample["scenario_id"]))
            track_ids.append(int(target["track_id"]))
            world_trajectories.append(
                agent_frame_to_world_frame(
                    trajectories, target, scenario_array
                )
            )
            confidences.append(target_confidences)

    stacked_trajectories = np.stack(world_trajectories)
    assert stacked_trajectories.shape[1:] == (
        contract.NUM_PREDICTED_MODES,
        contract.SUBMISSION_STEPS,
        2,
    ), f"submission trajectories have shape {stacked_trajectories.shape}"
    np.savez_compressed(
        output_path,
        scenario_id=np.array(scenario_ids),
        track_id=np.array(track_ids, dtype=np.int64),
        world_trajectories=stacked_trajectories,
        confidences=np.stack(confidences).astype(np.float64),
        provenance=contract.artifact_provenance(
            "submit.py", staged_directory
        ),
    )
    return len(track_ids)


# Writes a submission .npz from a trained model or from the
# constant-velocity baseline.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constant-velocity", action="store_true")
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    if arguments.constant_velocity:
        if len(arguments.paths) != 2:
            raise SystemExit(
                "usage: submit.py --constant-velocity STAGED OUTPUT"
            )
        staged_directory, output_path = arguments.paths
        predictor = None
    else:
        if len(arguments.paths) != 4:
            raise SystemExit(
                "usage: submit.py CHECKPOINT STAGED ANCHORS OUTPUT"
            )
        (
            checkpoint_path,
            staged_directory,
            anchors_path,
            output_path,
        ) = arguments.paths
        predictor = load_predictor(checkpoint_path, anchors_path)

    agent_count = write_submission_arrays(
        predictor, staged_directory, output_path
    )
    print(
        f"{agent_count} designated targets written to {output_path}"
        f" ({output_path.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
