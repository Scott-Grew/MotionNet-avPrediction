import womd.runtime_env
import argparse
from pathlib import Path

import numpy as np
import torch

from womd import baseline, contract, frame_ops, loader, model, pipeline

BATCH_SIZE = 16
SUBMISSION_STEP_SELECTOR = torch.tensor(contract.SUBMISSION_FUTURE_INDICES)


def agent_frame_to_world_frame(agent_frame_positions, sample, scenario_array):
    storage_frame_positions = frame_ops.positions_to_world_frame(
        agent_frame_positions, sample["frame_origin"], sample["frame_heading"]
    )
    return frame_ops.positions_to_world_frame(
        storage_frame_positions,
        scenario_array["frame_origin"],
        scenario_array["frame_heading"],
    )


def designated_target_samples(staged_directory):
    for scenario_path in sorted(Path(staged_directory).glob("*.npz")):
        scenario_array = loader.read_scenario(scenario_path)
        designated_count = int(scenario_array["is_designated_target"].sum())
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
        for track_index in track_indices:
            yield scenario_array, loader.build_sample(scenario_array, int(track_index))


def grouped(pairs, group_size):
    group = []
    for pair in pairs:
        group.append(pair)
        if len(group) == group_size:
            yield group
            group = []
    if group:
        yield group


def submission_trajectories_and_confidences(predictor, samples):
    batch = pipeline.collate_samples(samples)
    with torch.no_grad():
        if predictor is None:
            trajectories, confidence_logits = baseline.constant_velocity(batch)
        else:
            trajectories, confidence_logits = predictor(batch)
    pruned_trajectories, pruned_logits = model.prune_modes_batched(trajectories, confidence_logits)
    decimated = pruned_trajectories.index_select(2, SUBMISSION_STEP_SELECTOR)
    return decimated.numpy(), torch.softmax(pruned_logits, dim=-1).numpy()


def load_predictor(checkpoint_path, anchors_path):
    unit_anchors, anchor_counts = model.load_anchor_file(anchors_path)
    predictor = model.MotionPredictor(unit_anchors, anchor_counts)
    predictor.load_state_dict(model.load_checkpoint_state(checkpoint_path)["model_state"])
    return predictor.eval()


def write_submission_arrays(predictor, staged_directory, output_path):
    scenario_ids, track_ids, world_trajectories, confidences = [], [], [], []
    for group in grouped(designated_target_samples(staged_directory), BATCH_SIZE):
        samples = [sample for _, sample in group]
        group_trajectories, group_confidences = submission_trajectories_and_confidences(
            predictor, samples
        )
        for (scenario_array, sample), trajectories, sample_confidences in zip(
            group, group_trajectories, group_confidences
        ):
            scenario_ids.append(str(sample["scenario_id"]))
            track_ids.append(int(sample["track_id"]))
            world_trajectories.append(
                agent_frame_to_world_frame(trajectories, sample, scenario_array)
            )
            confidences.append(sample_confidences)

    stacked_trajectories = np.stack(world_trajectories)
    assert stacked_trajectories.shape[1:] == (
        contract.NUM_PREDICTED_MODES, contract.SUBMISSION_STEPS, 2
    ), f"submission trajectories have shape {stacked_trajectories.shape}"
    np.savez_compressed(
        output_path,
        scenario_id=np.array(scenario_ids),
        track_id=np.array(track_ids, dtype=np.int64),
        world_trajectories=stacked_trajectories,
        confidences=np.stack(confidences).astype(np.float64),
        provenance=contract.artifact_provenance("submit.py", staged_directory),
    )
    return len(track_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constant-velocity", action="store_true")
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    if arguments.constant_velocity:
        if len(arguments.paths) != 2:
            raise SystemExit("usage: submit.py --constant-velocity STAGED OUTPUT")
        staged_directory, output_path = arguments.paths
        predictor = None
    else:
        if len(arguments.paths) != 4:
            raise SystemExit("usage: submit.py CHECKPOINT STAGED ANCHORS OUTPUT")
        checkpoint_path, staged_directory, anchors_path, output_path = arguments.paths
        predictor = load_predictor(checkpoint_path, anchors_path)

    agent_count = write_submission_arrays(predictor, staged_directory, output_path)
    print(
        f"{agent_count} designated targets written to {output_path}"
        f" ({output_path.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
