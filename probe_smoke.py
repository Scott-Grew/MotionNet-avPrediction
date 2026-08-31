import womd.runtime_env
import argparse
import math
import os
from pathlib import Path

import torch

import train
from womd import baseline, metrics, model, pipeline
from womd.model import QUERY_COUNT, MotionPredictor


def first_scenario_paths(staged_directory, needed):
    paths = []
    with os.scandir(staged_directory) as entries:
        for entry in entries:
            if entry.name.endswith(".npz"):
                paths.append(Path(entry.path))
                if len(paths) >= needed:
                    break
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("anchors_path", type=Path)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-clip-norm", type=float, required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    predictor = MotionPredictor(
        model.load_anchor_file(arguments.anchors_path)
    )
    optimizer = torch.optim.AdamW(
        train.parameter_groups(predictor), lr=train.LEARNING_RATE
    )
    scenario_paths = first_scenario_paths(
        arguments.staged_directory,
        arguments.batches * arguments.batch_size,
    )
    if len(scenario_paths) < arguments.batch_size:
        raise SystemExit(
            f"{arguments.staged_directory} has fewer than one batch of scenarios"
        )

    winner_counts = torch.zeros(QUERY_COUNT, dtype=torch.long)
    predicted_end_norms, logged_end_norms, null_end_norms = [], [], []
    for batch_index, batch in enumerate(
        pipeline.batches(
            scenario_paths,
            0,
            arguments.batch_size,
            0,
            arguments.seed,
            True,
        )
    ):
        if batch_index >= arguments.batches:
            break
        (
            total,
            regression,
            classification,
            trajectories,
            confidence_logits,
        ) = train.training_losses(predictor, batch)
        for name, tensor in (
            ("total", total),
            ("trajectories", trajectories),
            ("logits", confidence_logits),
        ):
            if not tensor.isfinite().all():
                raise SystemExit(
                    f"non-finite {name} at batch {batch_index}"
                )
        optimizer.zero_grad()
        total.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), arguments.gradient_clip_norm
            )
        )
        if not math.isfinite(gradient_norm):
            raise SystemExit(
                f"non-finite gradient norm at batch {batch_index}"
            )
        endpoints = trajectories.detach()[:, :, -1]
        if torch.allclose(endpoints[:, :1], endpoints, atol=1e-4):
            raise SystemExit(
                f"all modes collapsed to one point at batch {batch_index}"
            )
        optimizer.step()

        scoreable = batch["future_mask"].any(dim=-1)
        winners = metrics.mean_distance_per_mode(
            trajectories.detach(),
            batch["future_positions"],
            batch["future_mask"],
        ).argmin(dim=1)
        winner_counts.scatter_add_(
            0, winners, torch.ones_like(winners)
        )
        predicted_end_norms.append(
            trajectories.detach()[scoreable, :, -1]
            .norm(dim=-1)
            .median()
        )
        logged_end_norms.append(
            batch["future_positions"][scoreable, -1]
            .norm(dim=-1)
            .median()
        )
        null_trajectories, _ = baseline.constant_velocity(batch)
        null_end_norms.append(
            null_trajectories[scoreable, 0, -1].norm(dim=-1).median()
        )
        print(
            f"batch {batch_index} loss {float(total):.4f} reg {float(regression):.4f}"
            f" cls {float(classification):.4f} grad {gradient_norm:.1f}",
            flush=True,
        )

    if not predicted_end_norms:
        raise SystemExit("no batches ran")
    never_win = int((winner_counts == 0).sum())
    predicted = float(torch.stack(predicted_end_norms).median())
    logged = float(torch.stack(logged_end_norms).median())
    null = float(torch.stack(null_end_norms).median())
    print(
        f"never-win {never_win}/{QUERY_COUNT} median |end| pred {predicted:.2f} m"
        f" logged {logged:.2f} m cv {null:.2f} m",
        flush=True,
    )
    if never_win == QUERY_COUNT:
        raise SystemExit("every mode never won: assignment is dead")


if __name__ == "__main__":
    main()
