"""A few training batches as a smoke test that stops on non-finite losses or
on the modes all collapsing to one endpoint.
"""
from __future__ import annotations

import womd.runtime_env
import argparse
import math
import os
from pathlib import Path

import torch

import train
from womd import baseline, metrics, pipeline
from womd.checkpoint import load_anchor_file
from womd.loader import SceneBatch
from womd.model import QUERY_COUNT, MotionPredictor


def first_scenario_paths(staged_directory: Path, needed: int) -> list[Path]:
    """Returns up to needed .npz scenario paths from the directory, in whatever
    order the filesystem yields them.
    """
    paths = []
    with os.scandir(staged_directory) as entries:
        for entry in entries:
            if entry.name.endswith(".npz"):
                paths.append(Path(entry.path))
                if len(paths) >= needed:
                    break
    return paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("anchors_path", type=Path)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def check_step_is_healthy(step: train.TrainingStep, batch_index: int) -> None:
    """Stops the run if a loss, prediction or logit is not finite, or if the
    modes have all landed on one endpoint.
    """
    for name, tensor in (
        ("total", step.total),
        ("trajectories", step.trajectories),
        ("logits", step.confidence_logits),
    ):
        if not tensor.isfinite().all():
            raise SystemExit(f"non-finite {name} at batch {batch_index}")
    # Compares each mode's final step to the first mode's; a match
    # across all of them means the modes never diverged.
    endpoints = step.trajectories.detach()[:, :, -1]
    if torch.allclose(endpoints[:, :1], endpoints, atol=1e-4):
        raise SystemExit(
            f"all modes collapsed to one point at batch {batch_index}")


def median_end_distances(
        step: train.TrainingStep,
        batch: SceneBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Median distance from the start to the final step, in metres, for the
    predictions, the logged futures and the constant-velocity null.
    """
    scoreable = batch.future_mask.any(dim=-1)
    predicted_ends = step.trajectories.detach()[scoreable, :, -1]
    logged_ends = batch.future_positions[scoreable, -1]
    null_trajectories, _ = baseline.constant_velocity(batch.agent_history)
    null_ends = null_trajectories[scoreable, 0, -1]
    return (
        predicted_ends.norm(dim=-1).median(),
        logged_ends.norm(dim=-1).median(),
        null_ends.norm(dim=-1).median(),
    )


def main() -> None:
    """Runs a few training batches and stops at non-finite losses or gradients,
    or at the modes all collapsing to one endpoint.
    """
    arguments = parse_arguments()

    torch.manual_seed(arguments.seed)
    predictor = MotionPredictor(load_anchor_file(arguments.anchors_path))
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor),
                                  lr=train.LEARNING_RATE)
    scenario_paths = first_scenario_paths(
        arguments.staged_directory, arguments.batches * arguments.batch_size)
    if len(scenario_paths) < arguments.batch_size:
        raise SystemExit(
            f"{arguments.staged_directory} has fewer than one batch"
            f" of scenarios")
    batches = pipeline.batches(
        scenario_paths,
        worker_count=0,
        batch_size=arguments.batch_size,
        prefetch_batches=0,
        seed=arguments.seed,
        designated_targets_only=True,
    )

    winner_counts = torch.zeros(QUERY_COUNT, dtype=torch.long)
    end_distances = []
    for batch_index, batch in enumerate(batches):
        if batch_index >= arguments.batches:
            break
        step = train.training_losses(predictor, batch)
        check_step_is_healthy(step, batch_index)
        optimizer.zero_grad()
        step.total.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(predictor.parameters(),
                                           arguments.gradient_clip_norm))
        if not math.isfinite(gradient_norm):
            raise SystemExit(f"non-finite gradient norm at batch {batch_index}")
        optimizer.step()

        mode_distances = metrics.mean_distance_per_mode(
            step.trajectories.detach(), batch.future_positions,
            batch.future_mask)
        winners = mode_distances.argmin(dim=1)
        winner_counts.scatter_add_(0, winners, torch.ones_like(winners))
        end_distances.append(median_end_distances(step, batch))
        print(
            f"batch {batch_index} loss {float(step.total):.4f}"
            f" reg {float(step.regression):.4f}"
            f" cls {float(step.classification):.4f}"
            f" grad {gradient_norm:.1f}",
            flush=True,
        )

    if not end_distances:
        raise SystemExit("no batches ran")
    never_win = int((winner_counts == 0).sum())
    predicted, logged, null = (
        float(torch.stack(column).median()) for column in zip(*end_distances))
    print(
        f"never-win {never_win}/{QUERY_COUNT} median |end|"
        f" pred {predicted:.2f} m logged {logged:.2f} m"
        f" cv {null:.2f} m",
        flush=True,
    )
    if never_win == QUERY_COUNT:
        raise SystemExit("no mode ever won: assignment is dead")


if __name__ == "__main__":
    main()
