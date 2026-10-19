"""A few training batches as a smoke test that stops on non-finite losses or
on the modes all collapsing to one endpoint.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import math
import os
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

import train
from womd import baseline, metrics, pipeline
from womd.checkpoint import load_anchor_file
from womd.loader import SceneBatch
from womd.loss import LossTerms
from womd.metrics import TrainingStep
from womd.model import QUERY_COUNT, MotionPredictor
from womd.pipeline import BatchSettings

# Mode endpoints closer than this to the first mode's all count as one
# point.
COLLAPSE_TOLERANCE_METRES = 1e-4


class EndDistances(NamedTuple):
    """Median distance from the start to the final step, in metres, for the
    predictions, the logged futures and the constant-velocity null.
    """

    predicted: Tensor
    logged: Tensor
    null: Tensor


def first_scenario_paths(staged_directory: Path, needed: int) -> list[Path]:
    """Up to needed .npz scenario paths from the directory, in whatever order
    the filesystem yields them.
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


def check_health(step: TrainingStep, batch_index: int) -> None:
    """Stops the run if a loss, prediction or logit is not finite, or if the
    modes have all landed on one endpoint.
    """
    predicted = step.predicted
    named = (
        ("total", step.losses.total),
        ("trajectories", predicted.trajectories),
        ("logits", predicted.confidence_logits),
    )
    for name, tensor in named:
        if not tensor.isfinite().all():
            raise SystemExit(f"non-finite {name} at batch {batch_index}")
    # Compares each mode's final step to the first mode's; a match
    # across all of them means the modes never diverged.
    endpoints = predicted.trajectories.detach()[:, :, -1]
    if torch.allclose(
        endpoints[:, :1], endpoints, atol=COLLAPSE_TOLERANCE_METRES
    ):
        raise SystemExit(
            f"all modes collapsed to one point at batch {batch_index}"
        )


def end_distances(step: TrainingStep, batch: SceneBatch) -> EndDistances:
    targets = batch.targets
    scoreable = targets.future_mask.any(dim=-1)
    predicted_ends = step.predicted.trajectories.detach()[scoreable, :, -1]
    logged_ends = targets.future_positions[scoreable, -1]
    null_trajectories, _ = baseline.constant_velocity(targets.agent_history)
    null_ends = null_trajectories[scoreable, 0, -1]
    return EndDistances(
        predicted_ends.norm(dim=-1).median(),
        logged_ends.norm(dim=-1).median(),
        null_ends.norm(dim=-1).median(),
    )


def print_batch(index: int, losses: LossTerms, gradient_norm: float) -> None:
    print(
        f"batch {index} loss {float(losses.total):.4f}"
        f" reg {float(losses.regression):.4f}"
        f" cls {float(losses.classification):.4f}"
        f" grad {gradient_norm:.1f}",
        flush=True,
    )


def print_summary(winner_counts: Tensor, distances: list[EndDistances]) -> None:
    """The anchors that never won a batch and the median end distances over
    the run; a run where no mode ever won has a dead assignment.
    """
    never_win = int((winner_counts == 0).sum())
    columns = zip(*distances, strict=True)
    medians = EndDistances(
        *(torch.stack(column).median() for column in columns)
    )
    print(
        f"never-win {never_win}/{QUERY_COUNT} median |end|"
        f" pred {float(medians.predicted):.2f} m"
        f" logged {float(medians.logged):.2f} m"
        f" cv {float(medians.null):.2f} m",
        flush=True,
    )
    if never_win == QUERY_COUNT:
        raise SystemExit("no mode ever won: assignment is dead")


def main() -> None:
    """Runs a few training batches and stops at non-finite losses or gradients,
    or at the modes all collapsing to one endpoint.
    """
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    predictor = MotionPredictor(load_anchor_file(arguments.anchors_path))
    optimizer = torch.optim.AdamW(
        train.parameter_groups(predictor), lr=train.LEARNING_RATE
    )
    needed = arguments.batches * arguments.batch_size
    scenario_paths = first_scenario_paths(arguments.staged_directory, needed)
    if len(scenario_paths) < arguments.batch_size:
        raise SystemExit(
            f"{arguments.staged_directory} has fewer than one batch"
            f" of scenarios"
        )
    settings = BatchSettings(
        batch_size=arguments.batch_size,
        worker_count=0,
        prefetch_batches=0,
        seed=arguments.seed,
        designated_only=True,
    )
    batches = pipeline.batches(scenario_paths, settings)

    winner_counts = torch.zeros(QUERY_COUNT, dtype=torch.long)
    distances = []
    for batch_index, batch in enumerate(batches):
        if batch_index >= arguments.batches:
            break
        step = train.training_step(predictor, batch)
        check_health(step, batch_index)
        optimizer.zero_grad()
        step.losses.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), arguments.gradient_clip_norm
        )
        gradient_norm = float(norm)
        if not math.isfinite(gradient_norm):
            raise SystemExit(f"non-finite gradient norm at batch {batch_index}")
        optimizer.step()

        trajectories = step.predicted.trajectories.detach()
        winners = metrics.mode_distances(trajectories, batch.targets).argmin(
            dim=1
        )
        winner_counts.scatter_add_(0, winners, torch.ones_like(winners))
        distances.append(end_distances(step, batch))
        print_batch(batch_index, step.losses, gradient_norm)

    if not distances:
        raise SystemExit("no batches ran")
    print_summary(winner_counts, distances)


if __name__ == "__main__":
    main()
