"""Step 3 of 5.

Trains the model on staged scenarios; resumable, with a time budget so a session
stops cleanly. Runs as one process, or as one process per GPU under torchrun.
"""
from __future__ import annotations

import womd.runtime_env
import argparse
from collections import namedtuple
import contextlib
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as distributed
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from womd import loader, loss, pipeline
from womd.checkpoint import (
    checkpoint_state,
    load_anchor_file,
    load_checkpoint_state,
    save_checkpoint,
)
from womd.loader import SceneBatch
from womd.metrics import (
    LOG_EVERY_BATCHES,
    EpochProgress,
    TrainingStep,
    epoch_scalars,
)
from womd.model import MotionPredictor

# The peak learning rate unless --learning-rate sets another, and
# AdamW's own default weight decay.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

# Where this process sits among the processes torchrun started; a
# plain launch is one process of one.
Processes = namedtuple("Processes", "count rank local_rank is_main")

GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)


def parameter_groups(predictor: torch.nn.Module) -> list[dict[str, Any]]:
    """Splits parameters so weight decay applies to matrices only, not to
    biases, norms or the learned anchor queries.
    """
    decayed = []
    undecayed = []
    for name, parameter in predictor.named_parameters():
        if parameter.ndim >= 2 and not name.endswith("queries"):
            decayed.append(parameter)
        else:
            undecayed.append(parameter)
    return [
        {
            "params": decayed,
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params": undecayed,
            "weight_decay": 0.0
        },
    ]


def optimiser_steps_per_epoch(scenario_paths: list[Path], worker_count: int,
                              batch_size: int,
                              designated_targets_only: bool) -> int:
    """Steps per epoch, the sum over worker streams of ceil(targets / batch
    size), because each worker fills its own batches.
    """
    stream_count = max(worker_count, 1)
    step_count = 0
    for stream_index in range(stream_count):
        stream_sample_count = 0
        for scenario_path in scenario_paths[stream_index::stream_count]:
            with np.load(scenario_path) as scenario_file:
                stream_sample_count += len(
                    loader.eligible_track_indices(
                        scenario_file["track_rows"],
                        scenario_file["track_valid"],
                        scenario_file["is_designated_target"],
                        designated_targets_only=designated_targets_only,
                    ))
        step_count += math.ceil(stream_sample_count / batch_size)
    return step_count


def scheduled_learning_rate(process_steps: int,
                            *,
                            warmup_steps: int,
                            learning_rate: float = LEARNING_RATE,
                            decay_start_step: int | None = None,
                            decay_end_step: int | None = None) -> float:
    """Learning rate as a pure function of the global step, a linear warm-up,
    a hold, then an optional cosine fall to zero.
    """
    if process_steps < warmup_steps:
        return learning_rate * (process_steps + 1) / warmup_steps
    if decay_start_step is None or process_steps < decay_start_step:
        return learning_rate
    steps_into_decay = process_steps - decay_start_step
    decay_length = max(decay_end_step - 1 - decay_start_step, 1)
    decay_fraction = min(steps_into_decay / decay_length, 1.0)
    # The rate is peak * (1 + cos(pi f)) / 2 for fraction f through the fall.
    cosine_term = 1.0 + math.cos(math.pi * decay_fraction)
    return learning_rate * 0.5 * cosine_term


def training_losses(predictor: torch.nn.Module,
                    batch: SceneBatch) -> TrainingStep:
    """Runs the predictor on a batch and returns its losses plus its raw
    trajectories and confidence logits.
    """
    predictions = predictor(batch, with_likelihood_outputs=True)
    total, regression, classification = loss.prediction_loss(
        predictions.trajectories,
        predictions.log_standard_deviation,
        predictions.confidence_logits,
        batch.targets.future_positions,
        batch.targets.future_mask,
        predictions.anchors,
    )
    return TrainingStep(
        total,
        regression,
        classification,
        predictions.trajectories,
        predictions.confidence_logits,
    )


def agreed_with_main_process(decision: bool, device: torch.device) -> bool:
    """Process 0's decision, shared with every process so they all stop
    together. Every process must make this call or the broadcast hangs.
    """
    if not distributed.is_initialized():
        return decision
    shared_decision = torch.tensor(int(decision), device=device)
    distributed.broadcast(shared_decision, src=0)
    return bool(shared_decision.item())


def epochs_left_to_train(completed_epochs: int, requested_epochs: int) -> range:
    return range(completed_epochs, requested_epochs)


def report_scalars(summary_writer: SummaryWriter, heading: str,
                   scalars: dict[str, float], global_step: int) -> None:
    """Writes one table of scalars to TensorBoard and prints the same table as
    the log line, so the console and the board never drift.
    """
    for name, value in scalars.items():
        summary_writer.add_scalar(name, value, global_step)
    printed_scalars = [f"{name} {value:.4g}" for name, value in scalars.items()]
    print(" | ".join([heading] + printed_scalars), flush=True)


# The settings of a run that stay fixed from its first batch to its
# last. Only the main process holds a summary writer.
RunSettings = namedtuple(
    "RunSettings",
    "checkpoint_path previous_checkpoint_path checkpoint_every_seconds seed"
    " warmup_steps gradient_clip_norm learning_rate decay_start_step"
    " decay_end_step summary_writer is_main_process epoch_plan",
)


def optimisation_step(predictor: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      gradient_scaler: GradScaler, total: torch.Tensor,
                      step_learning_rate: float,
                      gradient_clip_norm: float) -> tuple[bool, bool]:
    """Sets this step's rate, then backward pass, gradient clip and optimiser
    step. Returns whether it clipped and whether it skipped.
    """
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = step_learning_rate
    optimizer.zero_grad()
    gradient_scaler.scale(total).backward()
    gradient_scaler.unscale_(optimizer)
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(predictor.parameters(),
                                       gradient_clip_norm))
    gradient_scaler.step(optimizer)
    # The scaler lowers its scale only when it skipped a step over
    # non-finite gradients, which is how skips are detected.
    scale_before_update = gradient_scaler.get_scale()
    gradient_scaler.update()
    was_skipped = gradient_scaler.get_scale() < scale_before_update
    return gradient_norm > gradient_clip_norm, was_skipped


def train_epoch(
    predictor: torch.nn.Module, optimizer: torch.optim.Optimizer,
    gradient_scaler: GradScaler, batches: Iterable[SceneBatch],
    device: torch.device, settings: RunSettings, epoch_index: int,
    steps_before_epoch: int
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """One pass over the data.

    Each batch goes through forward and loss, the optimiser step and the
    training monitor, then a log line and a timed checkpoint. Only the main
    process logs and checkpoints.
    With several processes the caller wraps this in Join, so a process whose
    shard runs out first does not hang the others.
    """
    progress = EpochProgress(device)
    wait_start = time.perf_counter()
    checkpoint_wait_start = time.perf_counter()
    for batch in batches:
        progress.seconds["data_wait"] += time.perf_counter() - wait_start

        step_start = time.perf_counter()
        batch = batch.each(lambda tensor: tensor.to(device, non_blocking=True))
        with torch.amp.autocast(device_type=device.type,
                                enabled=gradient_scaler.is_enabled()):
            step = training_losses(predictor, batch)
        # The rate is a function of the step count alone, so a
        # resumed run needs no scheduler state.
        step_learning_rate = scheduled_learning_rate(
            steps_before_epoch + progress.batch_count,
            warmup_steps=settings.warmup_steps,
            learning_rate=settings.learning_rate,
            decay_start_step=settings.decay_start_step,
            decay_end_step=settings.decay_end_step,
        )
        was_clipped, was_skipped = optimisation_step(
            predictor, optimizer, gradient_scaler, step.total,
            step_learning_rate, settings.gradient_clip_norm)
        total_is_finite = progress.record_step(
            step, was_clipped, was_skipped,
            batch.targets.agent_history.shape[0])
        progress.seconds["step"] += time.perf_counter() - step_start

        monitor_start = time.perf_counter()
        progress.record_monitor(step, batch)
        progress.seconds["monitor"] += time.perf_counter() - monitor_start

        if progress.batch_count % LOG_EVERY_BATCHES == 0:
            if settings.is_main_process:
                report_scalars(
                    settings.summary_writer,
                    f"  batch {progress.batch_count}",
                    progress.window_scalars(step_learning_rate),
                    steps_before_epoch + progress.batch_count,
                )
            progress.start_new_window()

        # Timed checkpoint, skipped while the loss is non-finite.
        seconds_since_checkpoint = time.perf_counter() - checkpoint_wait_start
        checkpoint_is_due = (seconds_since_checkpoint
                             >= settings.checkpoint_every_seconds)
        if settings.is_main_process and checkpoint_is_due:
            if total_is_finite:
                save_checkpoint(
                    settings.checkpoint_path,
                    settings.previous_checkpoint_path,
                    checkpoint_state(predictor,
                                     optimizer,
                                     gradient_scaler,
                                     seed=settings.seed,
                                     completed_epochs=epoch_index,
                                     epoch_plan=settings.epoch_plan),
                )
            checkpoint_wait_start = time.perf_counter()
        wait_start = time.perf_counter()
    return progress.averages(), progress.accumulator.results(), progress.seconds


def parse_arguments() -> argparse.Namespace:
    """The command line.

    Everything that shapes a run is required, so a launch script states its
    recipe in full.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--checkpoint-every-seconds", type=int, required=True)
    parser.add_argument("--stop-after-seconds", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--decay-from-epoch", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--all-eligible-agents", action="store_true")
    return parser.parse_args()


def start_processes() -> tuple[Processes, torch.device]:
    """Reads the process layout torchrun sets in the environment, picks this
    process's device, and joins the process group when there are several.
    """
    count = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    if count > 1:
        distributed.init_process_group("nccl" if device.type ==
                                       "cuda" else "gloo")
    return Processes(count, rank, local_rank, rank == 0), device


def stopped_by_budget(should_stop: bool, processes: Processes,
                      device: torch.device, message: str) -> bool:
    """Whether the main process's stop decision holds, announcing message
    when it does. Every process must make this call together.
    """
    if not agreed_with_main_process(should_stop, device):
        return False
    announce(processes, message)
    return True


def announce(processes: Processes, message: str) -> None:
    """Prints from the main process only."""
    if processes.is_main:
        print(message, flush=True)


def resume_training(arguments: argparse.Namespace, predictor: MotionPredictor,
                    optimizer: torch.optim.Optimizer,
                    gradient_scaler: GradScaler, previous_checkpoint_path: Path,
                    device: torch.device,
                    processes: Processes) -> tuple[int, dict[str, Any] | None]:
    """Loads the checkpoint when --resume is set and returns how many epochs
    are already complete and the epoch plan it was trained under.

    Falls back to the previous checkpoint when the current one is missing,
    which a crash between save_checkpoint's two renames leaves behind.
    """
    if not arguments.resume:
        return 0, None
    resume_path = arguments.checkpoint_path
    if not resume_path.exists() and previous_checkpoint_path.exists():
        resume_path = previous_checkpoint_path
        announce(
            processes,
            f"{arguments.checkpoint_path} missing, resuming from {resume_path}")
    if not resume_path.exists():
        return 0, None

    checkpoint = load_checkpoint_state(resume_path, map_location=device)
    predictor.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "gradient_scaler_state" in checkpoint:
        gradient_scaler.load_state_dict(checkpoint["gradient_scaler_state"])
    completed_epochs = checkpoint["completed_epochs"]
    announce(processes,
             f"resuming {resume_path}: {completed_epochs} epochs complete")
    return completed_epochs, checkpoint.get("epoch_plan")


def decay_window(arguments: argparse.Namespace,
                 steps_per_epoch: int) -> tuple[int | None, int | None]:
    """The steps at which the cosine fall starts and ends, or (None, None) when
    --decay-from-epoch is not given.
    """
    if arguments.decay_from_epoch is None:
        return None, None
    decay_start_step = (arguments.decay_from_epoch - 1) * steps_per_epoch
    decay_end_step = arguments.epochs * steps_per_epoch
    return decay_start_step, decay_end_step


def build_training_objects(
    arguments: argparse.Namespace, device: torch.device
) -> tuple[MotionPredictor, torch.optim.Optimizer, GradScaler]:
    """Builds the model from its anchor file, its optimiser, and the gradient
    scaler that mixed precision needs.
    """
    predictor = MotionPredictor(load_anchor_file(arguments.anchors)).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(predictor),
                                  lr=arguments.learning_rate)
    gradient_scaler = GradScaler(
        enabled=arguments.mixed_precision and device.type == "cuda")
    return predictor, optimizer, gradient_scaler


def plan_epochs(
        arguments: argparse.Namespace, processes: Processes,
        device: torch.device, stored_plan: dict[str, Any] | None
) -> tuple[list[Path], dict[str, Any]]:
    """Finds this process's share of the staged scenarios and the optimiser
    steps in one epoch, which the learning-rate schedule is measured in.

    Counting opens every staged file, so the count is saved in the checkpoint
    and reused while the scenarios and the launch layout are unchanged.
    """
    every_scenario_path = sorted(arguments.staged_directory.glob("*.npz"))
    scenario_paths = every_scenario_path[processes.rank::processes.count]
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"
    epoch_plan = {
        "scenario_count": len(every_scenario_path),
        "process_count": processes.count,
        "worker_count": arguments.workers,
        "per_process_batch_size": arguments.batch_size // processes.count,
        "designated_targets_only": not arguments.all_eligible_agents,
    }
    if stored_plan is not None and all(
            stored_plan.get(name) == value
            for name, value in epoch_plan.items()):
        epoch_plan["steps_per_epoch"] = stored_plan["steps_per_epoch"]
        announce(processes, "optimiser step count read from the checkpoint")
        return scenario_paths, epoch_plan
    steps_per_epoch = optimiser_steps_per_epoch(
        scenario_paths, arguments.workers,
        arguments.batch_size // processes.count,
        not arguments.all_eligible_agents)
    if processes.count > 1:
        # Every process uses the longest share's step count, so the
        # schedule is the same everywhere.
        longest_process_steps = torch.tensor(steps_per_epoch, device=device)
        distributed.all_reduce(longest_process_steps,
                               op=distributed.ReduceOp.MAX)
        steps_per_epoch = int(longest_process_steps.item())
    announce(
        processes, f"{steps_per_epoch} optimiser steps per epoch,"
        f" {steps_per_epoch * arguments.epochs} over"
        f" {arguments.epochs} epochs, learning rate"
        f" {arguments.learning_rate} held after"
        f" {arguments.warmup_steps} warmup steps")
    epoch_plan["steps_per_epoch"] = steps_per_epoch
    return scenario_paths, epoch_plan


def build_run_settings(arguments: argparse.Namespace,
                       previous_checkpoint_path: Path, epoch_plan: dict[str,
                                                                        Any],
                       processes: Processes) -> RunSettings:
    """Gathers the settings that stay fixed for the whole run, and opens the
    TensorBoard event file beside the checkpoint on the main process.
    """
    decay_start_step, decay_end_step = decay_window(
        arguments, epoch_plan["steps_per_epoch"])
    summary_writer = None
    if processes.is_main:
        summary_writer = SummaryWriter(arguments.checkpoint_path.parent /
                                       "tensorboard")
    return RunSettings(
        checkpoint_path=arguments.checkpoint_path,
        previous_checkpoint_path=previous_checkpoint_path,
        checkpoint_every_seconds=arguments.checkpoint_every_seconds,
        seed=arguments.seed,
        warmup_steps=arguments.warmup_steps,
        gradient_clip_norm=arguments.gradient_clip_norm,
        learning_rate=arguments.learning_rate,
        decay_start_step=decay_start_step,
        decay_end_step=decay_end_step,
        summary_writer=summary_writer,
        is_main_process=processes.is_main,
        epoch_plan=epoch_plan,
    )


def main() -> None:
    """Builds the model, resumes from a checkpoint if asked, and trains the
    remaining epochs inside the time budget.
    """
    arguments = parse_arguments()
    processes, device = start_processes()
    assert arguments.batch_size % processes.count == 0, (
        f"--batch-size {arguments.batch_size} does not split evenly"
        f" over {processes.count} processes")

    torch.manual_seed(arguments.seed)
    predictor, optimizer, gradient_scaler = build_training_objects(
        arguments, device)
    designated_targets_only = not arguments.all_eligible_agents

    previous_checkpoint_path = arguments.checkpoint_path.with_suffix(
        arguments.checkpoint_path.suffix + ".previous")
    completed_epochs, stored_plan = resume_training(arguments, predictor,
                                                    optimizer, gradient_scaler,
                                                    previous_checkpoint_path,
                                                    device, processes)
    remaining_epochs = epochs_left_to_train(completed_epochs, arguments.epochs)
    if not remaining_epochs:
        announce(
            processes, f"NOTHING TO TRAIN: {arguments.checkpoint_path} already"
            f" holds {completed_epochs} completed epochs and"
            f" --epochs is {arguments.epochs}.")
        return

    scenario_paths, epoch_plan = plan_epochs(arguments, processes, device,
                                             stored_plan)
    steps_per_epoch = epoch_plan["steps_per_epoch"]
    settings = build_run_settings(arguments, previous_checkpoint_path,
                                  epoch_plan, processes)
    if arguments.compile:
        predictor = torch.compile(predictor, dynamic=True)
    if processes.count > 1:
        device_ids = [processes.local_rank] if device.type == "cuda" else None
        predictor = DistributedDataParallel(predictor, device_ids=device_ids)
    budget_hours = arguments.stop_after_seconds / 3600

    training_start = time.perf_counter()
    steps_before_epoch = completed_epochs * steps_per_epoch
    last_epoch_seconds = 0.0
    for epoch_index in remaining_epochs:
        # Stop before an epoch that would not fit in the time budget, since
        # Kaggle ends a session without warning.
        elapsed_seconds = time.perf_counter() - training_start
        next_epoch_overruns = (elapsed_seconds + last_epoch_seconds
                               > arguments.stop_after_seconds)
        if stopped_by_budget(
                next_epoch_overruns, processes, device,
                f"STOPPING BEFORE EPOCH {epoch_index + 1}:"
                f" {elapsed_seconds / 3600:.2f} h elapsed, the last epoch"
                f" took {last_epoch_seconds / 3600:.2f} h, and the"
                f" --stop-after-seconds budget is {budget_hours:.2f} h."
                f" {epoch_index} of {arguments.epochs} epochs are complete"
                f" and {arguments.checkpoint_path} holds them."):
            return

        epoch_start = time.perf_counter()
        batches = pipeline.batches(
            scenario_paths,
            worker_count=arguments.workers,
            batch_size=arguments.batch_size // processes.count,
            prefetch_batches=arguments.prefetch,
            seed=arguments.seed + epoch_index,
            designated_targets_only=designated_targets_only,
        )
        uneven_shares = contextlib.nullcontext()
        if processes.count > 1:
            uneven_shares = Join([predictor])
        with uneven_shares:
            averages, monitor, seconds = train_epoch(predictor, optimizer,
                                                     gradient_scaler, batches,
                                                     device, settings,
                                                     epoch_index,
                                                     steps_before_epoch)
        steps_before_epoch += steps_per_epoch
        last_epoch_seconds = time.perf_counter() - epoch_start
        if processes.is_main:
            report_scalars(
                settings.summary_writer,
                f"epoch {epoch_index + 1}/{arguments.epochs}",
                epoch_scalars(averages, monitor, seconds),
                epoch_index + 1,
            )
            settings.summary_writer.flush()

        if not math.isfinite(averages["total"]):
            announce(
                processes, f"epoch {epoch_index + 1} mean total loss"
                f" {averages['total']}, checkpoint left as it was")
        elif processes.is_main:
            save_checkpoint(
                arguments.checkpoint_path,
                previous_checkpoint_path,
                checkpoint_state(predictor,
                                 optimizer,
                                 gradient_scaler,
                                 seed=arguments.seed,
                                 completed_epochs=epoch_index + 1,
                                 epoch_plan=settings.epoch_plan),
            )

        elapsed_seconds = time.perf_counter() - training_start
        budget_is_spent = elapsed_seconds >= arguments.stop_after_seconds
        if stopped_by_budget(
                budget_is_spent, processes, device,
                f"STOPPING EARLY: {elapsed_seconds / 3600:.2f} h"
                f" elapsed against a --stop-after-seconds budget of"
                f" {budget_hours:.2f} h. {epoch_index + 1} of"
                f" {arguments.epochs} epochs are complete and"
                f" {arguments.checkpoint_path} holds them."):
            return


# ------------------------------------------------------------------
# THE LEARNING RATE, a function of the step count alone
#
#   peak |      ______________________
#        |     /                      `.
#        |    /                         `.
#        |   /                            `._
#      0 +--+---------------------------+-----+--> step
#         warm-up         hold          cosine fall, reaching
#                                       zero on the last step
#
#   The fall starts at --decay-from-epoch and is off without it.
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
    if distributed.is_initialized():
        distributed.destroy_process_group()
