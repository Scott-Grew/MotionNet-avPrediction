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

from womd import contract, loader, loss, metrics, pipeline
from womd.checkpoint import (
    load_anchor_file,
    load_checkpoint_state,
    parameter_fingerprint,
)
from womd.model import QUERY_COUNT, MotionPredictor

# One training step's three loss values and the raw predictions
# the training monitor reads.
TrainingStep = namedtuple(
    "TrainingStep",
    "total regression classification trajectories confidence_logits",
)

# The peak learning rate unless --learning-rate sets another, and
# AdamW's own default weight decay.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
# A progress line is printed once per this many batches.
LOG_EVERY_BATCHES = 20

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
    """Steps per epoch: the sum over worker streams of ceil(targets / batch
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
                        designated_targets_only,
                    ))
        step_count += math.ceil(stream_sample_count / batch_size)
    return step_count


def scheduled_learning_rate(process_steps: int,
                            warmup_steps: int,
                            learning_rate: float = LEARNING_RATE,
                            decay_start_step: int | None = None,
                            decay_end_step: int | None = None) -> float:
    """Learning rate as a pure function of the global step: linear warm-up,
    hold, then an optional cosine fall to zero.
    """
    if process_steps < warmup_steps:
        return learning_rate * (process_steps + 1) / warmup_steps
    if decay_start_step is None or process_steps < decay_start_step:
        return learning_rate
    steps_into_decay = process_steps - decay_start_step
    decay_length = max(decay_end_step - 1 - decay_start_step, 1)
    decay_fraction = min(steps_into_decay / decay_length, 1.0)
    cosine_term = 1.0 + math.cos(math.pi * decay_fraction)
    return learning_rate * 0.5 * cosine_term


def training_losses(predictor: torch.nn.Module,
                    batch: dict[str, torch.Tensor]) -> TrainingStep:
    """Runs the predictor on a batch and returns its losses plus its raw
    trajectories and confidence logits.
    """
    predictions = predictor(batch, with_likelihood_outputs=True)
    total, regression, classification = loss.prediction_loss(
        predictions.trajectories,
        predictions.log_standard_deviation,
        predictions.confidence_logits,
        batch["future_positions"],
        batch["future_mask"],
        predictions.anchors,
    )
    return TrainingStep(
        total,
        regression,
        classification,
        predictions.trajectories,
        predictions.confidence_logits,
    )


def unwrapped(predictor: torch.nn.Module) -> torch.nn.Module:
    """The bare model under the multi-process wrapper, so a checkpoint holds
    plain parameter names whatever the process count.
    """
    if isinstance(predictor, DistributedDataParallel):
        return predictor.module
    return predictor


def agreed_with_main_process(decision: bool, device: torch.device) -> bool:
    """Process 0's decision, shared with every process so they all stop
    together. Every process must make this call or the broadcast hangs.
    """
    if not distributed.is_initialized():
        return decision
    shared_decision = torch.tensor(int(decision), device=device)
    distributed.broadcast(shared_decision, src=0)
    return bool(shared_decision.item())


def checkpoint_state(predictor: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     gradient_scaler: GradScaler, seed: int,
                     completed_epochs: int) -> dict[str, Any]:
    """Bundles the model, optimizer and scaler state, training progress and a
    parameter fingerprint into a checkpoint dict.
    """
    model_state = unwrapped(predictor).state_dict()
    return {
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": gradient_scaler.state_dict(),
        "completed_epochs": completed_epochs,
        "seed": seed,
        "code_version": contract.STAGING_CODE_VERSION,
        "parameter_fingerprint": parameter_fingerprint(model_state),
    }


def epochs_left_to_train(completed_epochs: int, requested_epochs: int) -> range:
    """Epoch indices still to run, given how many are already done."""
    return range(completed_epochs, requested_epochs)


def save_checkpoint(checkpoint_path: Path, previous_checkpoint_path: Path,
                    state: dict[str, Any]) -> None:
    """Writes to a temporary file, then swaps it in, so a crash never truncates
    the checkpoint; the prior one is kept as a fallback.
    """
    partial_suffix = checkpoint_path.suffix + ".partial"
    partial_path = checkpoint_path.with_suffix(partial_suffix)
    torch.save(state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_checkpoint_path)
    partial_path.replace(checkpoint_path)


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
    " decay_end_step summary_writer is_main_process",
)


class EpochProgress:
    """Running totals for one epoch: losses, the training monitor, optimiser
    health and timing, overall and per logging window.
    """

    def __init__(self, device: torch.device) -> None:
        """Starts every total at zero."""
        self.device = device
        self.accumulator = metrics.MetricAccumulator()
        self.window_accumulator = metrics.MetricAccumulator()
        self.loss_sums = {
            "total": 0.0,
            "regression": 0.0,
            "classification": 0.0,
        }
        self.window_loss_sums = dict.fromkeys(self.loss_sums, 0.0)
        self.seconds = {"data_wait": 0.0, "step": 0.0, "monitor": 0.0}
        self.batch_count = 0
        self.sample_count = 0
        self.non_finite_total_count = 0
        self.gradient_scaler_skip_count = 0
        self.clipped_step_count = 0
        self.window_winner_counts = torch.zeros(QUERY_COUNT,
                                                dtype=torch.long,
                                                device=device)

    def record_step(self, step: TrainingStep, was_clipped: bool,
                    was_skipped: bool) -> bool:
        """Folds one step's losses and optimiser health into the totals, and
        returns whether the total loss was finite.
        """
        loss_values = {
            "total": float(step.total.detach()),
            "regression": float(step.regression.detach()),
            "classification": float(step.classification.detach()),
        }
        for name, value in loss_values.items():
            self.loss_sums[name] += value
            self.window_loss_sums[name] += value
        total_is_finite = math.isfinite(loss_values["total"])
        self.non_finite_total_count += int(not total_is_finite)
        self.clipped_step_count += int(was_clipped)
        self.gradient_scaler_skip_count += int(was_skipped)
        return total_is_finite

    def record_monitor(self, step: TrainingStep,
                       batch: dict[str, torch.Tensor]) -> None:
        """Training monitor: steers runs, never a reported number. Also counts
        which mode came closest, to spot modes that never win.
        """
        trajectories = step.trajectories.detach().float()
        confidence_logits = step.confidence_logits.detach().float()
        future_positions = batch["future_positions"]
        future_mask = batch["future_mask"]
        with torch.no_grad():
            for metric_accumulator in (self.accumulator,
                                       self.window_accumulator):
                metric_accumulator.update(trajectories, confidence_logits,
                                          future_positions, future_mask)
            mode_distances = metrics.mean_distance_per_mode(
                trajectories, future_positions, future_mask)
            window_winners = mode_distances.argmin(dim=1)
            self.window_winner_counts.scatter_add_(
                0, window_winners, torch.ones_like(window_winners))
        self.batch_count += 1
        self.sample_count += batch["agent_history"].shape[0]

    def window_scalars(self, step_learning_rate: float) -> dict[str, float]:
        """The table reported every LOG_EVERY_BATCHES: losses, the training
        monitor, optimiser health, and where the time went.
        """
        monitor = self.accumulator.results()
        window_monitor = self.window_accumulator.results()
        never_win_count = int((self.window_winner_counts == 0).sum())
        peak_gigabytes = 0.0
        if self.device.type == "cuda":
            peak_gigabytes = torch.cuda.max_memory_allocated() / 1e9
        loss_sums = self.loss_sums
        batch_count = self.batch_count
        seconds = self.seconds
        elapsed = sum(seconds.values())
        return {
            "loss/total": loss_sums["total"] / batch_count,
            "loss/regression": loss_sums["regression"] / batch_count,
            "loss/classification": loss_sums["classification"] / batch_count,
            "loss_window/total":
                self.window_loss_sums["total"] / LOG_EVERY_BATCHES,
            "monitor/ade_80step": monitor["min_ade"],
            "monitor/fde_80step": monitor["min_fde"],
            "monitor_window/ade_80step": window_monitor["min_ade"],
            "monitor_window/fde_80step": window_monitor["min_fde"],
            "monitor_window/kept_modes": window_monitor["mean_kept_modes"],
            "monitor_window/backfill_rate": window_monitor["backfill_rate"],
            "monitor_window/never_win_anchors": never_win_count,
            "health/non_finite_losses": self.non_finite_total_count,
            "health/skipped_steps": self.gradient_scaler_skip_count,
            "health/clipped_steps": self.clipped_step_count,
            "optimisation/learning_rate": step_learning_rate,
            "throughput/samples_per_second": self.sample_count / elapsed,
            "time_share/data_wait": seconds["data_wait"] / elapsed,
            "time_share/step": seconds["step"] / elapsed,
            "time_share/monitor": seconds["monitor"] / elapsed,
            "memory/peak_gigabytes": peak_gigabytes,
        }

    def start_new_window(self) -> None:
        """Clears the per-window totals once their table is reported."""
        self.window_accumulator = metrics.MetricAccumulator()
        self.window_loss_sums = dict.fromkeys(self.loss_sums, 0.0)
        self.window_winner_counts.zero_()

    def averages(self) -> dict[str, float]:
        """Mean of each loss term over the batches seen so far."""
        return {
            name: value / max(self.batch_count, 1)
            for name, value in self.loss_sums.items()
        }


def epoch_scalars(averages: dict[str, float], monitor: dict[str, float],
                  seconds: dict[str, float]) -> dict[str, float]:
    """The summary table reported once an epoch finishes."""
    return {
        "epoch/loss_total": averages["total"],
        "epoch/loss_regression": averages["regression"],
        "epoch/loss_classification": averages["classification"],
        "epoch/ade_80step": monitor["min_ade"],
        "epoch/fde_80step": monitor["min_fde"],
        "epoch/kept_modes": monitor["mean_kept_modes"],
        "epoch/backfill_rate": monitor["backfill_rate"],
        "epoch/data_wait_seconds": seconds["data_wait"],
        "epoch/step_seconds": seconds["step"],
        "epoch/monitor_seconds": seconds["monitor"],
    }


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
    gradient_scaler: GradScaler, batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device, settings: RunSettings, epoch_index: int,
    steps_before_epoch: int
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """One pass over the data.

    Each batch: forward and loss, optimiser step, training monitor, then a log
    line and a timed checkpoint. Only the main process logs and checkpoints.
    With several processes the caller wraps this in Join, so a process whose
    shard runs out first does not hang the others.
    """
    progress = EpochProgress(device)
    wait_start = time.perf_counter()
    checkpoint_wait_start = time.perf_counter()
    for batch in batches:
        progress.seconds["data_wait"] += time.perf_counter() - wait_start

        step_start = time.perf_counter()
        batch = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()
        }
        with torch.amp.autocast(device_type=device.type,
                                enabled=gradient_scaler.is_enabled()):
            step = training_losses(predictor, batch)
        # The rate is a function of the step count alone, so a
        # resumed run needs no scheduler state.
        step_learning_rate = scheduled_learning_rate(
            steps_before_epoch + progress.batch_count,
            settings.warmup_steps,
            settings.learning_rate,
            settings.decay_start_step,
            settings.decay_end_step,
        )
        was_clipped, was_skipped = optimisation_step(
            predictor, optimizer, gradient_scaler, step.total,
            step_learning_rate, settings.gradient_clip_norm)
        total_is_finite = progress.record_step(step, was_clipped, was_skipped)
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
                    checkpoint_state(predictor, optimizer, gradient_scaler,
                                     settings.seed, epoch_index),
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
        distributed.init_process_group(
            "nccl" if device.type == "cuda" else "gloo")
    return Processes(count, rank, local_rank, rank == 0), device


def announce(processes: Processes, message: str) -> None:
    """Prints from the main process only."""
    if processes.is_main:
        print(message, flush=True)


def resume_training(arguments: argparse.Namespace, predictor: MotionPredictor,
                    optimizer: torch.optim.Optimizer,
                    gradient_scaler: GradScaler, previous_checkpoint_path: Path,
                    device: torch.device, processes: Processes) -> int:
    """Loads the checkpoint when --resume is set and returns how many epochs
    are already complete.

    Falls back to the previous checkpoint when the current one is missing,
    which a crash between save_checkpoint's two renames leaves behind.
    """
    if not arguments.resume:
        return 0
    resume_path = arguments.checkpoint_path
    if not resume_path.exists() and previous_checkpoint_path.exists():
        resume_path = previous_checkpoint_path
        announce(
            processes,
            f"{arguments.checkpoint_path} missing, resuming from {resume_path}")
    if not resume_path.exists():
        return 0

    checkpoint = load_checkpoint_state(resume_path, map_location=device)
    predictor.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "gradient_scaler_state" in checkpoint:
        gradient_scaler.load_state_dict(checkpoint["gradient_scaler_state"])
    completed_epochs = checkpoint["completed_epochs"]
    announce(processes,
             f"resuming {resume_path}: {completed_epochs} epochs complete")
    return completed_epochs


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


def plan_epochs(arguments: argparse.Namespace, processes: Processes,
                device: torch.device) -> tuple[list[Path], int]:
    """Finds this process's share of the staged scenarios and counts the
    optimiser steps in one epoch, which the learning-rate schedule is measured
    in.
    """
    every_scenario_path = sorted(arguments.staged_directory.glob("*.npz"))
    scenario_paths = every_scenario_path[processes.rank::processes.count]
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"
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
    return scenario_paths, steps_per_epoch


def build_run_settings(arguments: argparse.Namespace,
                       previous_checkpoint_path: Path, steps_per_epoch: int,
                       processes: Processes) -> RunSettings:
    """Gathers the settings that stay fixed for the whole run, and opens the
    TensorBoard event file beside the checkpoint on the main process.
    """
    decay_start_step, decay_end_step = decay_window(arguments, steps_per_epoch)
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
    )


def main() -> None:
    """Entry point: builds the model, resumes from a checkpoint if asked, and
    trains the remaining epochs inside the time budget.
    """
    arguments = parse_arguments()
    processes, device = start_processes()
    assert arguments.batch_size % processes.count == 0, (
        f"--batch-size {arguments.batch_size} does not split evenly"
        f" over {processes.count} processes")

    torch.manual_seed(arguments.seed)
    predictor, optimizer, gradient_scaler = build_training_objects(
        arguments, device)
    scenario_paths, steps_per_epoch = plan_epochs(arguments, processes, device)
    designated_targets_only = not arguments.all_eligible_agents

    previous_checkpoint_path = arguments.checkpoint_path.with_suffix(
        arguments.checkpoint_path.suffix + ".previous")
    completed_epochs = resume_training(arguments, predictor, optimizer,
                                       gradient_scaler,
                                       previous_checkpoint_path, device,
                                       processes)
    remaining_epochs = epochs_left_to_train(completed_epochs, arguments.epochs)
    if not remaining_epochs:
        announce(
            processes, f"NOTHING TO TRAIN: {arguments.checkpoint_path} already"
            f" holds {completed_epochs} completed epochs and"
            f" --epochs is {arguments.epochs}.")
        return

    settings = build_run_settings(arguments, previous_checkpoint_path,
                                  steps_per_epoch, processes)
    if processes.count > 1:
        device_ids = [processes.local_rank] if device.type == "cuda" else None
        predictor = DistributedDataParallel(predictor, device_ids=device_ids)
    budget_hours = arguments.stop_after_seconds / 3600

    training_start = time.perf_counter()
    steps_before_epoch = completed_epochs * steps_per_epoch
    last_epoch_seconds = 0.0
    for epoch_index in remaining_epochs:
        # Stop before an epoch that would not fit in the time budget:
        # Kaggle ends a session without warning.
        elapsed_seconds = time.perf_counter() - training_start
        next_epoch_overruns = (elapsed_seconds + last_epoch_seconds
                               > arguments.stop_after_seconds)
        if agreed_with_main_process(next_epoch_overruns, device):
            announce(
                processes, f"STOPPING BEFORE EPOCH {epoch_index + 1}:"
                f" {elapsed_seconds / 3600:.2f} h elapsed, the last epoch"
                f" took {last_epoch_seconds / 3600:.2f} h, and the"
                f" --stop-after-seconds budget is {budget_hours:.2f} h."
                f" {epoch_index} of {arguments.epochs} epochs are complete"
                f" and {arguments.checkpoint_path} holds them.")
            return

        epoch_start = time.perf_counter()
        batches = pipeline.batches(
            scenario_paths,
            arguments.workers,
            arguments.batch_size // processes.count,
            arguments.prefetch,
            arguments.seed + epoch_index,
            designated_targets_only,
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
                checkpoint_state(predictor, optimizer, gradient_scaler,
                                 arguments.seed, epoch_index + 1),
            )

        elapsed_seconds = time.perf_counter() - training_start
        budget_is_spent = elapsed_seconds >= arguments.stop_after_seconds
        if agreed_with_main_process(budget_is_spent, device):
            announce(
                processes, f"STOPPING EARLY: {elapsed_seconds / 3600:.2f} h"
                f" elapsed against a --stop-after-seconds budget of"
                f" {budget_hours:.2f} h. {epoch_index + 1} of"
                f" {arguments.epochs} epochs are complete and"
                f" {arguments.checkpoint_path} holds them.")
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
