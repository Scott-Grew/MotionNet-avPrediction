"""Step 3 of 5.

Trains the model on staged scenarios; resumable, with a time budget so a session
stops cleanly. Runs as one process, or as one process per GPU under torchrun.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import contextlib
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.distributed as distributed
from torch import Tensor, nn
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from womd import loader, loss, pipeline
from womd.checkpoint import (
    GradScaler,
    Progress,
    TrainingObjects,
    checkpoint_of,
    load_anchor_file,
    load_checkpoint,
    previous_path,
    save_checkpoint,
)
from womd.loader import SceneBatch
from womd.metrics import (
    LOG_EVERY_BATCHES,
    EpochProgress,
    EpochSummary,
    StepOutcome,
    TrainingStep,
    epoch_scalars,
)
from womd.model import MotionPredictor
from womd.pipeline import BatchSettings

# The peak learning rate unless --learning-rate sets another, and
# AdamW's own default weight decay.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
SECONDS_PER_HOUR = 3600

Batches = Iterable[SceneBatch]
Scalars = dict[str, float]


class Processes(NamedTuple):
    """Where this process sits among the processes torchrun started; a plain
    launch is one process of one.
    """

    count: int
    rank: int
    local_rank: int
    is_main: bool


class EpochPlan(NamedTuple):
    """The optimiser steps in one epoch and the launch layout that fixes it; a
    checkpoint stores it so a resumed run with the same layout reuses it.
    """

    scenario_count: int
    process_count: int
    worker_count: int
    per_process_batch_size: int
    designated_only: bool
    steps_per_epoch: int


class Schedule(NamedTuple):
    """The learning rate as a function of the step count alone: a linear
    warm-up, a hold, then an optional cosine fall to zero.
    """

    peak: float
    warmup_steps: int
    decay_start_step: int | None
    decay_end_step: int | None

    def at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak * (step + 1) / self.warmup_steps
        if self.decay_start_step is None or step < self.decay_start_step:
            return self.peak
        decay_length = max(self.decay_end_step - 1 - self.decay_start_step, 1)
        fraction = min((step - self.decay_start_step) / decay_length, 1.0)
        # The rate is peak * (1 + cos(pi f)) / 2 for fraction f through the
        # fall.
        return self.peak * 0.5 * (1.0 + math.cos(math.pi * fraction))


class Reporter:
    """Writes each table of scalars to TensorBoard and prints it as the log
    line, so the console and the board never drift.
    """

    def __init__(self, directory: Path) -> None:
        self.writer = SummaryWriter(directory)

    def report(self, heading: str, scalars: Scalars, step: int) -> None:
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        printed = [f"{name} {value:.4g}" for name, value in scalars.items()]
        print(" | ".join([heading] + printed), flush=True)

    def flush(self) -> None:
        self.writer.flush()


class RunSettings(NamedTuple):
    """The settings of a run that stay fixed from its first batch to its
    last. Only the main process holds a reporter.
    """

    checkpoint_path: Path
    checkpoint_every_seconds: float
    seed: int
    gradient_clip_norm: float
    schedule: Schedule
    reporter: Reporter | None
    epoch_plan: EpochPlan


def parameter_groups(predictor: nn.Module) -> list[dict[str, Any]]:
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
        {"params": decayed, "weight_decay": WEIGHT_DECAY},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def steps_per_epoch(paths: list[Path], settings: BatchSettings) -> int:
    """Optimiser steps per epoch, the sum over worker streams of ceil(targets
    / batch size), because each worker fills its own batches.
    """
    stream_count = max(settings.worker_count, 1)
    step_count = 0
    for stream_index in range(stream_count):
        stream_targets = 0
        for path in paths[stream_index::stream_count]:
            scenario = loader.read_scenario(path)
            eligible = loader.eligible_tracks(
                scenario, settings.designated_only
            )
            stream_targets += len(eligible)
        step_count += math.ceil(stream_targets / settings.batch_size)
    return step_count


def remaining_epochs(completed: int, requested: int) -> range:
    """The epochs still to train; a resumed run restarts at the first epoch
    its checkpoint has not finished.
    """
    return range(completed, requested)


def training_step(predictor: nn.Module, batch: SceneBatch) -> TrainingStep:
    """Runs the predictor on a batch and returns its losses and predictions."""
    predicted = predictor(batch, for_training=True)
    return TrainingStep(loss.loss_terms(predicted, batch.targets), predicted)


def agreed_with_main_process(decision: bool, device: torch.device) -> bool:
    """Process 0's decision, shared with every process so they all stop
    together. Every process must make this call or the broadcast hangs.
    """
    if not distributed.is_initialized():
        return decision
    shared = torch.tensor(int(decision), device=device)
    distributed.broadcast(shared, src=0)
    return bool(shared.item())


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
            "nccl" if device.type == "cuda" else "gloo"
        )
    return Processes(count, rank, local_rank, rank == 0), device


def parse_arguments() -> argparse.Namespace:
    """The command line. Everything that shapes a run is required, so a launch
    script states its recipe in full.
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


class TrainingRun:
    """One launch of train.py: the model, its optimiser and scaler, the
    processes it runs on, and the loop over the remaining epochs. run() fixes
    the settings once the epoch plan is known.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.processes, self.device = start_processes()
        if arguments.batch_size % self.processes.count != 0:
            raise ValueError(
                f"--batch-size {arguments.batch_size} does not split"
                f" over {self.processes.count} processes"
            )
        torch.manual_seed(arguments.seed)
        predictor = MotionPredictor(load_anchor_file(arguments.anchors)).to(
            self.device
        )
        optimizer = torch.optim.AdamW(
            parameter_groups(predictor), lr=arguments.learning_rate
        )
        scaler = GradScaler(
            enabled=arguments.mixed_precision and self.device.type == "cuda"
        )
        self.trained = TrainingObjects(predictor, optimizer, scaler)
        self.completed_epochs, self.stored_plan = self.resume()

    def announce(self, message: str) -> None:
        if self.processes.is_main:
            print(message, flush=True)

    def stopped(self, should_stop: bool, message: str) -> bool:
        """Whether the main process's stop decision holds, announcing message
        when it does. Every process must make this call together.
        """
        if not agreed_with_main_process(should_stop, self.device):
            return False
        self.announce(message)
        return True

    def resume(self) -> tuple[int, EpochPlan | None]:
        """Loads the checkpoint when --resume is set and returns how many epochs
        are complete and the plan they were trained under. Falls back to the
        previous checkpoint, which a crash between save_checkpoint's two
        renames leaves behind.
        """
        if not self.arguments.resume:
            return 0, None
        path = self.arguments.checkpoint_path
        if not path.exists() and previous_path(path).exists():
            path = previous_path(path)
            self.announce(
                f"{self.arguments.checkpoint_path} missing,"
                f" resuming from {path}"
            )
        if not path.exists():
            return 0, None
        checkpoint = load_checkpoint(path, self.device)
        self.trained.predictor.load_state_dict(checkpoint["model_state"])
        self.trained.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "gradient_scaler_state" in checkpoint:
            self.trained.scaler.load_state_dict(
                checkpoint["gradient_scaler_state"]
            )
        completed = checkpoint["completed_epochs"]
        self.announce(f"resuming {path}: {completed} epochs complete")
        stored = checkpoint.get("epoch_plan")
        return completed, None if stored is None else EpochPlan(**stored)

    def batch_settings(self, epoch_index: int) -> BatchSettings:
        arguments = self.arguments
        return BatchSettings(
            batch_size=arguments.batch_size // self.processes.count,
            worker_count=arguments.workers,
            prefetch_batches=arguments.prefetch,
            seed=arguments.seed + epoch_index,
            designated_only=not arguments.all_eligible_agents,
        )

    def plan_epochs(self, paths: list[Path]) -> EpochPlan:
        """The optimiser steps in one epoch, which the learning-rate schedule is
        measured in. Counting opens every staged file, so the count is saved in
        the checkpoint and reused while the scenarios and the launch layout are
        unchanged.
        """
        settings = self.batch_settings(0)
        layout = EpochPlan(
            scenario_count=len(paths) * self.processes.count,
            process_count=self.processes.count,
            worker_count=settings.worker_count,
            per_process_batch_size=settings.batch_size,
            designated_only=settings.designated_only,
            steps_per_epoch=0,
        )
        stored = self.stored_plan
        if stored is not None and stored._replace(steps_per_epoch=0) == layout:
            self.announce("optimiser step count read from the checkpoint")
            return stored
        count = steps_per_epoch(paths, settings)
        if self.processes.count > 1:
            # Every process uses the longest share's step count, so the
            # schedule is the same everywhere.
            longest = torch.tensor(count, device=self.device)
            distributed.all_reduce(longest, op=distributed.ReduceOp.MAX)
            count = int(longest.item())
        epochs = self.arguments.epochs
        self.announce(
            f"{count} optimiser steps per epoch,"
            f" {count * epochs} over {epochs} epochs"
        )
        return layout._replace(steps_per_epoch=count)

    def run_settings(self, plan: EpochPlan) -> RunSettings:
        arguments = self.arguments
        decay_start = None
        decay_end = None
        if arguments.decay_from_epoch is not None:
            decay_start = (
                arguments.decay_from_epoch - 1
            ) * plan.steps_per_epoch
            decay_end = arguments.epochs * plan.steps_per_epoch
        reporter = None
        if self.processes.is_main:
            reporter = Reporter(
                arguments.checkpoint_path.parent / "tensorboard"
            )
        schedule = Schedule(
            arguments.learning_rate,
            arguments.warmup_steps,
            decay_start,
            decay_end,
        )
        return RunSettings(
            checkpoint_path=arguments.checkpoint_path,
            checkpoint_every_seconds=arguments.checkpoint_every_seconds,
            seed=arguments.seed,
            gradient_clip_norm=arguments.gradient_clip_norm,
            schedule=schedule,
            reporter=reporter,
            epoch_plan=plan,
        )

    def save(self, completed_epochs: int) -> None:
        settings = self.settings
        progress = Progress(
            settings.seed, completed_epochs, settings.epoch_plan._asdict()
        )
        save_checkpoint(
            settings.checkpoint_path, checkpoint_of(self.trained, progress)
        )

    def optimise(self, total: Tensor, rate: float, clip: float) -> StepOutcome:
        """Sets this step's rate, then backward pass, gradient clip to clip and
        optimiser step.
        """
        trained = self.trained
        for group in trained.optimizer.param_groups:
            group["lr"] = rate
        trained.optimizer.zero_grad()
        trained.scaler.scale(total).backward()
        trained.scaler.unscale_(trained.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trained.predictor.parameters(), clip
        )
        trained.scaler.step(trained.optimizer)
        # The scaler lowers its scale only when it skipped a step over
        # non-finite gradients, which is how skips are detected.
        scale_before = trained.scaler.get_scale()
        trained.scaler.update()
        skipped = trained.scaler.get_scale() < scale_before
        return StepOutcome(float(gradient_norm) > clip, skipped)

    def train_epoch(self, batches: Batches, epoch_index: int) -> EpochSummary:
        """One pass over the data. Each batch goes through forward and loss,
        the optimiser step and the training monitor, then a log line and a
        timed checkpoint. Only the main process logs and checkpoints.
        """
        settings = self.settings
        progress = EpochProgress(self.device)
        steps_before = epoch_index * settings.epoch_plan.steps_per_epoch
        wait_start = time.perf_counter()
        checkpoint_start = time.perf_counter()
        for batch in batches:
            progress.seconds.data_wait += time.perf_counter() - wait_start
            step_start = time.perf_counter()
            batch = batch.each(
                lambda tensor: tensor.to(self.device, non_blocking=True)
            )
            mixed_precision = self.trained.scaler.is_enabled()
            with torch.amp.autocast(
                device_type=self.device.type, enabled=mixed_precision
            ):
                step = training_step(self.trained.predictor, batch)
            global_step = steps_before + progress.batch_count
            rate = settings.schedule.at(global_step)
            outcome = self.optimise(
                step.losses.total, rate, settings.gradient_clip_norm
            )
            total_is_finite = progress.record_step(step, outcome)
            progress.seconds.step += time.perf_counter() - step_start

            monitor_start = time.perf_counter()
            progress.record_monitor(step, batch)
            progress.seconds.monitor += time.perf_counter() - monitor_start

            if progress.batch_count % LOG_EVERY_BATCHES == 0:
                if settings.reporter is not None:
                    heading = f"  batch {progress.batch_count}"
                    scalars = progress.window_scalars(rate)
                    settings.reporter.report(heading, scalars, global_step + 1)
                progress.start_new_window()

            # Timed checkpoint, skipped while the loss is non-finite.
            seconds_since = time.perf_counter() - checkpoint_start
            if (
                self.processes.is_main
                and seconds_since >= settings.checkpoint_every_seconds
            ):
                if total_is_finite:
                    self.save(epoch_index)
                checkpoint_start = time.perf_counter()
            wait_start = time.perf_counter()
        return progress.summary()

    def stop_message(self, prefix: str, elapsed: float, completed: int) -> str:
        arguments = self.arguments
        return (
            f"{prefix}: {elapsed / SECONDS_PER_HOUR:.2f} h elapsed against a"
            f" --stop-after-seconds budget of"
            f" {arguments.stop_after_seconds / SECONDS_PER_HOUR:.2f} h."
            f" {completed} of {arguments.epochs} epochs are complete and"
            f" {arguments.checkpoint_path} holds them."
        )

    def run(self) -> None:
        """Trains the remaining epochs inside the time budget, checkpointing
        after each finished epoch.
        """
        arguments = self.arguments
        remaining = remaining_epochs(self.completed_epochs, arguments.epochs)
        if not remaining:
            self.announce(
                f"NOTHING TO TRAIN: {arguments.checkpoint_path}"
                f" holds {self.completed_epochs} epochs"
            )
            return
        every_path = sorted(arguments.staged_directory.glob("*.npz"))
        paths = every_path[self.processes.rank :: self.processes.count]
        if not paths:
            raise FileNotFoundError(
                f"no .npz scenarios in {arguments.staged_directory}"
            )
        self.settings = self.run_settings(self.plan_epochs(paths))
        settings = self.settings
        predictor = self.trained.predictor
        if self.processes.count > 1:
            device_ids = (
                [self.processes.local_rank]
                if self.device.type == "cuda"
                else None
            )
            predictor = DistributedDataParallel(
                predictor, device_ids=device_ids
            )
            self.trained = self.trained._replace(predictor=predictor)
        budget = arguments.stop_after_seconds

        start = time.perf_counter()
        last_epoch_seconds = 0.0
        for epoch_index in remaining:
            # Stop before an epoch that would not fit in the time budget,
            # since Kaggle ends a session without warning.
            elapsed = time.perf_counter() - start
            would_overrun = elapsed + last_epoch_seconds > budget
            before = self.stop_message(
                f"STOPPING BEFORE EPOCH {epoch_index + 1}", elapsed, epoch_index
            )
            if self.stopped(would_overrun, before):
                return

            epoch_start = time.perf_counter()
            batches = pipeline.batches(paths, self.batch_settings(epoch_index))
            uneven_shares = contextlib.nullcontext()
            if self.processes.count > 1:
                uneven_shares = Join([predictor])
            with uneven_shares:
                summary = self.train_epoch(batches, epoch_index)
            last_epoch_seconds = time.perf_counter() - epoch_start
            if settings.reporter is not None:
                heading = f"epoch {epoch_index + 1}/{arguments.epochs}"
                settings.reporter.report(
                    heading, epoch_scalars(summary), epoch_index + 1
                )
                settings.reporter.flush()

            if not math.isfinite(summary.losses.total):
                self.announce(
                    f"epoch {epoch_index + 1} mean total loss"
                    f" {summary.losses.total}, checkpoint left as it was"
                )
            elif self.processes.is_main:
                self.save(epoch_index + 1)

            elapsed = time.perf_counter() - start
            early = self.stop_message(
                "STOPPING EARLY", elapsed, epoch_index + 1
            )
            if self.stopped(elapsed >= budget, early):
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
    TrainingRun(parse_arguments()).run()
    if distributed.is_initialized():
        distributed.destroy_process_group()
