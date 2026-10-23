"""Step 3 of 5.

Trains the model on staged scenarios on one device; resumable, with a time
budget so a Kaggle session stops cleanly before it is cut off.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import faulthandler
import hashlib
import math
import signal
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from womd import layout, loss, pipeline
from womd.checkpoint import (
    ANCHORS_KEY,
    GradScaler,
    check_model_state,
    load_anchor_file,
    load_checkpoint,
    previous_path,
    save_checkpoint,
)
from womd.metrics import LOG_EVERY_BATCHES, EpochProgress, Scalars
from womd.model import MotionPredictor
from womd.pipeline import BatchSettings
from womd.views import TargetInputs

# The peak learning rate unless --learning-rate sets another, and
# AdamW's own default weight decay.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
SECONDS_PER_HOUR = 3600
STACK_DUMP_SECONDS = 1800

Batches = Iterable[TargetInputs]


class EpochPlan(NamedTuple):
    """The optimiser steps in one epoch and the settings that fix it. Counting
    opens every staged file, so a checkpoint stores the plan and a resumed run
    with the same settings reuses it.
    """

    scenario_names_digest: str
    worker_count: int
    batch_size: int
    designated_only: bool
    steps_per_epoch: int

    @classmethod
    def stored(cls, record: dict[str, Any] | None) -> EpochPlan | None:
        """The plan a checkpoint holds, or None when it holds none or was
        written with other fields.
        """
        if record is None or set(record) != set(cls._fields):
            return None
        return cls(**record)


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


def parameter_groups(predictor: nn.Module) -> list[dict[str, Any]]:
    """Splits parameters so weight decay applies to matrices only, not to
    biases or norms.
    """
    parameters = list(predictor.parameters())
    decayed = [parameter for parameter in parameters if parameter.ndim >= 2]
    undecayed = [parameter for parameter in parameters if parameter.ndim < 2]
    return [
        {"params": decayed, "weight_decay": WEIGHT_DECAY},
        {"params": undecayed, "weight_decay": 0.0},
    ]


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
    """One launch of train.py, set up in full before the first batch: the
    model with its optimiser and scaler restored from any checkpoint, the
    epoch plan and the learning-rate schedule. run() is the epoch loop.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.launch_time = time.perf_counter()
        self.arguments = arguments
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        torch.manual_seed(arguments.seed)
        anchors = load_anchor_file(arguments.anchors)
        self.predictor = MotionPredictor(anchors).to(self.device)
        groups = parameter_groups(self.predictor)
        self.optimizer = torch.optim.AdamW(groups, lr=arguments.learning_rate)
        scaling = arguments.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=scaling)
        self.completed_epochs, stored_plan = self.resume()
        self.saved_epochs = self.completed_epochs

        self.paths = sorted(arguments.staged_directory.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(
                f"no .npz scenarios in {arguments.staged_directory}"
            )
        self.plan = self.plan_epochs(stored_plan)
        self.schedule = self.learning_rate_schedule()
        tensorboard = arguments.checkpoint_path.parent / "tensorboard"
        self.writer = SummaryWriter(tensorboard)

    def report(self, heading: str, scalars: Scalars, step: int) -> None:
        """Writes a table of scalars to TensorBoard and prints it as the log
        line, so the console and the board never drift.
        """
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        printed = [f"{name} {value:.4g}" for name, value in scalars.items()]
        print(" | ".join([heading] + printed))
        self.writer.flush()

    def resume(self) -> tuple[int, EpochPlan | None]:
        """Loads the checkpoint when --resume is set and returns how many epochs
        are complete and the plan they were trained under. Falls back to the
        previous checkpoint, which a crash between save_checkpoint's two
        renames leaves behind.
        """
        if not self.arguments.resume:
            return 0, None
        path = self.arguments.checkpoint_path
        fallback = previous_path(path)
        if not path.exists() and fallback.exists():
            # A crash between save_checkpoint's two renames leaves only the
            # fallback, which is moved back so the checkpoint path holds it.
            fallback.replace(path)
            print(f"checkpoint missing, restored it from {fallback}")
        if not path.exists():
            return 0, None
        checkpoint = load_checkpoint(path, self.device)
        model_state = checkpoint["model_state"]
        check_model_state(self.predictor, model_state, path)
        stored_anchors = model_state[ANCHORS_KEY].to(self.device)
        given_anchors = self.predictor.state_dict()[ANCHORS_KEY]
        if not torch.equal(stored_anchors, given_anchors):
            raise ValueError(
                f"{path} was trained with other anchors than"
                f" {self.arguments.anchors}; resume with the anchors it holds"
            )
        self.predictor.load_state_dict(model_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["gradient_scaler_state"])
        completed = checkpoint["completed_epochs"]
        print(f"resuming {path}: {completed} epochs complete")
        stored = EpochPlan.stored(checkpoint.get("epoch_plan"))
        return completed, stored

    def batch_settings(self, epoch_index: int) -> BatchSettings:
        arguments = self.arguments
        return BatchSettings(
            batch_size=arguments.batch_size,
            worker_count=arguments.workers,
            prefetch_batches=arguments.prefetch,
            seed=arguments.seed + epoch_index,
            designated_only=not arguments.all_eligible_agents,
        )

    def plan_epochs(self, stored: EpochPlan | None) -> EpochPlan:
        """The optimiser steps in one epoch, which the learning-rate schedule is
        measured in, read from the checkpoint when the settings match.
        """
        settings = self.batch_settings(0)
        names = "\n".join(path.name for path in self.paths).encode()
        settings_now = (
            hashlib.sha256(names).hexdigest(),
            settings.worker_count,
            settings.batch_size,
            settings.designated_only,
        )
        if stored is not None and stored[:-1] == settings_now:
            print("optimiser step count read from the checkpoint")
            return stored
        count = pipeline.steps_per_epoch(self.paths, settings)
        epochs = self.arguments.epochs
        print(
            f"{count} optimiser steps per epoch,"
            f" {count * epochs} over {epochs} epochs"
        )
        return EpochPlan(*settings_now, steps_per_epoch=count)

    def learning_rate_schedule(self) -> Schedule:
        arguments = self.arguments
        steps = self.plan.steps_per_epoch
        decay_start = None
        decay_end = None
        if arguments.decay_from_epoch is not None:
            # The fall starts no earlier than the end of warm-up, so the
            # rate never drops from the peak in one step.
            first_decay_step = (arguments.decay_from_epoch - 1) * steps
            decay_start = max(first_decay_step, arguments.warmup_steps)
            decay_end = arguments.epochs * steps
        peak = arguments.learning_rate
        return Schedule(peak, arguments.warmup_steps, decay_start, decay_end)

    def save(self, completed_epochs: int) -> None:
        model_state = self.predictor.state_dict()
        state = {
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "gradient_scaler_state": self.scaler.state_dict(),
            "completed_epochs": completed_epochs,
            "code_version": layout.STAGING_CODE_VERSION,
            "epoch_plan": self.plan._asdict(),
        }
        save_checkpoint(self.arguments.checkpoint_path, state)
        self.saved_epochs = completed_epochs

    def optimise(self, total: Tensor, rate: float) -> tuple[bool, bool]:
        """Sets this step's rate, then backward pass, gradient clip and
        optimiser step. Returns whether the gradient was clipped and whether
        the scaler skipped the step over non-finite gradients.
        """
        clip = self.arguments.gradient_clip_norm
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad()
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        parameters = self.predictor.parameters()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
        # A disabled scaler steps on any gradient, so a non-finite one is
        # skipped here instead.
        step_is_finite = torch.isfinite(total) & torch.isfinite(gradient_norm)
        if not self.scaler.is_enabled() and not step_is_finite:
            self.optimizer.zero_grad()
            return False, True
        self.scaler.step(self.optimizer)
        # The scaler lowers its scale only when it skipped a step over
        # non-finite gradients, which is how skips are detected.
        scale_before = self.scaler.get_scale()
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale_before
        return float(gradient_norm) > clip, skipped

    def train_epoch(self, batches: Batches, epoch_index: int) -> Scalars:
        """One pass over the data: forward and loss, the optimiser step and
        the monitor for each batch, then a log line every LOG_EVERY_BATCHES
        and a checkpoint every --checkpoint-every-seconds.
        """
        progress = EpochProgress(self.device)
        steps_before = epoch_index * self.plan.steps_per_epoch
        mixed_precision = self.scaler.is_enabled()
        last_checkpoint = time.perf_counter()
        wait_start = time.perf_counter()
        for host_batch in batches:
            progress.data_wait_seconds += time.perf_counter() - wait_start
            batch = host_batch.each(
                lambda array: array.to(self.device, non_blocking=True)
            )
            with torch.amp.autocast(self.device.type, enabled=mixed_precision):
                rounds = self.predictor.every_round(batch)
                losses = loss.round_loss_terms(rounds, batch)
            predicted = rounds[-1]
            global_step = steps_before + progress.batch_count
            rate = self.schedule.at(global_step)
            clipped, skipped = self.optimise(losses.total, rate)
            loss_is_finite = progress.record(losses, clipped, skipped)
            progress.monitor_batch(predicted, batch)

            if progress.batch_count % LOG_EVERY_BATCHES == 0:
                heading = f"  batch {progress.batch_count}"
                scalars = progress.log_scalars(rate)
                self.report(heading, scalars, global_step + 1)

            seconds = time.perf_counter() - last_checkpoint
            if seconds >= self.arguments.checkpoint_every_seconds:
                if loss_is_finite:
                    self.save(epoch_index)
                last_checkpoint = time.perf_counter()
            wait_start = time.perf_counter()
        return progress.epoch_scalars()

    def stop_message(self, prefix: str, elapsed: float, completed: int) -> str:
        arguments = self.arguments
        return (
            f"{prefix}: {elapsed / SECONDS_PER_HOUR:.2f} h elapsed against a"
            f" --stop-after-seconds budget of"
            f" {arguments.stop_after_seconds / SECONDS_PER_HOUR:.2f} h."
            f" {completed} of {arguments.epochs} epochs are complete and"
            f" {arguments.checkpoint_path} holds {self.saved_epochs}."
        )

    def run(self) -> None:
        """Trains the remaining epochs inside the time budget, checkpointing
        after each finished epoch.
        """
        arguments = self.arguments
        remaining = range(self.completed_epochs, arguments.epochs)
        budget = arguments.stop_after_seconds
        start = self.launch_time
        last_epoch_seconds = 0.0
        for epoch_index in remaining:
            # Stop before an epoch that would not fit in the time budget,
            # since Kaggle ends a session without warning.
            elapsed = time.perf_counter() - start
            if elapsed + last_epoch_seconds > budget:
                heading = f"STOPPING BEFORE EPOCH {epoch_index + 1}"
                print(self.stop_message(heading, elapsed, epoch_index))
                return

            epoch_start = time.perf_counter()
            settings = self.batch_settings(epoch_index)
            batches = pipeline.batches(self.paths, settings)
            scalars = self.train_epoch(batches, epoch_index)
            last_epoch_seconds = time.perf_counter() - epoch_start
            heading = f"epoch {epoch_index + 1}/{arguments.epochs}"
            self.report(heading, scalars, epoch_index + 1)

            total = scalars["epoch/loss_total"]
            if math.isfinite(total):
                self.save(epoch_index + 1)
            else:
                print(
                    f"epoch {epoch_index + 1} mean total loss {total},"
                    f" checkpoint left as it was"
                )

            elapsed = time.perf_counter() - start
            if elapsed >= budget:
                completed = epoch_index + 1
                print(self.stop_message("STOPPING EARLY", elapsed, completed))
                return
        print(
            f"done: {arguments.checkpoint_path} holds"
            f" {self.saved_epochs} epochs"
        )


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
    faulthandler.dump_traceback_later(
        STACK_DUMP_SECONDS, repeat=True, file=sys.stderr
    )
    # The Kaggle launcher sends SIGUSR1 before it kills a run that passed
    # its deadline, so the log shows where the run was.
    faulthandler.register(signal.SIGUSR1, file=sys.stderr)
    # Kaggle pipes stdout to its log, so each line is flushed as printed.
    sys.stdout.reconfigure(line_buffering=True)
    TrainingRun(parse_arguments()).run()
