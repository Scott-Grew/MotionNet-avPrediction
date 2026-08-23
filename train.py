import womd.runtime_env
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from womd import contract, loader, loss, metrics, model, pipeline
from womd.model import QUERY_COUNT, MotionPredictor

LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.01
HEADING_LOSS_WEIGHT = 0.5
CLASSIFICATION_LOSS_WEIGHT = 1.0
NEIGHBOUR_FUTURE_LOSS_WEIGHT = 0.5
SPEED_LOSS_WEIGHT = 0.5
DECAY_EPOCHS = 1
LOG_EVERY_BATCHES = 20

GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)

def parameter_groups(predictor):
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

def optimiser_steps_per_epoch(scenario_paths, worker_count, batch_size, designated_targets_only):
    stream_count = max(worker_count, 1)
    steps = 0
    for stream_index in range(stream_count):
        stream_sample_count = 0
        for scenario_path in scenario_paths[stream_index::stream_count]:
            with np.load(scenario_path) as scenario_file:
                stream_sample_count += len(loader.eligible_track_indices(
                    scenario_file["track_rows"],
                    scenario_file["track_valid"],
                    scenario_file["is_designated_target"],
                    designated_targets_only,
                ))
        steps += math.ceil(stream_sample_count / batch_size)
    return steps

def scheduled_learning_rate(
    completed_steps, warmup_steps, total_steps, decay_steps,
    elapsed_seconds=0.0, budget_seconds=float("inf"), seconds_per_step=None,
    process_steps=None,
):
    if process_steps is None:
        process_steps = completed_steps
    if process_steps < warmup_steps:
        return LEARNING_RATE * (process_steps + 1) / warmup_steps
    decay_start = max(total_steps - decay_steps, warmup_steps)
    rate_by_steps = LEARNING_RATE
    if completed_steps >= decay_start:
        rate_by_steps = LEARNING_RATE * max(total_steps - completed_steps, 0) / max(
            total_steps - decay_start, 1
        )
    rate_by_clock = LEARNING_RATE
    if seconds_per_step is not None and math.isfinite(budget_seconds):
        decay_window_seconds = decay_steps * seconds_per_step
        remaining_seconds = budget_seconds - elapsed_seconds
        rate_by_clock = LEARNING_RATE * min(
            max(remaining_seconds / max(decay_window_seconds, 1e-9), 0.0), 1.0
        )
    return min(rate_by_steps, rate_by_clock)

def round_summed_prediction_loss(
    round_outputs, batch, selected_unit_anchors, mode_valid,
    heading_loss_weight, classification_loss_weight, speed_loss_weight,
):
    summed = None
    for round_output in round_outputs:
        components = loss.prediction_loss(
            *round_output,
            batch["future_positions"], batch["future_headings"], batch["future_mask"],
            selected_unit_anchors,
            heading_loss_weight, classification_loss_weight, speed_loss_weight,
            mode_valid,
        )
        summed = components if summed is None else tuple(
            running + component for running, component in zip(summed, components)
        )
    return summed

class TrainingStep(nn.Module):
    def __init__(
        self, predictor, heading_loss_weight, classification_loss_weight,
        neighbour_future_loss_weight, speed_loss_weight,
    ):
        super().__init__()
        self.predictor = predictor
        self.heading_loss_weight = heading_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.neighbour_future_loss_weight = neighbour_future_loss_weight
        self.speed_loss_weight = speed_loss_weight

    def forward(self, batch):
        (
            round_outputs, selected_unit_anchors, mode_valid,
            neighbour_future_positions, neighbour_log_standard_deviation,
        ) = self.predictor.predict_every_round(batch)
        total, regression, heading, classification, speed = round_summed_prediction_loss(
            round_outputs, batch, selected_unit_anchors, mode_valid,
            self.heading_loss_weight, self.classification_loss_weight, self.speed_loss_weight,
        )
        neighbour_future = loss.neighbour_future_loss(
            neighbour_future_positions, neighbour_log_standard_deviation,
            batch["neighbour_future_positions"],
            batch["neighbour_future_mask"],
            batch["neighbour_history_mask"].any(dim=-1),
        )
        total = total + self.neighbour_future_loss_weight * neighbour_future
        sample_count = batch["agent_history"].shape[0]
        weight = torch.full((1,), float(sample_count), device=total.device, dtype=torch.float32)
        trajectories, heading_cosine_sine, _, _, confidence_logits, _, _ = round_outputs[-1]
        return {
            "sample_count": weight,
            "total": total[None] * weight,
            "regression": regression[None] * weight,
            "heading": heading[None] * weight,
            "classification": classification[None] * weight,
            "neighbour_future": neighbour_future[None] * weight,
            "speed": speed[None] * weight,
            "trajectories": trajectories,
            "heading_cosine_sine": heading_cosine_sine,
            "confidence_logits": confidence_logits,
            "mode_valid": mode_valid,
        }


def combine_step_outputs(outputs):
    sample_count = outputs["sample_count"].sum()
    return {
        name: (value.sum() / sample_count if name in LOSS_COMPONENT_NAMES else value)
        for name, value in outputs.items()
    }


LOSS_COMPONENT_NAMES = ("total", "regression", "heading", "classification", "neighbour_future", "speed")


class SampleSplittingDataParallel(nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        (batch,) = inputs
        parts = pipeline.split_batch_by_samples(batch, len(device_ids))
        scattered = [
            (({name: tensor.to(device_id, non_blocking=True) for name, tensor in part.items()},), {})
            for part, device_id in zip(parts, device_ids)
        ]
        return [pair[0] for pair in scattered], [pair[1] for pair in scattered]


def training_step_module(predictor, weights, device):
    step = TrainingStep(predictor, *weights)
    device_ids = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    if len(device_ids) > 1:
        return SampleSplittingDataParallel(step, device_ids=device_ids), len(device_ids)
    return step, 1


def checkpoint_state(predictor, optimizer, gradient_scaler, seed, completed_epochs, batch_index):
    return {
        "model_state": getattr(predictor, "_orig_mod", predictor).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": gradient_scaler.state_dict(),
        "completed_epochs": completed_epochs,
        "batch_index": batch_index,
        "seed": seed,
        "code_version": contract.STAGING_CODE_VERSION,
        "parameter_fingerprint": model.parameter_fingerprint(
            getattr(predictor, "_orig_mod", predictor).state_dict()
        ),
    }

def epochs_left_to_train(completed_epochs, requested_epochs):
    return range(completed_epochs, requested_epochs)

def accumulate_averaged_weights(running_average, model_state, averaged_count):
    if running_average is None:
        return {name: tensor.detach().clone().double() for name, tensor in model_state.items()}, 1
    for name, tensor in model_state.items():
        running_average[name] += (tensor.detach().double() - running_average[name]) / (
            averaged_count + 1
        )
    return running_average, averaged_count + 1

def save_checkpoint(checkpoint_path, previous_checkpoint_path, state):
    partial_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
    torch.save(state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_checkpoint_path)
    partial_path.replace(checkpoint_path)

def train_epoch(
    predictor, step_module, optimizer, batches, device, gradient_scaler,
    checkpoint_path, previous_checkpoint_path,
    checkpoint_every_seconds, epoch_index, seed,
    completed_steps_before_epoch, warmup_steps, total_optimiser_steps, decay_steps,
    gradient_clip_norm, training_start, stop_after_seconds, process_steps_before_epoch,
):
    step = step_module.module if isinstance(step_module, nn.DataParallel) else step_module
    heading_loss_weight = step.heading_loss_weight
    neighbour_future_loss_weight = step.neighbour_future_loss_weight
    speed_loss_weight = step.speed_loss_weight
    accumulator = metrics.MetricAccumulator()
    window_accumulator = metrics.MetricAccumulator()
    loss_sums = {
        "total": 0.0, "regression": 0.0, "heading": 0.0, "classification": 0.0,
        "neighbour_future": 0.0, "speed": 0.0,
    }
    window_loss_sums = dict.fromkeys(loss_sums, 0.0)
    seconds = {"data_wait": 0.0, "step": 0.0, "monitor": 0.0}
    batch_count = 0
    sample_count = 0
    polyline_slots = 0
    window_peak_tokens = 0
    non_finite_total_count = 0
    gradient_scaler_skip_count = 0
    clipped_step_count = 0
    window_winner_counts = torch.zeros(QUERY_COUNT, dtype=torch.long, device=device)
    wait_start = time.perf_counter()
    checkpoint_wait_start = time.perf_counter()
    for batch in batches:
        seconds["data_wait"] += time.perf_counter() - wait_start

        step_start = time.perf_counter()
        batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
        with torch.amp.autocast(device_type=device.type, enabled=gradient_scaler.is_enabled()):
            step_outputs = combine_step_outputs(step_module(batch))
            total = step_outputs["total"]
            regression = step_outputs["regression"]
            heading = step_outputs["heading"]
            classification = step_outputs["classification"]
            neighbour_future = step_outputs["neighbour_future"]
            speed = step_outputs["speed"]
            trajectories = step_outputs["trajectories"]
            heading_cosine_sine = step_outputs["heading_cosine_sine"]
            confidence_logits = step_outputs["confidence_logits"]
            mode_valid = step_outputs["mode_valid"]
        elapsed_seconds = time.perf_counter() - training_start
        process_steps = process_steps_before_epoch + batch_count
        learning_rate = scheduled_learning_rate(
            completed_steps_before_epoch + batch_count, warmup_steps,
            total_optimiser_steps, decay_steps,
            elapsed_seconds, stop_after_seconds,
            elapsed_seconds / process_steps if process_steps else None,
            process_steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.zero_grad()
        gradient_scaler.scale(total).backward()
        gradient_scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), gradient_clip_norm
        ))
        clipped_step_count += int(gradient_norm > gradient_clip_norm)
        gradient_scaler.step(optimizer)
        scale_before_update = gradient_scaler.get_scale()
        gradient_scaler.update()
        gradient_scaler_skip_count += int(gradient_scaler.get_scale() < scale_before_update)
        component_values = {
            "total": float(total.detach()),
            "regression": float(regression.detach()),
            "heading": float(heading.detach()),
            "classification": float(classification.detach()),
            "neighbour_future": float(neighbour_future.detach()),
            "speed": float(speed.detach()),
        }
        for name, value in component_values.items():
            loss_sums[name] += value
            window_loss_sums[name] += value
        non_finite_total_count += int(not math.isfinite(component_values["total"]))
        seconds["step"] += time.perf_counter() - step_start

        monitor_start = time.perf_counter()
        with torch.no_grad():
            accumulator.update(
                trajectories.detach().float(), confidence_logits.detach().float(),
                batch["future_positions"], batch["future_mask"], mode_valid,
            )
            window_accumulator.update(
                trajectories.detach().float(), confidence_logits.detach().float(),
                batch["future_positions"], batch["future_mask"], mode_valid,
            )
            window_winners = metrics.mean_distance_per_mode(
                trajectories.detach().float(), batch["future_positions"], batch["future_mask"],
                mode_valid,
            ).argmin(dim=1)
            window_winner_counts.scatter_add_(
                0, window_winners, torch.ones_like(window_winners)
            )
        seconds["monitor"] += time.perf_counter() - monitor_start

        batch_count += 1
        sample_count += batch["agent_history"].shape[0]
        chunk_slots = int(batch["max_polylines_in_batch"])
        polyline_slots += chunk_slots
        window_peak_tokens = max(
            window_peak_tokens, 1 + batch["neighbour_history"].shape[1] + chunk_slots
        )
        if batch_count % LOG_EVERY_BATCHES == 0:
            elapsed = sum(seconds.values())
            monitor = accumulator.results()
            window_monitor = window_accumulator.results()
            median_heading_norm = float(
                heading_cosine_sine.detach().float().norm(dim=-1).median()
            )
            winner_counts = window_winner_counts.cpu()
            never_win_count = int((winner_counts == 0).sum())
            winner_total = int(winner_counts.sum())
            cover_ninety = (
                int(
                    torch.searchsorted(
                        winner_counts.sort(descending=True).values.cumsum(0).to(torch.float64),
                        torch.tensor(0.9 * winner_total, dtype=torch.float64),
                    )
                    + 1
                )
                if winner_total
                else QUERY_COUNT
            )
            peak_gigabytes = (
                torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            )
            print(
                f"  batch {batch_count} | loss {loss_sums['total'] / batch_count:.4f} "
                f"(window {window_loss_sums['total'] / LOG_EVERY_BATCHES:.4f}) "
                f"reg {loss_sums['regression'] / batch_count:.4f} "
                f"(window {window_loss_sums['regression'] / LOG_EVERY_BATCHES:.4f}) | "
                f"ade_80step {monitor['min_ade']:.3f} (window {window_monitor['min_ade']:.3f}) "
                f"fde_80step {monitor['min_fde']:.3f} (window {window_monitor['min_fde']:.3f}) | "
                f"kept modes {window_monitor['mean_kept_modes']:.2f} "
                f"backfilled {100 * window_monitor['backfill_rate']:.0f}% "
                f"never-win {never_win_count}/{QUERY_COUNT} cover90 {cover_ninety} | "
                f"hdg/reg {heading_loss_weight * window_loss_sums['heading'] / max(window_loss_sums['regression'], 1e-12):.4f} "
                f"hdg norm {median_heading_norm:.4f} | "
                f"nbr/reg {neighbour_future_loss_weight * window_loss_sums['neighbour_future'] / max(window_loss_sums['regression'], 1e-12):.4f} "
                f"spd/reg {speed_loss_weight * window_loss_sums['speed'] / max(window_loss_sums['regression'], 1e-12):.4f} "
                f"| "
                f"non-finite {non_finite_total_count} skipped steps {gradient_scaler_skip_count} "
                f"clipped {clipped_step_count} | "
                f"lr {learning_rate:.3e} | "
                f"{sample_count / elapsed:.1f} samples/s | "
                f"wait {100 * seconds['data_wait'] / elapsed:.0f}% "
                f"step {100 * seconds['step'] / elapsed:.0f}% "
                f"monitor {100 * seconds['monitor'] / elapsed:.0f}% | "
                f"polylines/sample {polyline_slots / batch_count:.0f} "
                f"peak tokens {window_peak_tokens} | "
                f"peak {peak_gigabytes:.1f} GB",
                flush=True,
            )
            window_accumulator = metrics.MetricAccumulator()
            window_loss_sums = dict.fromkeys(loss_sums, 0.0)
            window_peak_tokens = 0
            window_winner_counts.zero_()
        if time.perf_counter() - checkpoint_wait_start >= checkpoint_every_seconds:
            if math.isfinite(component_values["total"]):
                save_checkpoint(
                    checkpoint_path, previous_checkpoint_path,
                    checkpoint_state(predictor, optimizer, gradient_scaler, seed, epoch_index, batch_count),
                )
            else:
                print(
                    f"batch {batch_count} total loss {component_values['total']},"
                    f" checkpoint {checkpoint_path} left as it was",
                    flush=True,
                )
            checkpoint_wait_start = time.perf_counter()
        if time.perf_counter() - training_start >= stop_after_seconds:
            if math.isfinite(component_values["total"]):
                save_checkpoint(
                    checkpoint_path, previous_checkpoint_path,
                    checkpoint_state(predictor, optimizer, gradient_scaler, seed, epoch_index, batch_count),
                )
            print(
                f"STOPPING ON THE CLOCK: {(time.perf_counter() - training_start) / 3600:.2f} h"
                f" elapsed against a --stop-after-seconds budget of {stop_after_seconds / 3600:.2f} h,"
                f" {batch_count} batches into epoch {epoch_index + 1} at learning rate"
                f" {learning_rate:.3e}; {checkpoint_path} holds this point.",
                flush=True,
            )
            averages = {name: value / max(batch_count, 1) for name, value in loss_sums.items()}
            return averages, accumulator.results(), seconds, True
        wait_start = time.perf_counter()
    averages = {name: value / max(batch_count, 1) for name, value in loss_sums.items()}
    return averages, accumulator.results(), seconds, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--heading-loss-weight", type=float, default=HEADING_LOSS_WEIGHT)
    parser.add_argument("--classification-loss-weight", type=float, default=CLASSIFICATION_LOSS_WEIGHT)
    parser.add_argument("--neighbour-future-loss-weight", type=float, default=NEIGHBOUR_FUTURE_LOSS_WEIGHT)
    parser.add_argument("--speed-loss-weight", type=float, default=SPEED_LOSS_WEIGHT)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--checkpoint-every-seconds", type=int, required=True)
    parser.add_argument("--stop-after-seconds", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--average-last-epochs", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--all-eligible-agents", action="store_true")
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_unit_anchors, anchor_counts = model.load_anchor_file(arguments.anchors)
    predictor = MotionPredictor(initial_unit_anchors, anchor_counts).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(predictor), lr=LEARNING_RATE)
    gradient_scaler = GradScaler(enabled=arguments.mixed_precision and device.type == "cuda")
    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"
    steps_per_epoch = optimiser_steps_per_epoch(
        scenario_paths, arguments.workers, arguments.batch_size,
        not arguments.all_eligible_agents,
    )
    total_optimiser_steps = steps_per_epoch * arguments.epochs
    decay_steps = steps_per_epoch * DECAY_EPOCHS
    print(
        f"{steps_per_epoch} optimiser steps per epoch,"
        f" {total_optimiser_steps} over {arguments.epochs} epochs,"
        f" peak learning rate {LEARNING_RATE} after {arguments.warmup_steps} warmup steps,"
        f" linear decay to zero over the last {decay_steps} steps or the last"
        f" {decay_steps} steps' worth of the --stop-after-seconds budget, whichever comes first,"
        f" anchors per type {anchor_counts.tolist()}",
        flush=True,
    )
    step_module, device_count = training_step_module(
        predictor,
        (
            arguments.heading_loss_weight, arguments.classification_loss_weight,
            arguments.neighbour_future_loss_weight, arguments.speed_loss_weight,
        ),
        device,
    )
    print(f"training on {device_count} {device.type} device(s)", flush=True)
    previous_checkpoint_path = arguments.checkpoint_path.with_suffix(
        arguments.checkpoint_path.suffix + ".previous"
    )

    completed_epochs = 0
    resume_path = arguments.checkpoint_path
    if arguments.resume and not resume_path.exists() and previous_checkpoint_path.exists():
        resume_path = previous_checkpoint_path
        print(f"{arguments.checkpoint_path} missing, resuming from {resume_path}", flush=True)
    if arguments.resume and resume_path.exists():
        checkpoint = model.load_checkpoint_state(resume_path, map_location=device)
        predictor.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "gradient_scaler_state" in checkpoint:
            gradient_scaler.load_state_dict(checkpoint["gradient_scaler_state"])
        completed_epochs = checkpoint["completed_epochs"]
        interrupted_batches = checkpoint["batch_index"]
        print(
            f"resuming {resume_path}: {completed_epochs} epochs complete"
            + (
                ""
                if interrupted_batches is None
                else f", epoch {completed_epochs + 1} was {interrupted_batches} batches in"
                f" when it was checkpointed and restarts from its first batch"
            ),
            flush=True,
        )

    remaining_epochs = epochs_left_to_train(completed_epochs, arguments.epochs)
    if not remaining_epochs:
        print(
            f"NOTHING TO TRAIN: {arguments.checkpoint_path} already holds {completed_epochs}"
            f" completed epochs and --epochs is {arguments.epochs}."
            f" Raise --epochs above {completed_epochs} to train further.",
            flush=True,
        )
        return

    if arguments.compile:
        predictor = torch.compile(predictor, dynamic=True)

    training_start = time.perf_counter()
    averaged_weights = None
    averaged_epoch_count = 0
    averaged_checkpoint_path = arguments.checkpoint_path.with_name(
        arguments.checkpoint_path.stem + "_averaged" + arguments.checkpoint_path.suffix
    )
    first_averaged_epoch = arguments.epochs - arguments.average_last_epochs
    if completed_epochs > first_averaged_epoch:
        print(
            f"AVERAGING WINDOW TRUNCATED BY RESUME: --average-last-epochs asked for"
            f" {arguments.average_last_epochs} epochs from epoch {first_averaged_epoch + 1}, but"
            f" this process starts at epoch {completed_epochs + 1}, so {averaged_checkpoint_path}"
            f" will average only the {arguments.epochs - completed_epochs} epochs this process"
            f" trains.",
            flush=True,
        )
    process_steps_before_epoch = 0
    for epoch_index in remaining_epochs:
        batches = pipeline.batches(
            scenario_paths, arguments.workers, arguments.batch_size,
            arguments.prefetch, arguments.seed + epoch_index,
            not arguments.all_eligible_agents,
        )
        averages, monitor, seconds, stopped_on_the_clock = train_epoch(
            predictor, step_module, optimizer, batches, device, gradient_scaler,
            arguments.checkpoint_path, previous_checkpoint_path,
            arguments.checkpoint_every_seconds, epoch_index, arguments.seed,
            epoch_index * steps_per_epoch,
            arguments.warmup_steps, total_optimiser_steps, decay_steps,
            arguments.gradient_clip_norm,
            training_start, arguments.stop_after_seconds, process_steps_before_epoch,
        )
        process_steps_before_epoch += steps_per_epoch
        print(
            f"epoch {epoch_index + 1}/{arguments.epochs}"
            f"{' (partial, stopped on the clock)' if stopped_on_the_clock else ''} | "
            f"loss {averages['total']:.4f} (reg {averages['regression']:.4f}"
            f" + hdg {averages['heading']:.4f}"
            f" + cls {averages['classification']:.4f}"
            f" + nbr {averages['neighbour_future']:.4f}"
            f" + spd {averages['speed']:.4f}"
            f") | "
            f"ade_80step {monitor['min_ade']:.4f} | fde_80step {monitor['min_fde']:.4f} | "
            f"kept modes {monitor['mean_kept_modes']:.2f}"
            f" | backfilled {100 * monitor['backfill_rate']:.0f}% | "
            f"data_wait {seconds['data_wait']:.0f} s · step {seconds['step']:.0f} s"
            f" · monitor {seconds['monitor']:.0f} s",
            flush=True,
        )
        if stopped_on_the_clock:
            return
        if not math.isfinite(averages["total"]):
            print(
                f"epoch {epoch_index + 1} mean total loss {averages['total']},"
                f" checkpoint {arguments.checkpoint_path} left as it was",
                flush=True,
            )
            continue
        epoch_state = checkpoint_state(
            predictor, optimizer, gradient_scaler, arguments.seed, epoch_index + 1, None
        )
        save_checkpoint(arguments.checkpoint_path, previous_checkpoint_path, epoch_state)
        if epoch_index + 1 > arguments.epochs - arguments.average_last_epochs:
            averaged_weights, averaged_epoch_count = accumulate_averaged_weights(
                averaged_weights, epoch_state["model_state"], averaged_epoch_count
            )
            averaged_state = dict(epoch_state)
            averaged_state["model_state"] = {
                name: tensor.to(epoch_state["model_state"][name].dtype)
                for name, tensor in averaged_weights.items()
            }
            torch.save(averaged_state, averaged_checkpoint_path)
            print(
                f"averaged the last {averaged_epoch_count} epoch checkpoints into"
                f" {averaged_checkpoint_path}",
                flush=True,
            )
        elapsed_seconds = time.perf_counter() - training_start
        if elapsed_seconds >= arguments.stop_after_seconds:
            print(
                f"STOPPING EARLY: {elapsed_seconds / 3600:.2f} h elapsed against a"
                f" --stop-after-seconds budget of {arguments.stop_after_seconds / 3600:.2f} h."
                f" {epoch_index + 1} of {arguments.epochs} epochs are complete and"
                f" {arguments.checkpoint_path} holds them.",
                flush=True,
            )
            return

if __name__ == "__main__":
    main()
