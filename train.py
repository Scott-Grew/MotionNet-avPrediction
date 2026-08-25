import womd.runtime_env
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from womd import contract, loader, loss, metrics, model, pipeline
from womd.model import QUERY_COUNT, MotionPredictor

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
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

def scheduled_learning_rate(process_steps, warmup_steps, learning_rate=LEARNING_RATE):
    if process_steps < warmup_steps:
        return learning_rate * (process_steps + 1) / warmup_steps
    return learning_rate

def training_losses(predictor, batch):
    trajectories, log_standard_deviation, confidence_logits, unit_anchors = predictor.predict(batch)
    total, regression, classification = loss.prediction_loss(
        trajectories, log_standard_deviation, confidence_logits,
        batch["future_positions"], batch["future_mask"], unit_anchors,
    )
    return total, regression, classification, trajectories, confidence_logits

def checkpoint_state(predictor, optimizer, gradient_scaler, seed, completed_epochs, batch_index):
    return {
        "model_state": predictor.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": gradient_scaler.state_dict(),
        "completed_epochs": completed_epochs,
        "batch_index": batch_index,
        "seed": seed,
        "code_version": contract.STAGING_CODE_VERSION,
        "parameter_fingerprint": model.parameter_fingerprint(predictor.state_dict()),
    }

def epochs_left_to_train(completed_epochs, requested_epochs):
    return range(completed_epochs, requested_epochs)

def save_checkpoint(checkpoint_path, previous_checkpoint_path, state):
    partial_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
    torch.save(state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_checkpoint_path)
    partial_path.replace(checkpoint_path)

def train_epoch(
    predictor, optimizer, batches, device, gradient_scaler,
    checkpoint_path, previous_checkpoint_path, checkpoint_every_seconds,
    epoch_index, seed, warmup_steps, gradient_clip_norm, process_steps_before_epoch, learning_rate,
):
    accumulator = metrics.MetricAccumulator()
    window_accumulator = metrics.MetricAccumulator()
    loss_sums = {"total": 0.0, "regression": 0.0, "classification": 0.0}
    window_loss_sums = dict.fromkeys(loss_sums, 0.0)
    seconds = {"data_wait": 0.0, "step": 0.0, "monitor": 0.0}
    batch_count = 0
    sample_count = 0
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
            total, regression, classification, trajectories, confidence_logits = training_losses(
                predictor, batch
            )
        step_learning_rate = scheduled_learning_rate(
            process_steps_before_epoch + batch_count, warmup_steps, learning_rate
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = step_learning_rate
        optimizer.zero_grad()
        gradient_scaler.scale(total).backward()
        gradient_scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip_norm))
        clipped_step_count += int(gradient_norm > gradient_clip_norm)
        gradient_scaler.step(optimizer)
        scale_before_update = gradient_scaler.get_scale()
        gradient_scaler.update()
        gradient_scaler_skip_count += int(gradient_scaler.get_scale() < scale_before_update)
        component_values = {
            "total": float(total.detach()),
            "regression": float(regression.detach()),
            "classification": float(classification.detach()),
        }
        for name, value in component_values.items():
            loss_sums[name] += value
            window_loss_sums[name] += value
        non_finite_total_count += int(not math.isfinite(component_values["total"]))
        seconds["step"] += time.perf_counter() - step_start

        monitor_start = time.perf_counter()
        with torch.no_grad():
            for tracker in (accumulator, window_accumulator):
                tracker.update(
                    trajectories.detach().float(), confidence_logits.detach().float(),
                    batch["future_positions"], batch["future_mask"],
                )
            window_winners = metrics.mean_distance_per_mode(
                trajectories.detach().float(), batch["future_positions"], batch["future_mask"],
            ).argmin(dim=1)
            window_winner_counts.scatter_add_(0, window_winners, torch.ones_like(window_winners))
        seconds["monitor"] += time.perf_counter() - monitor_start

        batch_count += 1
        sample_count += batch["agent_history"].shape[0]
        if batch_count % LOG_EVERY_BATCHES == 0:
            elapsed = sum(seconds.values())
            monitor = accumulator.results()
            window_monitor = window_accumulator.results()
            never_win_count = int((window_winner_counts == 0).sum())
            peak_gigabytes = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(
                f"  batch {batch_count} | loss {loss_sums['total'] / batch_count:.4f} "
                f"(window {window_loss_sums['total'] / LOG_EVERY_BATCHES:.4f}) "
                f"reg {loss_sums['regression'] / batch_count:.4f} "
                f"cls {loss_sums['classification'] / batch_count:.4f} | "
                f"ade_80step {monitor['min_ade']:.3f} (window {window_monitor['min_ade']:.3f}) "
                f"fde_80step {monitor['min_fde']:.3f} (window {window_monitor['min_fde']:.3f}) | "
                f"kept modes {window_monitor['mean_kept_modes']:.2f} "
                f"backfilled {100 * window_monitor['backfill_rate']:.0f}% "
                f"never-win {never_win_count}/{QUERY_COUNT} | "
                f"non-finite {non_finite_total_count} skipped steps {gradient_scaler_skip_count} "
                f"clipped {clipped_step_count} | lr {step_learning_rate:.3e} | "
                f"{sample_count / elapsed:.1f} samples/s | "
                f"wait {100 * seconds['data_wait'] / elapsed:.0f}% "
                f"step {100 * seconds['step'] / elapsed:.0f}% "
                f"monitor {100 * seconds['monitor'] / elapsed:.0f}% | peak {peak_gigabytes:.1f} GB",
                flush=True,
            )
            window_accumulator = metrics.MetricAccumulator()
            window_loss_sums = dict.fromkeys(loss_sums, 0.0)
            window_winner_counts.zero_()
        if time.perf_counter() - checkpoint_wait_start >= checkpoint_every_seconds:
            if math.isfinite(component_values["total"]):
                save_checkpoint(
                    checkpoint_path, previous_checkpoint_path,
                    checkpoint_state(predictor, optimizer, gradient_scaler, seed, epoch_index, batch_count),
                )
            checkpoint_wait_start = time.perf_counter()
        wait_start = time.perf_counter()
    averages = {name: value / max(batch_count, 1) for name, value in loss_sums.items()}
    return averages, accumulator.results(), seconds

def main():
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--all-eligible-agents", action="store_true")
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = MotionPredictor(model.load_anchor_file(arguments.anchors)).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(predictor), lr=arguments.learning_rate)
    gradient_scaler = GradScaler(enabled=arguments.mixed_precision and device.type == "cuda")
    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"
    steps_per_epoch = optimiser_steps_per_epoch(
        scenario_paths, arguments.workers, arguments.batch_size, not arguments.all_eligible_agents,
    )
    print(
        f"{steps_per_epoch} optimiser steps per epoch, {steps_per_epoch * arguments.epochs} over"
        f" {arguments.epochs} epochs, learning rate {arguments.learning_rate} held after"
        f" {arguments.warmup_steps} warmup steps",
        flush=True,
    )
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
        print(f"resuming {resume_path}: {completed_epochs} epochs complete", flush=True)

    remaining_epochs = epochs_left_to_train(completed_epochs, arguments.epochs)
    if not remaining_epochs:
        print(
            f"NOTHING TO TRAIN: {arguments.checkpoint_path} already holds {completed_epochs}"
            f" completed epochs and --epochs is {arguments.epochs}.",
            flush=True,
        )
        return

    training_start = time.perf_counter()
    process_steps_before_epoch = 0
    for epoch_index in remaining_epochs:
        batches = pipeline.batches(
            scenario_paths, arguments.workers, arguments.batch_size,
            arguments.prefetch, arguments.seed + epoch_index,
            not arguments.all_eligible_agents,
        )
        averages, monitor, seconds = train_epoch(
            predictor, optimizer, batches, device, gradient_scaler,
            arguments.checkpoint_path, previous_checkpoint_path,
            arguments.checkpoint_every_seconds, epoch_index, arguments.seed,
            arguments.warmup_steps, arguments.gradient_clip_norm, process_steps_before_epoch,
            arguments.learning_rate,
        )
        process_steps_before_epoch += steps_per_epoch
        print(
            f"epoch {epoch_index + 1}/{arguments.epochs} | "
            f"loss {averages['total']:.4f} (reg {averages['regression']:.4f}"
            f" + cls {averages['classification']:.4f}) | "
            f"ade_80step {monitor['min_ade']:.4f} | fde_80step {monitor['min_fde']:.4f} | "
            f"kept modes {monitor['mean_kept_modes']:.2f} | backfilled {100 * monitor['backfill_rate']:.0f}% | "
            f"data_wait {seconds['data_wait']:.0f} s · step {seconds['step']:.0f} s"
            f" · monitor {seconds['monitor']:.0f} s",
            flush=True,
        )
        if not math.isfinite(averages["total"]):
            print(f"epoch {epoch_index + 1} mean total loss {averages['total']}, checkpoint left as it was", flush=True)
            continue
        save_checkpoint(
            arguments.checkpoint_path, previous_checkpoint_path,
            checkpoint_state(predictor, optimizer, gradient_scaler, arguments.seed, epoch_index + 1, None),
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
