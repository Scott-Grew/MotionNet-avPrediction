import womd.runtime_env
import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

import train
from womd import baseline, contract, loss, metrics, model, pipeline
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
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    unit_anchors, anchor_counts = model.load_anchor_file(arguments.anchors_path)
    predictor = MotionPredictor(unit_anchors, anchor_counts)
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    scaler = train.GradScaler(enabled=arguments.mixed_precision)
    scenario_paths = first_scenario_paths(
        arguments.staged_directory, arguments.batches * arguments.batch_size
    )
    if len(scenario_paths) < arguments.batch_size:
        raise SystemExit(f"{arguments.staged_directory} has fewer than one batch of scenarios")

    scaler_skips = 0
    winner_counts = torch.zeros(QUERY_COUNT, dtype=torch.long)
    logged_end_norms = []
    predicted_end_norms = []
    null_end_norms = []
    for batch_index, batch in enumerate(pipeline.batches(
        scenario_paths, 0, arguments.batch_size, 0, arguments.seed, True
    )):
        if batch_index >= arguments.batches:
            break
        with torch.amp.autocast(device_type="cpu", enabled=arguments.mixed_precision):
            (
                round_outputs, selected_unit_anchors, mode_valid,
                neighbour_future_positions, neighbour_log_standard_deviation,
            ) = predictor.predict_every_round(batch)
            total, regression, heading, classification, speed = train.round_summed_prediction_loss(
                round_outputs, batch, selected_unit_anchors, mode_valid,
                train.HEADING_LOSS_WEIGHT,
                train.CLASSIFICATION_LOSS_WEIGHT,
                train.SPEED_LOSS_WEIGHT,
            )
            (
                trajectories, heading_cosine_sine, position_log_standard_deviation,
                heading_log_standard_deviation, confidence_logits, predicted_speed,
            ) = round_outputs[-1]
            neighbour_future = loss.neighbour_future_loss(
                neighbour_future_positions, neighbour_log_standard_deviation,
                batch["neighbour_future_positions"],
                batch["neighbour_future_mask"],
                batch["neighbour_history_mask"].any(dim=-1),
            )
            total = total + train.NEIGHBOUR_FUTURE_LOSS_WEIGHT * neighbour_future
        for name, tensor in (
            ("total", total),
            ("trajectories", trajectories),
            ("heading", heading_cosine_sine),
            ("position_log_sigma", position_log_standard_deviation),
            ("heading_log_sigma", heading_log_standard_deviation),
            ("speed", predicted_speed),
            ("neighbour", neighbour_future_positions),
        ):
            if not tensor.isfinite().all():
                raise SystemExit(f"non-finite {name} at batch {batch_index}")
        optimizer.zero_grad()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), arguments.gradient_clip_norm
        ))
        if not math.isfinite(gradient_norm):
            raise SystemExit(f"non-finite gradient norm at batch {batch_index}")
        endpoints = trajectories.detach()[:, :, -1]
        if torch.allclose(endpoints[:, :1], endpoints, atol=1e-4):
            raise SystemExit(f"all modes collapsed to one point at batch {batch_index}")
        if (
            position_log_standard_deviation == model.MINIMUM_LOG_STANDARD_DEVIATION
        ).all() or (
            position_log_standard_deviation == model.MAXIMUM_LOG_STANDARD_DEVIATION
        ).all():
            raise SystemExit(f"position σ dead (all at a clamp) at batch {batch_index}")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            scaler_skips += 1

        scoreable = batch["future_mask"].any(dim=-1)
        winners = metrics.mean_distance_per_mode(
            trajectories.detach().float(), batch["future_positions"], batch["future_mask"],
            mode_valid,
        ).argmin(dim=1)
        winner_counts.scatter_add_(0, winners.cpu(), torch.ones_like(winners.cpu()))
        predicted_end_norms.append(
            trajectories.detach().float()[scoreable, :, -1].norm(dim=-1).median()
        )
        logged_end_norms.append(batch["future_positions"][scoreable, -1].norm(dim=-1).median())
        null_trajectories, _ = baseline.constant_velocity(batch)
        null_end_norms.append(null_trajectories.detach()[scoreable, 0, -1].norm(dim=-1).median())
        print(
            f"batch {batch_index} loss {float(total):.4f} "
            f"reg {float(regression):.4f} cls {float(classification):.4f} "
            f"hdg {float(heading):.4f} spd {float(speed):.4f} "
            f"grad {gradient_norm:.1f}",
            flush=True,
        )

    if not predicted_end_norms:
        raise SystemExit("no batches ran")
    never_win = int((winner_counts == 0).sum())
    predicted = float(torch.stack(predicted_end_norms).median())
    logged = float(torch.stack(logged_end_norms).median())
    null = float(torch.stack(null_end_norms).median())
    print(
        f"never-win {never_win}/{QUERY_COUNT} scaler-skips {scaler_skips} "
        f"median |end| pred {predicted:.2f} m logged {logged:.2f} m "
        f"cv {null:.2f} m pred/logged {predicted / max(logged, 1e-6):.2f}",
        flush=True,
    )
    if never_win == QUERY_COUNT:
        raise SystemExit("every mode never won: assignment is dead")
    if scaler_skips == arguments.batches:
        raise SystemExit("GradScaler skipped every step")

if __name__ == "__main__":
    main()
