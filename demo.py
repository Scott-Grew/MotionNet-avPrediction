import womd.runtime_env
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from womd import baseline, contract, model, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "derivation"))
from plot_run_report import (
    draw_scene,
    frame,
    pick_turning_sample,
    scene_half_width,
    type_index_and_anchors,
)


def load_predictor(checkpoint_path, anchors_path):
    unit_anchors, anchor_counts = model.load_anchor_file(anchors_path)
    predictor = model.MotionPredictor(unit_anchors, anchor_counts)
    predictor.load_state_dict(model.load_checkpoint_state(checkpoint_path)["model_state"])
    return predictor.eval(), unit_anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("anchors_path", type=Path)
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()

    predictor, unit_anchors = load_predictor(
        arguments.checkpoint_path, arguments.anchors_path
    )
    sample = pick_turning_sample(arguments.staged_directory)
    batch = pipeline.collate_samples([sample])
    with torch.no_grad():
        trajectories, confidence_logits = predictor(batch)
        pruned, pruned_logits = model.prune_modes_batched(trajectories, confidence_logits)
        constant_velocity, _ = baseline.constant_velocity(batch)
    pruned = pruned[0].numpy()
    weights = torch.softmax(pruned_logits[0], dim=-1).numpy()
    constant_velocity = constant_velocity[0, 0].numpy()
    future = sample["future_positions"]
    _, anchors = type_index_and_anchors(sample, unit_anchors.numpy())
    half_width = scene_half_width(sample, anchors)

    figure, axis = plt.subplots(figsize=(8.5, 8.5))
    draw_scene(axis, sample)
    order = np.argsort(weights)
    colormap = plt.get_cmap("Blues")
    for rank, mode_index in enumerate(order):
        shade = colormap(0.35 + 0.6 * (rank + 1) / len(order))
        axis.plot(pruned[mode_index, :, 0], pruned[mode_index, :, 1], c=shade, linewidth=2.0)
        if weights[mode_index] >= 0.01:
            axis.annotate(
                f"{weights[mode_index]:.0%}",
                pruned[mode_index, -1],
                fontsize=8,
                color=shade,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axis.plot(
        constant_velocity[:, 0], constant_velocity[:, 1],
        c="#1baf7a", linewidth=1.8, linestyle=(0, (5, 4)),
    )
    axis.plot(future[:, 0], future[:, 1], c="#238b45", linewidth=2.6, zorder=6)
    frame(
        axis,
        (
            f"{arguments.checkpoint_path.stem} · scenario {sample['scenario_id']}"
            f" track {sample['track_id']}"
            f" · 6 futures (blue) · logged (green) · constant velocity (dashed)"
        ),
        half_width,
    )
    figure.tight_layout()
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output_path, dpi=150)
    plt.close(figure)
    print(f"wrote {arguments.output_path}")


if __name__ == "__main__":
    main()
