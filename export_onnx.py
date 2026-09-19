import womd.runtime_env
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from womd import model, pipeline

INPUT_NAMES = [
    "agent_history",
    "agent_history_mask",
    "agent_signal_history",
    "neighbour_history",
    "neighbour_history_mask",
    "neighbour_signal_history",
    "map_rows",
    "map_dot_polyline_slot",
    "map_chunk_signal_history",
]
OUTPUT_NAMES = ["trajectories", "confidence_logits"]
WARMUP_RUNS = 2


class PositionalInputs(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, *tensors):
        return self.predictor(dict(zip(INPUT_NAMES, tensors)))


def padded_along(tensor, dimension, length):
    padding_shape = list(tensor.shape)
    padding_shape[dimension] = length - tensor.shape[dimension]
    return torch.cat(
        [tensor, tensor.new_zeros(padding_shape)], dim=dimension
    )


def padded_to_fixed_shape(
    batch, neighbour_count, chunk_count, dot_count
):
    batch_chunk_count = batch["map_chunk_signal_history"].shape[1]
    slot = batch["map_dot_polyline_slot"]
    fixed_slot = (
        slot // batch_chunk_count
    ) * chunk_count + slot % batch_chunk_count
    repeated_last_dot = dot_count - len(slot)
    fixed = {
        name: batch[name]
        for name in INPUT_NAMES
        if name.startswith("agent_")
    }
    for name in (
        "neighbour_history",
        "neighbour_history_mask",
        "neighbour_signal_history",
    ):
        fixed[name] = padded_along(batch[name], 1, neighbour_count)
    fixed["map_chunk_signal_history"] = padded_along(
        batch["map_chunk_signal_history"], 1, chunk_count
    )
    fixed["map_rows"] = torch.cat(
        [
            batch["map_rows"],
            batch["map_rows"][-1:].expand(repeated_last_dot, -1),
        ]
    )
    fixed["map_dot_polyline_slot"] = torch.cat(
        [fixed_slot, fixed_slot[-1:].expand(repeated_last_dot)]
    )
    return fixed


def median_milliseconds(run_once, batches):
    for batch in batches[:WARMUP_RUNS]:
        run_once(batch)
    durations = []
    for batch in batches:
        start = time.perf_counter()
        run_once(batch)
        durations.append(1000 * (time.perf_counter() - start))
    return statistics.median(durations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--scenarios", type=int, required=True)
    parser.add_argument("--agents-per-batch", type=int, required=True)
    arguments = parser.parse_args()

    predictor = model.MotionPredictor(
        model.load_anchor_file(arguments.anchors)
    )
    if arguments.checkpoint is not None:
        predictor.load_state_dict(
            model.load_checkpoint_state(arguments.checkpoint)[
                "model_state"
            ]
        )
    predictor.eval()

    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))[
        : arguments.scenarios
    ]
    batches = [
        batch
        for batch in pipeline.batches(
            scenario_paths,
            0,
            arguments.agents_per_batch,
            None,
            0,
            True,
        )
        if len(batch["agent_history"]) == arguments.agents_per_batch
    ]
    assert batches, "no full batch in the chosen scenarios"
    neighbour_count = max(
        batch["neighbour_history"].shape[1] for batch in batches
    )
    chunk_count = max(
        batch["map_chunk_signal_history"].shape[1]
        for batch in batches
    )
    dot_count = max(len(batch["map_rows"]) for batch in batches)
    fixed_batches = [
        padded_to_fixed_shape(
            batch, neighbour_count, chunk_count, dot_count
        )
        for batch in batches
    ]

    with torch.no_grad():
        padding_gap = max(
            float(
                (predictor(batch)[0] - predictor(fixed_batch)[0])
                .abs()
                .max()
            )
            for batch, fixed_batch in zip(batches, fixed_batches)
        )

    torch.onnx.export(
        PositionalInputs(predictor),
        tuple(fixed_batches[0][name] for name in INPUT_NAMES),
        str(arguments.onnx_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={"map_chunk_signal_history": {1: "map_chunks"}},
        opset_version=18,
    )
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = torch.get_num_threads()
    session_options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(arguments.onnx_path),
        session_options,
        providers=["CPUExecutionProvider"],
    )

    def run_onnx(fixed_batch):
        return session.run(
            OUTPUT_NAMES,
            {name: fixed_batch[name].numpy() for name in INPUT_NAMES},
        )

    def run_torch(fixed_batch):
        with torch.no_grad():
            return predictor(fixed_batch)

    export_gap = max(
        float(
            np.abs(
                run_torch(fixed_batch)[0].numpy()
                - run_onnx(fixed_batch)[0]
            ).max()
        )
        for fixed_batch in fixed_batches
    )
    print(
        f"{len(fixed_batches)} batches of"
        f" {arguments.agents_per_batch} agents, padded to"
        f" {neighbour_count} neighbours, {chunk_count} map chunks,"
        f" {dot_count} map dots"
    )
    print(
        f"padding moves a trajectory by at most {padding_gap:.2e} m"
    )
    print(f"onnx differs from torch by at most {export_gap:.2e} m")
    print(
        f"torch {median_milliseconds(run_torch, fixed_batches):.1f} ms"
        f" per batch, onnxruntime"
        f" {median_milliseconds(run_onnx, fixed_batches):.1f} ms per"
        f" batch, {torch.get_num_threads()} thread"
    )


if __name__ == "__main__":
    main()
