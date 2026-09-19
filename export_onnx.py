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


def milliseconds_per_batch(run_once, batches):
    for _ in range(WARMUP_RUNS):
        run_once(batches[0])
    durations = []
    for batch in batches:
        start = time.perf_counter()
        run_once(batch)
        durations.append(1000 * (time.perf_counter() - start))
    return durations


def export_and_measure_bucket(
    predictor, batches, onnx_path, thread_count, device
):
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
        str(onnx_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={"map_chunk_signal_history": {1: "map_chunks"}},
        opset_version=18,
    )
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = thread_count
    session_options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        session_options,
        providers=[
            (
                "CUDAExecutionProvider"
                if device.type == "cuda"
                else "CPUExecutionProvider"
            )
        ],
    )
    device_predictor = PositionalInputs(predictor).to(device)

    def run_onnx(fixed_batch):
        return session.run(
            OUTPUT_NAMES,
            {name: fixed_batch[name].numpy() for name in INPUT_NAMES},
        )

    def run_torch(fixed_batch):
        with torch.no_grad():
            trajectories, confidence_logits = device_predictor(
                *(
                    fixed_batch[name].to(device)
                    for name in INPUT_NAMES
                )
            )
        return trajectories.cpu(), confidence_logits.cpu()

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
        f"{onnx_path.name}: {len(fixed_batches)} batches padded to"
        f" {neighbour_count} neighbours, {chunk_count} map chunks,"
        f" {dot_count} map dots"
    )
    return (
        padding_gap,
        export_gap,
        milliseconds_per_batch(run_torch, fixed_batches),
        milliseconds_per_batch(run_onnx, fixed_batches),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--scenarios", type=int, required=True)
    parser.add_argument("--agents-per-batch", type=int, required=True)
    parser.add_argument("--buckets", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], required=True
    )
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)

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
    batches = sorted(
        (
            batch
            for batch in pipeline.batches(
                scenario_paths,
                0,
                arguments.agents_per_batch,
                None,
                0,
                True,
            )
            if len(batch["agent_history"])
            == arguments.agents_per_batch
        ),
        key=lambda batch: batch["neighbour_history"].shape[1]
        + batch["map_chunk_signal_history"].shape[1],
    )
    assert (
        len(batches) >= arguments.buckets
    ), f"{len(batches)} full batches cannot fill {arguments.buckets} buckets"

    padding_gaps, export_gaps = [], []
    torch_milliseconds, onnx_milliseconds = [], []
    for bucket_index, bucket_positions in enumerate(
        np.array_split(np.arange(len(batches)), arguments.buckets)
    ):
        padding_gap, export_gap, torch_durations, onnx_durations = (
            export_and_measure_bucket(
                predictor,
                [batches[position] for position in bucket_positions],
                arguments.onnx_path.with_name(
                    f"{arguments.onnx_path.stem}_{bucket_index}"
                    f"{arguments.onnx_path.suffix}"
                ),
                arguments.threads,
                torch.device(arguments.device),
            )
        )
        padding_gaps.append(padding_gap)
        export_gaps.append(export_gap)
        torch_milliseconds.extend(torch_durations)
        onnx_milliseconds.extend(onnx_durations)

    print(
        f"padding moves a trajectory by at most"
        f" {max(padding_gaps):.2e} m"
    )
    print(
        f"onnx differs from torch by at most {max(export_gaps):.2e} m"
    )
    print(
        f"median per batch of {arguments.agents_per_batch} agents on"
        f" {arguments.device}, {arguments.threads} threads:"
        f" torch {statistics.median(torch_milliseconds):.1f} ms,"
        f" onnxruntime {statistics.median(onnx_milliseconds):.1f} ms"
    )


if __name__ == "__main__":
    main()
