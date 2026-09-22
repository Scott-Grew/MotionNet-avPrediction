"""Exports the trained model to ONNX, one graph per bucket of scene sizes.

Checks each exported graph against torch and times both.
"""
from __future__ import annotations

import womd.runtime_env
import argparse
import statistics
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import onnxruntime
import torch

from womd import loader, model, pipeline
from womd.checkpoint import load_anchor_file, load_checkpoint_state
from womd.loader import MapArrays, SceneArrays, SceneBatch, TargetArrays

# The graph inputs, one per batch array the model reads, as (group, field)
# in the order the exported graph takes them.
INPUT_FIELDS = ([("scene", name) for name in SceneArrays._fields] +
                [("map", name) for name in MapArrays._fields] +
                [("targets", name)
                 for name in TargetArrays._fields
                 if not name.startswith("future")])
INPUT_NAMES = [f"{group}_{name}" for group, name in INPUT_FIELDS]
OUTPUT_NAMES = ["trajectories", "confidence_logits"]
WARMUP_RUNS = 2


def graph_inputs(batch: SceneBatch) -> tuple[torch.Tensor, ...]:
    """The batch's arrays in INPUT_FIELDS order."""
    return tuple(
        getattr(getattr(batch, group), name) for group, name in INPUT_FIELDS)


def batch_from_graph_inputs(tensors: tuple[torch.Tensor, ...]) -> SceneBatch:
    """The SceneBatch behind positional tensors in INPUT_FIELDS order, with
    no future fields.
    """
    fields = {"scene": {}, "map": {}, "targets": {}}
    for (group, name), tensor in zip(INPUT_FIELDS, tensors):
        fields[group][name] = tensor
    return SceneBatch(
        scene=SceneArrays(**fields["scene"]),
        map=MapArrays(**fields["map"]),
        targets=TargetArrays(**fields["targets"]),
    )


class PositionalInputs(torch.nn.Module):
    """Wraps the model so ONNX export sees positional tensor arguments instead
    of the SceneBatch the model normally takes as input.
    """

    def __init__(self, predictor: torch.nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *tensors:
                torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor(batch_from_graph_inputs(tensors))


def padded_along(tensor: torch.Tensor, dimension: int,
                 length: int) -> torch.Tensor:
    """Pads a tensor with zeros along one dimension up to length; the added rows
    are absent tokens, not real agents or map chunks.
    """
    padding_shape = list(tensor.shape)
    padding_shape[dimension] = length - tensor.shape[dimension]
    return torch.cat([tensor, tensor.new_zeros(padding_shape)], dim=dimension)


def last_row_repeated_to(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Pads a tensor up to length by repeating its last row, which leaves max-
    pooled chunk tokens and real predictions unchanged.
    """
    return torch.cat([
        tensor,
        tensor[-1:].expand(length - len(tensor), *tensor.shape[1:]),
    ])


def padded_to_fixed_shape(batch: SceneBatch, *, agent_count: int,
                          chunk_count: int, dot_count: int,
                          target_count: int) -> SceneBatch:
    """Pads one scene batch to fixed agent, chunk, dot and target counts so
    differently sized scenes can share one exported graph.
    """
    batch_agent_count = batch.scene.agent_history.shape[1]
    batch_chunk_count = batch.map.chunk_signal_history.shape[1]
    slot = batch.map.dot_chunk_slot
    scene = SceneArrays(
        **{
            name: padded_along(array, 1, agent_count)
            for name, array in batch.scene._asdict().items()
        })
    # A slot is scene index * chunk count + chunk index, so the stride
    # changes when the chunk count is padded.
    scene_map = MapArrays(
        rows=last_row_repeated_to(batch.map.rows, dot_count),
        dot_chunk_slot=last_row_repeated_to(
            (slot // batch_chunk_count) * chunk_count +
            slot % batch_chunk_count, dot_count),
        chunk_signal_history=padded_along(batch.map.chunk_signal_history, 1,
                                          chunk_count),
    )
    # The token axis holds the scene agents and then the map chunks, so each
    # half is padded separately before being joined back together.
    target_arrays = batch.targets._asdict()
    for name in ("token_visible", "token_pose"):
        tokens = target_arrays[name]
        target_arrays[name] = torch.cat(
            [
                padded_along(tokens[:, :batch_agent_count], 1, agent_count),
                padded_along(tokens[:, batch_agent_count:], 1, chunk_count),
            ],
            dim=1,
        )
    targets = TargetArrays(
        **{
            name: None if array is None else last_row_repeated_to(
                array, target_count) for name, array in target_arrays.items()
        })
    return SceneBatch(scene=scene, map=scene_map, targets=targets)


def milliseconds_per_scene(run_once: Callable[[SceneBatch], object],
                           fixed_batches: list[SceneBatch]) -> list[float]:
    """Times run_once over each fixed batch after a few warm-up calls, returning
    one duration in milliseconds per scene.
    """
    for _ in range(WARMUP_RUNS):
        run_once(fixed_batches[0])
    durations = []
    for fixed_batch in fixed_batches:
        start = time.perf_counter()
        run_once(fixed_batch)
        durations.append(1000 * (time.perf_counter() - start))
    return durations


def export_and_measure_bucket(
        predictor: torch.nn.Module, batches: list[SceneBatch], onnx_path: Path,
        thread_count: int,
        device: torch.device) -> tuple[float, float, list[float], list[float]]:
    """Pads a bucket's scenes to its largest shape, exports at that shape,
    measures padding and export error, and times both.
    """
    agent_count = max(batch.scene.agent_history.shape[1] for batch in batches)
    chunk_count = max(
        batch.map.chunk_signal_history.shape[1] for batch in batches)
    dot_count = max(len(batch.map.rows) for batch in batches)
    target_count = max(len(batch.targets.agent_history) for batch in batches)
    fixed_batches = [
        padded_to_fixed_shape(batch,
                              agent_count=agent_count,
                              chunk_count=chunk_count,
                              dot_count=dot_count,
                              target_count=target_count) for batch in batches
    ]
    with torch.no_grad():
        padding_gap = max(
            float((predictor(batch)[0] -
                   predictor(fixed_batch)[0][:len(batch.targets.agent_history)]
                  ).abs().max())
            for batch, fixed_batch in zip(batches, fixed_batches))
    # The map-chunk axis is left symbolic because the exporter folds it
    # incorrectly when it is treated as fixed like the other axes.
    torch.onnx.export(
        PositionalInputs(predictor),
        graph_inputs(fixed_batches[0]),
        str(onnx_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={"map_chunk_signal_history": {
            1: "map_chunks"
        }},
        opset_version=18,
    )
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = thread_count
    session_options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        session_options,
        providers=[("CUDAExecutionProvider"
                    if device.type == "cuda" else "CPUExecutionProvider")],
    )
    device_predictor = PositionalInputs(predictor).to(device)

    def run_onnx(fixed_batch: SceneBatch) -> list[np.ndarray]:
        return session.run(
            OUTPUT_NAMES,
            {
                name: tensor.numpy()
                for name, tensor in zip(INPUT_NAMES, graph_inputs(fixed_batch))
            },
        )

    def run_torch(fixed_batch: SceneBatch) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            trajectories, confidence_logits = device_predictor(
                *(tensor.to(device) for tensor in graph_inputs(fixed_batch)))
        return trajectories.cpu(), confidence_logits.cpu()

    export_gap = max(
        float(
            np.abs(
                run_torch(fixed_batch)[0].numpy() -
                run_onnx(fixed_batch)[0]).max())
        for fixed_batch in fixed_batches)
    print(f"{onnx_path.name}: {len(fixed_batches)} scenes padded to"
          f" {target_count} predicted agents, {agent_count} scene"
          f" agents, {chunk_count} map chunks, {dot_count} map dots")
    return (
        padding_gap,
        export_gap,
        milliseconds_per_scene(run_torch, fixed_batches),
        milliseconds_per_scene(run_onnx, fixed_batches),
    )


def designated_target_scene_batches(
        scenario_paths: list[Path]) -> Iterator[SceneBatch]:
    """Yields one scene batch per scenario that has designated targets, skipping
    any scenario with none.
    """
    for scenario_path in scenario_paths:
        scenario = loader.read_scenario(scenario_path)
        track_indices = loader.eligible_track_indices(
            scenario.track_rows,
            scenario.track_valid,
            scenario.is_designated_target,
            designated_targets_only=True,
        )
        if not len(track_indices):
            continue
        yield pipeline.torch_batch(
            loader.build_scene_batch(
                [loader.build_scene_sample(scenario, track_indices.tolist())]))


def main() -> None:
    """Sorts scenes by token count into --buckets groups, exports one graph per
    group, and prints the errors and median timings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--scenarios", type=int, required=True)
    parser.add_argument("--buckets", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)

    predictor = model.MotionPredictor(load_anchor_file(arguments.anchors))
    if arguments.checkpoint is not None:
        predictor.load_state_dict(
            load_checkpoint_state(arguments.checkpoint)["model_state"])
    predictor.eval()

    batches = sorted(
        designated_target_scene_batches(
            sorted(arguments.staged_directory.glob("*.npz"))
            [:arguments.scenarios]),
        key=lambda batch: batch.targets.token_visible.shape[1],
    )
    assert (len(batches) >= arguments.buckets
           ), f"{len(batches)} scenes cannot fill {arguments.buckets} buckets"

    padding_gaps, export_gaps = [], []
    torch_milliseconds, onnx_milliseconds = [], []
    for bucket_index, bucket_positions in enumerate(
            np.array_split(np.arange(len(batches)), arguments.buckets)):
        padding_gap, export_gap, torch_durations, onnx_durations = (
            export_and_measure_bucket(
                predictor,
                [batches[position] for position in bucket_positions],
                arguments.onnx_path.with_name(
                    f"{arguments.onnx_path.stem}_{bucket_index}"
                    f"{arguments.onnx_path.suffix}"),
                arguments.threads,
                torch.device(arguments.device),
            ))
        padding_gaps.append(padding_gap)
        export_gaps.append(export_gap)
        torch_milliseconds.extend(torch_durations)
        onnx_milliseconds.extend(onnx_durations)

    print(f"padding moves a trajectory by at most"
          f" {max(padding_gaps):.2e} m")
    print(f"onnx differs from torch by at most {max(export_gaps):.2e} m")
    print(f"median per scene on {arguments.device},"
          f" {arguments.threads} threads:"
          f" torch {statistics.median(torch_milliseconds):.1f} ms,"
          f" onnxruntime {statistics.median(onnx_milliseconds):.1f} ms")


if __name__ == "__main__":
    main()
