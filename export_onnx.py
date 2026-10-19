"""Exports the whole inference path to ONNX: one scene in, each designated
target's 6 kept futures at Waymo's 2 Hz and their confidences out, in the
target's own frame. Scenes are sorted by size into buckets and each bucket's
graph is exported at its largest scene's shape. Checks every graph against
torch and times both.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import functools
import statistics
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime
import torch
from torch import Tensor, nn

from womd import loader, pipeline
from womd.checkpoint import load_predictor
from womd.loader import MapArrays, SceneArrays, SceneBatch, TargetArrays
from womd.pruning import submitted_modes

# The graph's inputs are the batch arrays the model reads, scene group
# first, named "<group>_<field>"; the logged future is not an input.
TARGET_INPUTS = [name for name in TargetArrays._fields if "future" not in name]
INPUT_FIELDS = (
    [("scene", name) for name in SceneArrays._fields]
    + [("map", name) for name in MapArrays._fields]
    + [("targets", name) for name in TARGET_INPUTS]
)
INPUT_NAMES = [f"{group}_{name}" for group, name in INPUT_FIELDS]
OUTPUT_NAMES = ["trajectories", "confidences"]
# Which padded count each map array's second axis follows.
MAP_AXIS_COUNTS = {
    "rows": "dots",
    "dot_present": "dots",
    "dot_chunk": "dots",
    "chunk_pose": "chunks",
    "chunk_signal_history": "chunks",
}
# Opset 18 is the first that expresses the max-pooling scatter of map
# chunks; torch 2.2 warns it is unsupported but exports it.
OPSET_VERSION = 18
WARMUP_RUNS = 2

Session = onnxruntime.InferenceSession
Outputs = tuple[Tensor, Tensor]
RunOnce = Callable[[SceneBatch], object]


class FixedShape(NamedTuple):
    """The counts a bucket's scenes are padded to, those of its largest."""

    agents: int
    dots: int
    chunks: int
    targets: int


class Runtime(NamedTuple):
    """Where the exported graph and its torch reference run."""

    device: torch.device
    threads: int


class BucketReport(NamedTuple):
    """How far padding and export moved a future, in metres, and the
    per-scene times in milliseconds.
    """

    padding_gap: float
    export_gap: float
    torch_milliseconds: list[float]
    onnx_milliseconds: list[float]


def graph_inputs(batch: SceneBatch) -> tuple[Tensor, ...]:
    """The batch's arrays in INPUT_FIELDS order."""
    tensors = []
    for group, name in INPUT_FIELDS:
        arrays = getattr(batch, group)
        tensors.append(getattr(arrays, name))
    return tuple(tensors)


def batch_from_graph_inputs(tensors: tuple[Tensor, ...]) -> SceneBatch:
    """The SceneBatch behind positional tensors in INPUT_FIELDS order."""
    scene_end = len(SceneArrays._fields)
    map_end = scene_end + len(MapArrays._fields)
    scene = SceneArrays(*tensors[:scene_end])
    scene_map = MapArrays(*tensors[scene_end:map_end])
    targets = TargetArrays(*tensors[map_end:])
    return SceneBatch(scene, scene_map, targets)


class InferenceGraph(nn.Module):
    """The model followed by pruning to the 6 submitted modes, taking the
    batch as positional tensors, as ONNX export needs.
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *tensors: Tensor) -> Outputs:
        predicted = self.predictor(batch_from_graph_inputs(tensors))
        trajectories = predicted.trajectories
        logits = predicted.confidence_logits
        submitted = submitted_modes(trajectories, logits)
        return submitted.trajectories, submitted.confidences


def pad_axis(tensor: Tensor, length: int) -> Tensor:
    """Pads a tensor's second axis with zeros up to length. A zero mask or
    presence flag marks the added agents, dots and chunks as absent.
    """
    padding_shape = list(tensor.shape)
    padding_shape[1] = length - tensor.shape[1]
    padding = tensor.new_zeros(padding_shape)
    return torch.cat([tensor, padding], dim=1)


def repeat_last_row(tensor: Tensor, length: int) -> Tensor:
    """Pads the target axis up to length by repeating the last target; each
    target is predicted on its own, so the real ones do not change.
    """
    repeats = tensor[-1:].expand(length - len(tensor), *tensor.shape[1:])
    return torch.cat([tensor, repeats])


def bucket_shape(batches: list[SceneBatch]) -> FixedShape:
    """The largest agent, dot, chunk and target counts across a bucket."""
    return FixedShape(
        agents=max(batch.scene.agent_history.shape[1] for batch in batches),
        dots=max(batch.map.rows.shape[1] for batch in batches),
        chunks=max(batch.map.chunk_pose.shape[1] for batch in batches),
        targets=max(len(batch.targets.agent_history) for batch in batches),
    )


def padded(batch: SceneBatch, shape: FixedShape) -> SceneBatch:
    """One scene batch padded to a bucket's fixed shape."""
    scene = [pad_axis(array, shape.agents) for array in batch.scene]
    scene_map = []
    for name, array in zip(MapArrays._fields, batch.map, strict=True):
        length = getattr(shape, MAP_AXIS_COUNTS[name])
        scene_map.append(pad_axis(array, length))
    targets = []
    for array in batch.targets:
        if array is not None:
            array = repeat_last_row(array, shape.targets)
        targets.append(array)
    scene_arrays = SceneArrays(*scene)
    map_arrays = MapArrays(*scene_map)
    return SceneBatch(scene_arrays, map_arrays, TargetArrays(*targets))


def onnx_run(session: Session, batch: SceneBatch) -> list[np.ndarray]:
    feeds = {}
    for name, tensor in zip(INPUT_NAMES, graph_inputs(batch), strict=True):
        feeds[name] = tensor.numpy()
    return session.run(OUTPUT_NAMES, feeds)


def scene_milliseconds(run: RunOnce, batches: list[SceneBatch]) -> list[float]:
    """Times run over each batch after a few warm-up calls, one duration in
    milliseconds per scene.
    """
    for _ in range(WARMUP_RUNS):
        run(batches[0])
    durations = []
    for batch in batches:
        start = time.perf_counter()
        run(batch)
        durations.append(1000 * (time.perf_counter() - start))
    return durations


class Exporter:
    """Exports buckets of scenes from one inference graph on one runtime, and
    measures each export against torch.
    """

    def __init__(self, graph: InferenceGraph, runtime: Runtime) -> None:
        self.graph = graph
        self.runtime = runtime
        self.device_graph = InferenceGraph(graph.predictor).to(runtime.device)

    def torch_run(self, batch: SceneBatch) -> Outputs:
        device = self.runtime.device
        inputs = [tensor.to(device) for tensor in graph_inputs(batch)]
        with torch.no_grad():
            trajectories, confidences = self.device_graph(*inputs)
        return trajectories.cpu(), confidences.cpu()

    def padding_gap(self, batch: SceneBatch, fixed: SceneBatch) -> float:
        """How far padding moves the real targets' futures, in metres."""
        target_count = len(batch.targets.agent_history)
        unpadded, _ = self.torch_run(batch)
        padded_output, _ = self.torch_run(fixed)
        gap = unpadded - padded_output[:target_count]
        return float(gap.abs().max())

    def export(self, batch: SceneBatch, onnx_path: Path) -> Session:
        """Exports the graph at the batch's shape and opens it."""
        torch.onnx.export(
            self.graph,
            graph_inputs(batch),
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=OPSET_VERSION,
        )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.runtime.threads
        options.log_severity_level = 3
        provider = "CPUExecutionProvider"
        if self.runtime.device.type == "cuda":
            provider = "CUDAExecutionProvider"
        return Session(str(onnx_path), options, providers=[provider])

    def export_gap(self, session: Session, fixed: SceneBatch) -> float:
        """How far the exported graph's futures sit from torch's, in metres."""
        torch_trajectories, _ = self.torch_run(fixed)
        onnx_trajectories = onnx_run(session, fixed)[0]
        gap = torch_trajectories.numpy() - onnx_trajectories
        return float(np.abs(gap).max())

    def measure(self, batches: list[SceneBatch], path: Path) -> BucketReport:
        """Pads a bucket's scenes to its largest shape, exports at that shape,
        measures padding and export error, and times both.
        """
        shape = bucket_shape(batches)
        fixed = [padded(batch, shape) for batch in batches]
        padding_gaps = []
        for batch, fixed_batch in zip(batches, fixed, strict=True):
            padding_gaps.append(self.padding_gap(batch, fixed_batch))
        session = self.export(fixed[0], path)
        export_gaps = [self.export_gap(session, batch) for batch in fixed]
        run_onnx = functools.partial(onnx_run, session)
        torch_milliseconds = scene_milliseconds(self.torch_run, fixed)
        onnx_milliseconds = scene_milliseconds(run_onnx, fixed)
        print(
            f"{path.name}: {len(fixed)} scenes padded to {shape.targets}"
            f" targets, {shape.agents} agents, {shape.chunks} map chunks,"
            f" {shape.dots} map dots"
        )
        return BucketReport(
            max(padding_gaps),
            max(export_gaps),
            torch_milliseconds,
            onnx_milliseconds,
        )


def scene_batches(paths: list[Path]) -> Iterator[SceneBatch]:
    """One scene batch per scenario with designated targets."""
    for path in paths:
        scenario = loader.read_scenario(path)
        track_indices = loader.eligible_tracks(scenario, designated_only=True)
        if len(track_indices) == 0:
            continue
        sample = loader.scene_sample(scenario, track_indices.tolist())
        yield pipeline.torch_batch(loader.scene_batch([sample]))


def scene_size(batch: SceneBatch) -> int:
    return batch.map.rows.shape[1] + batch.scene.agent_history.shape[1]


def bucket_path(onnx_path: Path, bucket_index: int) -> Path:
    name = f"{onnx_path.stem}_{bucket_index}{onnx_path.suffix}"
    return onnx_path.with_name(name)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, required=True)
    parser.add_argument("--buckets", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    return parser.parse_args()


def print_summary(reports: list[BucketReport], runtime: Runtime) -> None:
    padding_gap = max(report.padding_gap for report in reports)
    export_gap = max(report.export_gap for report in reports)
    torch_times = [ms for report in reports for ms in report.torch_milliseconds]
    onnx_times = [ms for report in reports for ms in report.onnx_milliseconds]
    print(f"padding moves a future by at most {padding_gap:.2e} m")
    print(f"onnx differs from torch by at most {export_gap:.2e} m")
    print(
        f"median per scene on {runtime.device.type}, {runtime.threads} threads:"
        f" torch {statistics.median(torch_times):.1f} ms,"
        f" onnxruntime {statistics.median(onnx_times):.1f} ms"
    )


def main() -> None:
    """Sorts scenes by size into --buckets groups, exports one graph per
    group, and prints the errors and median timings.
    """
    arguments = parse_arguments()
    torch.set_num_threads(arguments.threads)
    runtime = Runtime(torch.device(arguments.device), arguments.threads)
    graph = InferenceGraph(load_predictor(arguments.checkpoint))
    paths = sorted(arguments.staged_directory.glob("*.npz"))
    unsorted = scene_batches(paths[: arguments.scenarios])
    batches = sorted(unsorted, key=scene_size)
    if len(batches) < arguments.buckets:
        raise ValueError(
            f"{len(batches)} scenes cannot fill {arguments.buckets} buckets"
        )
    exporter = Exporter(graph, runtime)
    reports = []
    buckets = np.array_split(np.arange(len(batches)), arguments.buckets)
    for bucket_index, positions in enumerate(buckets):
        bucket = [batches[position] for position in positions]
        onnx_path = bucket_path(arguments.onnx_path, bucket_index)
        reports.append(exporter.measure(bucket, onnx_path))
    print_summary(reports, runtime)


if __name__ == "__main__":
    main()
