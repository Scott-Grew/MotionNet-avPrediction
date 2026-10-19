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
from womd.loader import (
    BatchSizes,
    MapArrays,
    SceneArrays,
    SceneBatch,
    SceneSample,
    TargetArrays,
)
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
OPSET_VERSION = 18
WARMUP_RUNS = 2

Session = onnxruntime.InferenceSession
Outputs = tuple[Tensor, Tensor]
RunOnce = Callable[[SceneBatch], object]


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
    groups: dict[str, dict[str, Tensor]] = {
        "scene": {},
        "map": {},
        "targets": {},
    }
    for (group, name), tensor in zip(INPUT_FIELDS, tensors, strict=True):
        groups[group][name] = tensor
    scene = SceneArrays(**groups["scene"])
    scene_map = MapArrays(**groups["map"])
    return SceneBatch(scene, scene_map, TargetArrays(**groups["targets"]))


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


def bucket_sizes(samples: list[SceneSample]) -> BatchSizes:
    """The largest agent, dot, chunk and target counts across a bucket."""
    sizes = [loader.largest_sizes([sample]) for sample in samples]
    return BatchSizes(*(max(column) for column in zip(*sizes, strict=True)))


def padded_batch(sample: SceneSample, sizes: BatchSizes) -> SceneBatch:
    """One scene's batch padded to a bucket's sizes."""
    batch = loader.BatchBuilder([sample], sizes).build()
    return pipeline.torch_batch(batch)


def scene_batch(sample: SceneSample) -> SceneBatch:
    return pipeline.torch_batch(loader.scene_batch([sample]))


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
        self.graph = graph.to(runtime.device)
        self.runtime = runtime

    def device_inputs(self, batch: SceneBatch) -> list[Tensor]:
        device = self.runtime.device
        return [tensor.to(device) for tensor in graph_inputs(batch)]

    def torch_run(self, batch: SceneBatch) -> Outputs:
        with torch.no_grad():
            trajectories, confidences = self.graph(*self.device_inputs(batch))
        return trajectories.cpu(), confidences.cpu()

    def padding_gap(self, sample: SceneSample, fixed: SceneBatch) -> float:
        """How far padding moves the real targets' futures, in metres."""
        unpadded, _ = self.torch_run(scene_batch(sample))
        padded_output, _ = self.torch_run(fixed)
        gap = unpadded - padded_output[: len(sample.targets)]
        return float(gap.abs().max())

    def export(self, batch: SceneBatch, onnx_path: Path) -> Session:
        """Exports the graph at the batch's shape and opens it."""
        torch.onnx.export(
            self.graph,
            tuple(self.device_inputs(batch)),
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

    def measure(self, samples: list[SceneSample], path: Path) -> BucketReport:
        """Pads a bucket's scenes to its largest sizes, exports at that shape,
        measures padding and export error, and times both.
        """
        sizes = bucket_sizes(samples)
        fixed = [padded_batch(sample, sizes) for sample in samples]
        padding_gaps = []
        for sample, fixed_batch in zip(samples, fixed, strict=True):
            padding_gaps.append(self.padding_gap(sample, fixed_batch))
        session = self.export(fixed[0], path)
        export_gaps = [self.export_gap(session, batch) for batch in fixed]
        run_onnx = functools.partial(onnx_run, session)
        torch_milliseconds = scene_milliseconds(self.torch_run, fixed)
        onnx_milliseconds = scene_milliseconds(run_onnx, fixed)
        print(
            f"{path.name}: {len(fixed)} scenes padded to {sizes.targets}"
            f" targets, {sizes.agents} agents, {sizes.chunks} map chunks,"
            f" {sizes.dots} map dots"
        )
        return BucketReport(
            max(padding_gaps),
            max(export_gaps),
            torch_milliseconds,
            onnx_milliseconds,
        )


def designated_samples(paths: list[Path]) -> Iterator[SceneSample]:
    """Each scenario's designated targets, skipping scenarios without any."""
    for path in paths:
        sample = loader.designated_scene(path).sample
        if sample.targets:
            yield sample


def scene_size(sample: SceneSample) -> int:
    return len(sample.map_rows) + len(sample.scene_agent_history)


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
    unsorted = designated_samples(paths[: arguments.scenarios])
    samples = sorted(unsorted, key=scene_size)
    if len(samples) < arguments.buckets:
        raise ValueError(
            f"{len(samples)} scenes cannot fill {arguments.buckets} buckets"
        )
    exporter = Exporter(graph, runtime)
    reports = []
    buckets = np.array_split(np.arange(len(samples)), arguments.buckets)
    for bucket_index, positions in enumerate(buckets):
        bucket = [samples[position] for position in positions]
        onnx_path = bucket_path(arguments.onnx_path, bucket_index)
        reports.append(exporter.measure(bucket, onnx_path))
    print_summary(reports, runtime)


if __name__ == "__main__":
    main()
