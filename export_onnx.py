"""Exports the trained model to ONNX, one graph per bucket of scene sizes.

Checks each exported graph against torch and times both.
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

# The graph inputs, one per batch array the model reads, as (group, field)
# in the order the exported graph takes them.
SCENE_FIELDS = [("scene", name) for name in SceneArrays._fields]
MAP_FIELDS = [("map", name) for name in MapArrays._fields]
TARGET_FIELDS = [
    ("targets", name)
    for name in TargetArrays._fields
    if not name.startswith("future")
]
INPUT_FIELDS = SCENE_FIELDS + MAP_FIELDS + TARGET_FIELDS
INPUT_NAMES = [f"{group}_{name}" for group, name in INPUT_FIELDS]
OUTPUT_NAMES = ["trajectories", "confidence_logits"]
TOKEN_FIELDS = ("token_visible", "token_pose")
# The map-chunk axis stays symbolic because the exporter folds it
# incorrectly when it is treated as fixed like the other axes.
TOKEN_AXIS = 1
DYNAMIC_AXES = {"map_chunk_signal_history": {TOKEN_AXIS: "map_chunks"}}
OPSET_VERSION = 18
WARMUP_RUNS = 2

Session = onnxruntime.InferenceSession
RunOnce = Callable[[SceneBatch], object]


class FixedShape(NamedTuple):
    """The counts a bucket's scenes are padded to, those of its largest."""

    agents: int
    chunks: int
    dots: int
    targets: int


class Runtime(NamedTuple):
    """Where the exported graph and its torch reference run."""

    device: torch.device
    threads: int


class BucketReport(NamedTuple):
    """How far padding and export moved a trajectory, in metres, and the
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
    """The SceneBatch behind positional tensors in INPUT_FIELDS order, with no
    future fields.
    """
    fields: dict[str, dict[str, Tensor]] = {
        "scene": {},
        "map": {},
        "targets": {},
    }
    for (group, name), tensor in zip(INPUT_FIELDS, tensors, strict=True):
        fields[group][name] = tensor
    scene = SceneArrays(**fields["scene"])
    scene_map = MapArrays(**fields["map"])
    targets = TargetArrays(**fields["targets"])
    return SceneBatch(scene, scene_map, targets)


class PositionalInputs(nn.Module):
    """Wraps the model so ONNX export sees positional tensor arguments instead
    of the SceneBatch the model normally takes.
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *tensors: Tensor) -> tuple[Tensor, Tensor]:
        predicted = self.predictor(batch_from_graph_inputs(tensors))
        return predicted.trajectories, predicted.confidence_logits


def pad_axis(tensor: Tensor, axis: int, length: int) -> Tensor:
    """Pads a tensor with zeros along one axis up to length; the added rows
    are absent tokens, not real agents or map chunks.
    """
    padding_shape = list(tensor.shape)
    padding_shape[axis] = length - tensor.shape[axis]
    padding = tensor.new_zeros(padding_shape)
    return torch.cat([tensor, padding], dim=axis)


def repeat_last_row(tensor: Tensor, length: int) -> Tensor:
    """Pads a tensor up to length by repeating its last row, which leaves
    max-pooled chunk tokens and real predictions unchanged.
    """
    repeats = tensor[-1:].expand(length - len(tensor), *tensor.shape[1:])
    return torch.cat([tensor, repeats])


def bucket_shape(batches: list[SceneBatch]) -> FixedShape:
    """The largest agent, chunk, dot and target counts across a bucket."""
    return FixedShape(
        agents=max(batch.scene.agent_history.shape[1] for batch in batches),
        chunks=max(
            batch.map.chunk_signal_history.shape[1] for batch in batches
        ),
        dots=max(len(batch.map.rows) for batch in batches),
        targets=max(len(batch.targets.agent_history) for batch in batches),
    )


class Padder:
    """Pads one scene batch to a bucket's fixed shape, so differently sized
    scenes can share one exported graph.
    """

    def __init__(self, batch: SceneBatch, shape: FixedShape) -> None:
        self.batch = batch
        self.shape = shape
        self.agents = batch.scene.agent_history.shape[1]
        self.chunks = batch.map.chunk_signal_history.shape[1]

    def scene_arrays(self) -> SceneArrays:
        padded = {}
        for name, array in self.batch.scene._asdict().items():
            padded[name] = pad_axis(array, TOKEN_AXIS, self.shape.agents)
        return SceneArrays(**padded)

    def map_arrays(self) -> MapArrays:
        """Pads the dots by repeating the last one and renumbers each dot's
        slot, since a slot is scene index * chunk count + chunk index.
        """
        scene_map = self.batch.map
        slot = scene_map.dot_chunk_slot
        renumbered = (
            slot // self.chunks
        ) * self.shape.chunks + slot % self.chunks
        signals = pad_axis(
            scene_map.chunk_signal_history, TOKEN_AXIS, self.shape.chunks
        )
        return MapArrays(
            rows=repeat_last_row(scene_map.rows, self.shape.dots),
            dot_chunk_slot=repeat_last_row(renumbered, self.shape.dots),
            chunk_signal_history=signals,
        )

    def tokens(self, tokens: Tensor) -> Tensor:
        """The token axis holds the scene agents and then the map chunks, so
        each half is padded separately before being joined back together.
        """
        agent_half = pad_axis(
            tokens[:, : self.agents], TOKEN_AXIS, self.shape.agents
        )
        chunk_half = pad_axis(
            tokens[:, self.agents :], TOKEN_AXIS, self.shape.chunks
        )
        return torch.cat([agent_half, chunk_half], dim=TOKEN_AXIS)

    def target_arrays(self) -> TargetArrays:
        padded = {}
        for name, array in self.batch.targets._asdict().items():
            if array is None:
                padded[name] = None
                continue
            if name in TOKEN_FIELDS:
                array = self.tokens(array)
            padded[name] = repeat_last_row(array, self.shape.targets)
        return TargetArrays(**padded)

    def padded(self) -> SceneBatch:
        return SceneBatch(
            self.scene_arrays(), self.map_arrays(), self.target_arrays()
        )


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
    """Exports buckets of scenes from one predictor on one runtime, and
    measures each export against torch.
    """

    def __init__(self, predictor: nn.Module, runtime: Runtime) -> None:
        self.predictor = predictor
        self.runtime = runtime
        self.device_predictor = PositionalInputs(predictor).to(runtime.device)

    def torch_run(self, batch: SceneBatch) -> tuple[Tensor, Tensor]:
        inputs = [
            tensor.to(self.runtime.device) for tensor in graph_inputs(batch)
        ]
        with torch.no_grad():
            trajectories, confidence_logits = self.device_predictor(*inputs)
        return trajectories.cpu(), confidence_logits.cpu()

    def padding_gap(self, batch: SceneBatch, fixed: SceneBatch) -> float:
        """How far padding moves the real targets' trajectories, in metres."""
        target_count = len(batch.targets.agent_history)
        with torch.no_grad():
            unpadded = self.predictor(batch).trajectories
            padded = self.predictor(fixed).trajectories
        return float((unpadded - padded[:target_count]).abs().max())

    def export(self, batch: SceneBatch, onnx_path: Path) -> Session:
        """Exports the graph at the batch's shape and opens it."""
        wrapped = PositionalInputs(self.predictor)
        torch.onnx.export(
            wrapped,
            graph_inputs(batch),
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=DYNAMIC_AXES,
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
        """How far the exported graph's trajectories sit from torch's."""
        torch_trajectories, _ = self.torch_run(fixed)
        onnx_trajectories = onnx_run(session, fixed)[0]
        return float(
            np.abs(torch_trajectories.numpy() - onnx_trajectories).max()
        )

    def measure(self, batches: list[SceneBatch], path: Path) -> BucketReport:
        """Pads a bucket's scenes to its largest shape, exports at that shape,
        measures padding and export error, and times both.
        """
        shape = bucket_shape(batches)
        fixed = [Padder(batch, shape).padded() for batch in batches]
        padding_gaps = []
        for batch, fixed_batch in zip(batches, fixed, strict=True):
            padding_gaps.append(self.padding_gap(batch, fixed_batch))
        session = self.export(fixed[0], path)
        export_gaps = [
            self.export_gap(session, fixed_batch) for fixed_batch in fixed
        ]
        run_onnx = functools.partial(onnx_run, session)
        torch_milliseconds = scene_milliseconds(self.torch_run, fixed)
        onnx_milliseconds = scene_milliseconds(run_onnx, fixed)
        print(
            f"{path.name}: {len(fixed)} scenes padded to {shape.targets}"
            f" predicted agents, {shape.agents} scene agents,"
            f" {shape.chunks} map chunks, {shape.dots} map dots"
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


def token_count(batch: SceneBatch) -> int:
    return batch.targets.token_visible.shape[1]


def bucket_path(onnx_path: Path, bucket_index: int) -> Path:
    return onnx_path.with_name(
        f"{onnx_path.stem}_{bucket_index}{onnx_path.suffix}"
    )


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
    print(f"padding moves a trajectory by at most {padding_gap:.2e} m")
    print(f"onnx differs from torch by at most {export_gap:.2e} m")
    print(
        f"median per scene on {runtime.device.type}, {runtime.threads} threads:"
        f" torch {statistics.median(torch_times):.1f} ms,"
        f" onnxruntime {statistics.median(onnx_times):.1f} ms"
    )


def main() -> None:
    """Sorts scenes by token count into --buckets groups, exports one graph per
    group, and prints the errors and median timings.
    """
    arguments = parse_arguments()
    torch.set_num_threads(arguments.threads)
    runtime = Runtime(torch.device(arguments.device), arguments.threads)
    predictor = load_predictor(arguments.checkpoint)
    paths = sorted(arguments.staged_directory.glob("*.npz"))
    unsorted = scene_batches(paths[: arguments.scenarios])
    batches = sorted(unsorted, key=token_count)
    if len(batches) < arguments.buckets:
        raise ValueError(
            f"{len(batches)} scenes cannot fill {arguments.buckets} buckets"
        )
    exporter = Exporter(predictor, runtime)
    reports = []
    buckets = np.array_split(np.arange(len(batches)), arguments.buckets)
    for bucket_index, positions in enumerate(buckets):
        bucket = [batches[position] for position in positions]
        onnx_path = bucket_path(arguments.onnx_path, bucket_index)
        reports.append(exporter.measure(bucket, onnx_path))
    print_summary(reports, runtime)


if __name__ == "__main__":
    main()
