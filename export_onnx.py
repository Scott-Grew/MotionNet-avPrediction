"""Exports the whole inference path to one ONNX graph: a scene of any size
in, each designated target's 6 kept futures at Waymo's 2 Hz and their
confidences out, in the target's own frame. Checks the graph against torch
on every scene and times both.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import functools
import statistics
import time
from collections.abc import Callable
from pathlib import Path

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
OPSET_VERSION = 18
WARMUP_RUNS = 2

Session = onnxruntime.InferenceSession
Outputs = tuple[Tensor, Tensor]
RunOnce = Callable[[SceneBatch], object]


def varying_axes(group: str, name: str) -> dict[int, str]:
    """The axes of one input whose size changes from scene to scene."""
    if group == "targets":
        return {0: "targets"}
    if group == "scene":
        return {0: "scenes", 1: "agents"}
    if name.startswith("chunk"):
        return {0: "scenes", 1: "chunks"}
    return {0: "scenes", 1: "dots"}


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


def export(graph: InferenceGraph, batch: SceneBatch, path: Path) -> None:
    """Traces the graph on one scene, with every scene-sized axis left free."""
    axes = {name: {0: "targets"} for name in OUTPUT_NAMES}
    for (group, name), input_name in zip(INPUT_FIELDS, INPUT_NAMES):
        axes[input_name] = varying_axes(group, name)
    torch.onnx.export(
        graph,
        graph_inputs(batch),
        str(path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=OPSET_VERSION,
        dynamic_axes=axes,
    )


def onnx_session(path: Path, threads: int) -> Session:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    return Session(str(path), options, providers=providers)


def torch_run(graph: InferenceGraph, batch: SceneBatch) -> Outputs:
    with torch.no_grad():
        return graph(*graph_inputs(batch))


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


def designated_batches(paths: list[Path]) -> list[SceneBatch]:
    """One batch per scenario holding its designated targets, skipping
    scenarios without any.
    """
    batches = []
    for path in paths:
        scenario = loader.read_scenario(path)
        tracks = loader.designated_tracks(scenario)
        if len(tracks):
            scene = loader.one_scene(scenario, tracks)
            batches.append(pipeline.torch_batch(scene))
    return batches


def main() -> None:
    """Exports the graph, then prints how far its futures sit from torch's
    and the median time per scene of each.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)

    graph = InferenceGraph(load_predictor(arguments.checkpoint))
    paths = sorted(arguments.staged_directory.glob("*.npz"))
    batches = designated_batches(paths[: arguments.scenarios])
    export(graph, batches[0], arguments.onnx_path)
    session = onnx_session(arguments.onnx_path, arguments.threads)

    run_torch = functools.partial(torch_run, graph)
    run_onnx = functools.partial(onnx_run, session)
    gaps = []
    for batch in batches:
        torch_trajectories = run_torch(batch)[0].numpy()
        onnx_trajectories = run_onnx(batch)[0]
        gap = np.abs(torch_trajectories - onnx_trajectories).max()
        gaps.append(float(gap))
    torch_times = scene_milliseconds(run_torch, batches)
    onnx_times = scene_milliseconds(run_onnx, batches)
    print(f"{len(batches)} scenes, one graph at {arguments.onnx_path}")
    print(f"onnx differs from torch by at most {max(gaps):.2e} m")
    print(
        f"median per scene on {arguments.threads} CPU threads:"
        f" torch {statistics.median(torch_times):.1f} ms,"
        f" onnxruntime {statistics.median(onnx_times):.1f} ms"
    )


if __name__ == "__main__":
    main()
