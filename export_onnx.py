"""Exports the whole inference path to one ONNX graph: any number of
targets' inputs in, each target's 6 kept futures at Waymo's 2 Hz and their
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

from womd import formats, views
from womd.checkpoint import load_predictor
from womd.pruning import submitted_modes
from womd.views import TargetInputs

# The graph's inputs are the arrays the model reads, in TargetInputs order;
# the logged future is not an input.
INPUT_NAMES = [name for name in TargetInputs._fields if "future" not in name]
OUTPUT_NAMES = ["trajectories", "confidences"]
OPSET_VERSION = 17
WARMUP_RUNS = 2

Session = onnxruntime.InferenceSession
Outputs = tuple[Tensor, Tensor]
RunOnce = Callable[[TargetInputs], object]


def graph_inputs(batch: TargetInputs) -> tuple[Tensor, ...]:
    """The batch's arrays in INPUT_NAMES order."""
    return tuple(getattr(batch, name) for name in INPUT_NAMES)


class InferenceGraph(nn.Module):
    """The model followed by pruning to the 6 submitted modes, taking the
    batch as positional tensors, as ONNX export needs.
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *tensors: Tensor) -> Outputs:
        predicted = self.predictor(TargetInputs(*tensors))
        trajectories = predicted.trajectories
        logits = predicted.confidence_logits
        submitted = submitted_modes(trajectories, logits)
        return submitted.trajectories, submitted.confidences


def export(graph: InferenceGraph, batch: TargetInputs, path: Path) -> None:
    """Traces the graph on one scene, with the target axis left free."""
    names = INPUT_NAMES + OUTPUT_NAMES
    axes = {name: {0: "targets"} for name in names}
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


def torch_run(graph: InferenceGraph, batch: TargetInputs) -> Outputs:
    with torch.no_grad():
        return graph(*graph_inputs(batch))


def onnx_run(session: Session, batch: TargetInputs) -> list[np.ndarray]:
    feeds = {}
    for name, tensor in zip(INPUT_NAMES, graph_inputs(batch), strict=True):
        feeds[name] = tensor.numpy()
    return session.run(OUTPUT_NAMES, feeds)


def milliseconds(run: RunOnce, batches: list[TargetInputs]) -> list[float]:
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


def designated_batches(paths: list[Path]) -> list[TargetInputs]:
    """One batch per scenario holding its designated targets, skipping
    scenarios without any.
    """
    batches = []
    for path in paths:
        scenario = formats.StagedFile.load(path)
        tracks = views.designated_tracks(scenario)
        if len(tracks):
            scene = views.scene_inputs(scenario, tracks)
            batches.append(scene.each(torch.from_numpy))
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
    torch_times = milliseconds(run_torch, batches)
    onnx_times = milliseconds(run_onnx, batches)
    print(f"{len(batches)} scenes, one graph at {arguments.onnx_path}")
    print(f"onnx differs from torch by at most {max(gaps):.2e} m")
    print(
        f"median per scene on {arguments.threads} CPU threads:"
        f" torch {statistics.median(torch_times):.1f} ms,"
        f" onnxruntime {statistics.median(onnx_times):.1f} ms"
    )


if __name__ == "__main__":
    main()
