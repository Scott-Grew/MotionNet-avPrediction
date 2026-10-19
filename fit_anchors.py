"""Step 2 of 5.

Clusters where training agents ended up after 8 s into 54 anchor endpoints per
object type.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

from womd import contract, loader, model

# Fixed, so a refit on the same endpoints gives the same anchors.
# Every other k-means setting is scikit-learn's default.
RANDOM_STATE_SEED = 0


class Endpoints(NamedTuple):
    """Where each designated training target ended up after 8 s, in its own
    frame, with its predicted type and how far it could have driven.
    """

    positions: Tensor  # (N, 2) metres
    type_index: Tensor  # (N,)
    reachable_metres: Tensor  # (N,)

    def of_type(self, type_index: int) -> Tensor:
        return self.positions[self.type_index == type_index]


class TypeFit(NamedTuple):
    """One object type's fitted anchors, most-used first, with how many
    endpoints each drew and how k-means stopped.
    """

    type_name: str
    sample_count: int
    anchors: Tensor  # (54, 2)
    counts: Tensor  # (54,)
    iteration_count: int
    stopped_by_convergence: bool


def extract_endpoints(paths: list[Path]) -> Endpoints:
    """Reads every designated target's final valid future position from the
    staged scenarios.
    """
    histories = []
    positions = []
    for path in paths:
        scenario = loader.read_scenario(path)
        eligible = loader.eligible_tracks(scenario, designated_only=True)
        sample = loader.scene_sample(scenario, eligible.tolist())
        for target in sample.targets:
            valid_steps = np.flatnonzero(target.future_mask)
            if valid_steps.size == 0:
                continue
            histories.append(target.agent_history)
            positions.append(target.future_positions[valid_steps[-1]])
    history = torch.from_numpy(np.stack(histories))
    return Endpoints(
        torch.from_numpy(np.stack(positions)),
        model.predicted_type_index(history),
        model.agent_reachable_distance(history),
    )


def cache_path_of(staged_directory: Path) -> Path:
    """Endpoints are cached beside the staged directory they came from."""
    name = f"{staged_directory.name}_endpoints_cache.npz"
    return staged_directory.parent / name


def read_cache(cache_path: Path) -> Endpoints | None:
    """The cached endpoints, or None when there is no cache or it was written
    by another code version.
    """
    if not cache_path.exists():
        return None
    with np.load(cache_path) as cache:
        stamp = cache["provenance"] if "provenance" in cache else None
        try:
            contract.check_provenance(stamp, cache_path, "Re-extracting.")
        except ValueError as refusal:
            print(refusal)
            return None
        positions = torch.from_numpy(cache["endpoints"])
        type_index = torch.from_numpy(cache["predicted_type_index"])
        reachable = torch.from_numpy(cache["reachable_distance_metres"])
    print(f"endpoints read from {cache_path}")
    return Endpoints(positions, type_index, reachable)


def write_cache(cache_path: Path, endpoints: Endpoints, source: Path) -> None:
    reachable = endpoints.reachable_metres.numpy()
    np.savez(
        cache_path,
        endpoints=endpoints.positions.numpy().astype(np.float32),
        predicted_type_index=endpoints.type_index.numpy().astype(np.int64),
        reachable_distance_metres=reachable.astype(np.float32),
        provenance=contract.artifact_provenance("fit_anchors.py", source),
    )
    print(f"endpoints cached to {cache_path}")


def cached_endpoints(staged_directory: Path) -> Endpoints:
    """Extracting endpoints reads the whole staged directory, so the result is
    cached and reused while its provenance matches this code.
    """
    cache_path = cache_path_of(staged_directory)
    cached = read_cache(cache_path)
    if cached is not None:
        return cached
    paths = sorted(staged_directory.glob("*.npz"))
    endpoints = extract_endpoints(paths)
    write_cache(cache_path, endpoints, staged_directory)
    return endpoints


def reachable_only(endpoints: Endpoints) -> Endpoints:
    """Leaves out endpoints further away than their agent could have driven.
    They stay training samples; they only do not shape the anchors.
    """
    reachable = endpoints.positions.norm(dim=-1) <= endpoints.reachable_metres
    print(
        f"{int((~reachable).sum())} endpoints beyond reachable distance"
        f" left out of the fit"
    )
    return Endpoints(*(field[reachable] for field in endpoints))


def fit_type(type_name: str, positions: Tensor) -> TypeFit:
    """Fits one object type's anchors with k-means and orders them most-used
    first.
    """
    if len(positions) < model.QUERY_COUNT:
        raise ValueError(
            f"{type_name} has {len(positions)} endpoints,"
            f" fewer than {model.QUERY_COUNT} anchors"
        )
    kmeans = KMeans(
        n_clusters=model.QUERY_COUNT, random_state=RANDOM_STATE_SEED
    )
    fitted = kmeans.fit(positions.numpy())
    assignment = torch.from_numpy(fitted.labels_).to(torch.long)
    counts = torch.zeros(model.QUERY_COUNT, dtype=torch.long)
    counts.index_add_(0, assignment, torch.ones_like(assignment))
    order = torch.argsort(counts, descending=True, stable=True)
    centres = torch.from_numpy(fitted.cluster_centers_).to(positions.dtype)
    return TypeFit(
        type_name=type_name,
        sample_count=len(positions),
        anchors=centres[order],
        counts=counts[order],
        iteration_count=fitted.n_iter_,
        stopped_by_convergence=fitted.n_iter_ < fitted.max_iter,
    )


def minimum_pairwise_distance(anchors: Tensor) -> float:
    """Smallest distance between any two distinct anchors, in metres."""
    separations = torch.cdist(anchors, anchors)
    separations.fill_diagonal_(float("inf"))
    return float(separations.min())


def print_fit(fit: TypeFit) -> None:
    """Prints one object type's anchors with their offset, distance, angle and
    share of endpoints.
    """
    stop_reason = (
        "convergence" if fit.stopped_by_convergence else "the iteration cap"
    )
    print(
        f"{fit.type_name} | samples {fit.sample_count}"
        f" | {len(fit.anchors)} anchors"
        f" | k-means ran {fit.iteration_count} iterations,"
        f" stopped by {stop_reason}"
    )
    print(
        f"{'anchor':>7}{'x':>10}{'y':>10}"
        f"{'distance':>10}{'angle deg':>11}{'share':>9}"
    )
    for index, (anchor, count) in enumerate(
        zip(fit.anchors, fit.counts, strict=True)
    ):
        offset_x, offset_y = anchor.tolist()
        distance = math.hypot(offset_x, offset_y)
        angle = math.degrees(math.atan2(offset_y, offset_x))
        share = int(count) / fit.sample_count
        print(
            f"{index:>7}{offset_x:>10.3f}{offset_y:>10.3f}"
            f"{distance:>10.3f}{angle:>11.1f}{share:>9.1%}"
        )
    largest_share = int(fit.counts.max()) / fit.sample_count
    print(f"{'largest single share':<28}{largest_share:>12.1%}")
    print(
        f"{'minimum pairwise distance':<28}"
        f"{minimum_pairwise_distance(fit.anchors):>12.3f}"
    )
    print()


def main() -> None:
    """Fits one set of 54 anchor endpoints per object type from the training
    futures and writes them with a provenance stamp.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()

    start = time.perf_counter()
    endpoints = reachable_only(cached_endpoints(arguments.staged_directory))
    print(
        f"samples {len(endpoints.positions)}"
        f" | {time.perf_counter() - start:.1f} s"
    )
    fits = [
        fit_type(type_name, endpoints.of_type(type_index))
        for type_index, type_name in enumerate(contract.PREDICTED_OBJECT_TYPES)
    ]
    for fit in fits:
        print_fit(fit)

    unit_anchors = torch.stack([fit.anchors for fit in fits])
    np.savez(
        arguments.output_path,
        unit_anchors=unit_anchors.numpy().astype(np.float32),
        provenance=contract.artifact_provenance(
            "fit_anchors.py", arguments.staged_directory
        ),
    )
    print(
        f"wrote {arguments.output_path},"
        f" unit_anchors {tuple(unit_anchors.shape)},"
        f" one set per {contract.PREDICTED_OBJECT_TYPES} in that order,"
        f" most-used anchor first"
    )


if __name__ == "__main__":
    main()
