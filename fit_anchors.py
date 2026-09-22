"""Step 2 of 5.

Clusters where training agents ended up after 8 s into 54 anchor endpoints per
object type.
"""
from __future__ import annotations

import womd.runtime_env
import argparse
import math
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import torch

from womd import contract, loader, model

# The result of one k-means fit, where the centres ended, which centre
# each endpoint belongs to and how the search stopped.
AnchorFit = namedtuple(
    "AnchorFit",
    "centres assignment iteration_count stopped_by_convergence",
)

# Fixed, so a refit on the same endpoints gives the same anchors.
# Every other k-means setting is scikit-learn's default.
RANDOM_STATE_SEED = 0


def metre_endpoints(
    scenario_paths: list[Path]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each designated target, its final valid future position, predicted
    type index and reachable-distance bound in metres.
    """
    agent_histories = []
    logged_endpoints = []
    for scenario_path in scenario_paths:
        scenario_array = loader.read_scenario(scenario_path)
        eligible = loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            designated_targets_only=True,
        )
        scene_sample = loader.build_scene_sample(scenario_array,
                                                 eligible.tolist())
        for target in scene_sample.targets:
            valid_future_steps = np.flatnonzero(target.future_mask)
            if valid_future_steps.size == 0:
                continue
            agent_histories.append(target.agent_history)
            logged_endpoints.append(
                target.future_positions[valid_future_steps[-1]])

    agent_history = torch.from_numpy(np.stack(agent_histories))
    endpoints = torch.from_numpy(np.stack(logged_endpoints))
    return (
        endpoints,
        model.predicted_type_index(agent_history),
        model.agent_reachable_distance(agent_history),
    )


def endpoints_per_centre(assignment: torch.Tensor,
                         centre_count: int) -> torch.Tensor:
    endpoint_counts = torch.zeros(centre_count, dtype=torch.long)
    return endpoint_counts.index_add_(0, assignment,
                                      torch.ones_like(assignment))


def fit_unit_anchors(endpoints: torch.Tensor,
                     centre_count: int = model.QUERY_COUNT) -> AnchorFit:
    fitted = KMeans(n_clusters=centre_count,
                    random_state=RANDOM_STATE_SEED).fit(endpoints.numpy())
    return AnchorFit(
        centres=torch.from_numpy(fitted.cluster_centers_).to(endpoints.dtype),
        assignment=torch.from_numpy(fitted.labels_).to(torch.long),
        iteration_count=fitted.n_iter_,
        stopped_by_convergence=fitted.n_iter_ < fitted.max_iter,
    )


def minimum_pairwise_distance(anchors: torch.Tensor) -> float:
    """Smallest distance between any two distinct anchors, in metres."""
    separations = torch.cdist(anchors, anchors)
    separations.fill_diagonal_(float("inf"))
    return float(separations.min())


def print_one_type(type_name: str, type_sample_count: int,
                   fitted_anchors: torch.Tensor, fitted_counts: torch.Tensor,
                   iteration_count: int, stopped_by_convergence: bool) -> None:
    """Prints one object type's fitted anchors with their offset, distance,
    angle and share of endpoints assigned to them.
    """
    stop_reason = ("convergence"
                   if stopped_by_convergence else "the iteration cap")
    print(f"{type_name} | samples {type_sample_count} |"
          f" {len(fitted_anchors)} anchors | k-means ran"
          f" {iteration_count} iterations, stopped by {stop_reason}")
    print(f"{'anchor':>7}{'x':>10}{'y':>10}{'distance':>10}"
          f"{'angle deg':>11}{'share':>9}")
    for anchor_index in range(len(fitted_anchors)):
        offset_x, offset_y = fitted_anchors[anchor_index].tolist()
        print(f"{anchor_index:>7}{offset_x:>10.3f}{offset_y:>10.3f}"
              f"{math.hypot(offset_x, offset_y):>10.3f}"
              f"{math.degrees(math.atan2(offset_y, offset_x)):>11.1f}"
              f"{int(fitted_counts[anchor_index]) / type_sample_count:>9.1%}")
    largest_share = int(fitted_counts.max()) / type_sample_count
    print(f"{'largest single share':<28}{largest_share:>12.1%}")
    print(f"{'minimum pairwise distance':<28}"
          f"{minimum_pairwise_distance(fitted_anchors):>12.3f}")
    print()


def load_or_extract_endpoints(
        staged_directory: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracting endpoints reads the whole staged directory, so the result is
    cached beside the staged directory and reused while valid.
    """
    cache_name = f"{staged_directory.name}_endpoints_cache.npz"
    endpoints_cache_path = staged_directory.parent / cache_name
    cached_arrays = None
    if endpoints_cache_path.exists():
        with np.load(endpoints_cache_path) as endpoints_cache:
            cache_provenance = (endpoints_cache["provenance"]
                                if "provenance" in endpoints_cache else None)
            try:
                contract.check_artifact_provenance(
                    cache_provenance,
                    endpoints_cache_path,
                    "Re-extracting endpoints now.",
                )
                cached_arrays = {
                    name: endpoints_cache[name] for name in (
                        "endpoints",
                        "predicted_type_index",
                        "reachable_distance_metres",
                    )
                }
            except AssertionError as refusal:
                print(refusal)
    if cached_arrays is not None:
        endpoints = torch.from_numpy(cached_arrays["endpoints"])
        predicted_type_index = torch.from_numpy(
            cached_arrays["predicted_type_index"])
        reachable_distance_metres = torch.from_numpy(
            cached_arrays["reachable_distance_metres"])
        print(f"endpoints read from {endpoints_cache_path}")
    else:
        scenario_paths = sorted(staged_directory.glob("*.npz"))
        endpoints, predicted_type_index, reachable_distance_metres = (
            metre_endpoints(scenario_paths))
        reachable_distances = reachable_distance_metres.numpy()
        cached_endpoint_arrays = {
            "endpoints": endpoints.numpy().astype(np.float32),
            "predicted_type_index": predicted_type_index.numpy().astype(np.int64
                                                                       ),
            "reachable_distance_metres": reachable_distances.astype(np.float32),
            "provenance": contract.artifact_provenance("fit_anchors.py",
                                                       staged_directory),
        }
        np.savez(endpoints_cache_path, **cached_endpoint_arrays)
        print(f"endpoints cached to {endpoints_cache_path}")
    return endpoints, predicted_type_index, reachable_distance_metres


def drop_unreachable_endpoints(
    endpoints: torch.Tensor, predicted_type_index: torch.Tensor,
    reachable_distance_metres: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaves out endpoints further away than their agent could have driven.

    They stay training samples; they only do not shape the anchors.
    """
    physically_reachable = endpoints.norm(dim=-1) <= reachable_distance_metres
    excluded_count = int((~physically_reachable).sum())
    print(f"{excluded_count} endpoints beyond reachable distance"
          f" left out of the fit")
    return (endpoints[physically_reachable],
            predicted_type_index[physically_reachable])


def count_endpoints_per_type(predicted_type_index: torch.Tensor,
                             staged_directory: Path) -> list[int]:
    """Counts the endpoints of each object type and refuses a type with fewer
    endpoints than anchors.
    """
    type_sample_counts = [
        int((predicted_type_index == type_index).sum())
        for type_index in range(contract.NUM_OBJECT_TYPES)
    ]
    starved_types = [
        f"{type_name} has {type_sample_count}" for type_name, type_sample_count
        in zip(contract.PREDICTED_OBJECT_TYPES, type_sample_counts)
        if type_sample_count < model.QUERY_COUNT
    ]
    starved_summary = ", ".join(starved_types)
    assert not starved_types, (
        f"each type needs at least {model.QUERY_COUNT} endpoints, but"
        f" {starved_summary} in {staged_directory}")
    return type_sample_counts


def fit_one_type(type_name: str, type_endpoints: torch.Tensor) -> torch.Tensor:
    """Fits one object type's anchors, orders them most-used first, and prints
    the table for that type.
    """
    anchor_fit = fit_unit_anchors(type_endpoints)
    fitted_counts = endpoints_per_centre(anchor_fit.assignment,
                                         model.QUERY_COUNT)
    share_order = torch.argsort(fitted_counts, descending=True, stable=True)
    fitted_anchors = anchor_fit.centres[share_order]
    print_one_type(
        type_name,
        type_endpoints.shape[0],
        fitted_anchors,
        fitted_counts[share_order],
        anchor_fit.iteration_count,
        anchor_fit.stopped_by_convergence,
    )
    return fitted_anchors


def main() -> None:
    """Fits one set of 54 anchor endpoints per object type from the training
    futures and writes them with a provenance stamp.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()
    staged_directory = arguments.staged_directory
    output_path = arguments.output_path

    start_seconds = time.perf_counter()
    endpoints, predicted_type_index, reachable_distance_metres = (
        load_or_extract_endpoints(staged_directory))
    elapsed_seconds = time.perf_counter() - start_seconds

    endpoints, predicted_type_index = drop_unreachable_endpoints(
        endpoints, predicted_type_index, reachable_distance_metres)
    type_sample_counts = count_endpoints_per_type(predicted_type_index,
                                                  staged_directory)

    print(f"samples {endpoints.shape[0]} | {elapsed_seconds:.1f} s")
    per_type_summary = " ".join(
        f"{type_name} {type_sample_count}" for type_name, type_sample_count in
        zip(contract.PREDICTED_OBJECT_TYPES, type_sample_counts))
    print(f"samples per type | {per_type_summary}")
    print()

    fitted_anchors_per_type = [
        fit_one_type(type_name, endpoints[predicted_type_index == type_index])
        for type_index, type_name in enumerate(contract.PREDICTED_OBJECT_TYPES)
    ]

    unit_anchors = torch.stack(fitted_anchors_per_type)
    anchor_arrays = {
        "unit_anchors": unit_anchors.numpy().astype(np.float32),
        "provenance": contract.artifact_provenance("fit_anchors.py",
                                                   staged_directory),
    }
    np.savez(output_path, **anchor_arrays)
    print(f"wrote {output_path}, unit_anchors"
          f" {tuple(unit_anchors.shape)}, one set per"
          f" {contract.PREDICTED_OBJECT_TYPES} in that order,"
          f" most-used anchor first")


if __name__ == "__main__":
    main()
