import womd.runtime_env
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from womd import contract, loader, model

MAXIMUM_ITERATIONS = 2000
RANDOM_STATE_SEED = 0
RESTART_COUNT = 10


def metre_endpoints(scenario_paths):
    agent_histories = []
    logged_endpoints = []
    for scenario_path in scenario_paths:
        scenario_array = loader.read_scenario(scenario_path)
        eligible = loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        )
        for target in loader.build_scene_sample(
            scenario_array, eligible.tolist()
        )["targets"]:
            valid_future_steps = np.flatnonzero(target["future_mask"])
            if valid_future_steps.size == 0:
                continue
            agent_histories.append(target["agent_history"])
            logged_endpoints.append(
                target["future_positions"][valid_future_steps[-1]]
            )

    agent_history = torch.from_numpy(np.stack(agent_histories))
    endpoints = torch.from_numpy(np.stack(logged_endpoints))
    return (
        endpoints,
        model.predicted_type_index(agent_history),
        model.agent_reachable_distance(agent_history),
    )


def endpoints_per_centre(assignment, centre_count):
    counts = torch.zeros(centre_count, dtype=torch.long)
    return counts.index_add_(
        0, assignment, torch.ones_like(assignment)
    )


def fit_unit_anchors(endpoints, centre_count=model.QUERY_COUNT):
    fitted = KMeans(
        n_clusters=centre_count,
        init="k-means++",
        n_init=RESTART_COUNT,
        max_iter=MAXIMUM_ITERATIONS,
        tol=0.0,
        random_state=RANDOM_STATE_SEED,
    ).fit(endpoints.numpy())
    return (
        torch.from_numpy(fitted.cluster_centers_).to(endpoints.dtype),
        torch.from_numpy(fitted.labels_).to(torch.long),
        fitted.n_iter_,
        fitted.n_iter_ < MAXIMUM_ITERATIONS,
    )


def minimum_pairwise_distance(anchors):
    separations = torch.cdist(anchors, anchors)
    separations.fill_diagonal_(float("inf"))
    return float(separations.min())


def print_one_type(
    type_name,
    type_sample_count,
    fitted_anchors,
    fitted_counts,
    iteration_count,
    stopped_by_convergence,
):
    print(
        f"{type_name} | samples {type_sample_count} | {len(fitted_anchors)} anchors |"
        f" k-means ran {iteration_count} iterations, stopped by "
        f"{'an unchanged assignment' if stopped_by_convergence else f'the {MAXIMUM_ITERATIONS} iteration cap'}"
    )
    print(
        f"{'anchor':>7}{'x':>10}{'y':>10}{'distance':>10}{'angle deg':>11}{'share':>9}"
    )
    for anchor_index in range(len(fitted_anchors)):
        offset_x, offset_y = fitted_anchors[anchor_index].tolist()
        print(
            f"{anchor_index:>7}{offset_x:>10.3f}{offset_y:>10.3f}"
            f"{math.hypot(offset_x, offset_y):>10.3f}"
            f"{math.degrees(math.atan2(offset_y, offset_x)):>11.1f}"
            f"{int(fitted_counts[anchor_index]) / type_sample_count:>9.1%}"
        )
    print(
        f"{'largest single share':<28}{int(fitted_counts.max()) / type_sample_count:>12.1%}"
    )
    print(
        f"{'minimum pairwise distance':<28}{minimum_pairwise_distance(fitted_anchors):>12.3f}"
    )
    print()


def main():
    staged_directory = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    endpoints_cache_path = (
        staged_directory.parent
        / f"{staged_directory.name}_endpoints_cache.npz"
    )

    start_seconds = time.perf_counter()
    cached_arrays = None
    if endpoints_cache_path.exists():
        with np.load(endpoints_cache_path) as endpoints_cache:
            cache_provenance = (
                endpoints_cache["provenance"]
                if "provenance" in endpoints_cache
                else None
            )
            try:
                contract.check_artifact_provenance(
                    cache_provenance,
                    endpoints_cache_path,
                    "Re-extracting endpoints now.",
                )
                cached_arrays = {
                    name: endpoints_cache[name]
                    for name in (
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
            cached_arrays["predicted_type_index"]
        )
        reachable_distance_metres = torch.from_numpy(
            cached_arrays["reachable_distance_metres"]
        )
        print(f"endpoints read from {endpoints_cache_path}")
    else:
        scenario_paths = sorted(staged_directory.glob("*.npz"))
        endpoints, predicted_type_index, reachable_distance_metres = (
            metre_endpoints(scenario_paths)
        )
        np.savez(
            endpoints_cache_path,
            endpoints=endpoints.numpy().astype(np.float32),
            predicted_type_index=predicted_type_index.numpy().astype(
                np.int64
            ),
            reachable_distance_metres=reachable_distance_metres.numpy().astype(
                np.float32
            ),
            provenance=contract.artifact_provenance(
                "fit_anchors.py", staged_directory
            ),
        )
        print(f"endpoints cached to {endpoints_cache_path}")
    elapsed_seconds = time.perf_counter() - start_seconds

    physically_reachable = (
        endpoints.norm(dim=-1) <= reachable_distance_metres
    )
    excluded_count = int((~physically_reachable).sum())
    endpoints = endpoints[physically_reachable]
    predicted_type_index = predicted_type_index[physically_reachable]
    print(
        f"{excluded_count} endpoints past their own reachable budget excluded from the fit;"
        f" they remain training samples and assign to their nearest fitted anchor"
    )

    type_sample_counts = [
        int((predicted_type_index == type_index).sum())
        for type_index in range(contract.NUM_OBJECT_TYPES)
    ]
    starved_types = [
        f"{type_name} has {type_sample_count}"
        for type_name, type_sample_count in zip(
            contract.PREDICTED_OBJECT_TYPES, type_sample_counts
        )
        if type_sample_count < model.QUERY_COUNT
    ]
    assert not starved_types, (
        f"a {model.QUERY_COUNT}-anchor search needs at least {model.QUERY_COUNT} endpoints per"
        f" type, but "
        + ", ".join(starved_types)
        + f" over {len(endpoints)} endpoints of"
        f" {staged_directory}"
    )

    print(f"samples {endpoints.shape[0]} | {elapsed_seconds:.1f} s")
    print(
        "samples per type | "
        + " ".join(
            f"{type_name} {type_sample_count}"
            for type_name, type_sample_count in zip(
                contract.PREDICTED_OBJECT_TYPES, type_sample_counts
            )
        )
    )
    print()

    fitted_anchors_per_type = []
    for type_index, type_name in enumerate(
        contract.PREDICTED_OBJECT_TYPES
    ):
        type_endpoints = endpoints[predicted_type_index == type_index]
        (
            centres,
            assignment,
            iteration_count,
            stopped_by_convergence,
        ) = fit_unit_anchors(type_endpoints)
        fitted_counts = endpoints_per_centre(
            assignment, model.QUERY_COUNT
        )
        share_order = torch.argsort(
            fitted_counts, descending=True, stable=True
        )
        fitted_anchors = centres[share_order]
        fitted_anchors_per_type.append(fitted_anchors)
        print_one_type(
            type_name,
            type_endpoints.shape[0],
            fitted_anchors,
            fitted_counts[share_order],
            iteration_count,
            stopped_by_convergence,
        )

    unit_anchors = torch.stack(fitted_anchors_per_type)
    np.savez(
        output_path,
        unit_anchors=unit_anchors.numpy().astype(np.float32),
        provenance=contract.artifact_provenance(
            "fit_anchors.py", staged_directory
        ),
    )
    print(
        f"wrote {output_path}, unit_anchors {tuple(unit_anchors.shape)}, one set per"
        f" {contract.PREDICTED_OBJECT_TYPES} in that order, most-used anchor first"
    )


if __name__ == "__main__":
    main()
