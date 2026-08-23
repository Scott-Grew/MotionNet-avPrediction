import womd.runtime_env
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from womd import contract, loader, model

MAXIMUM_ITERATIONS = 2000
SEED_GENERATOR_SEED = 0


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
        for track_index in eligible:
            sample = loader.build_sample(scenario_array, int(track_index))
            valid_future_steps = np.flatnonzero(sample["future_mask"])
            if valid_future_steps.size == 0:
                continue
            agent_histories.append(sample["agent_history"])
            logged_endpoints.append(sample["future_positions"][valid_future_steps[-1]])

    agent_history = torch.from_numpy(np.stack(agent_histories))
    endpoints = torch.from_numpy(np.stack(logged_endpoints))
    return (
        endpoints,
        model.predicted_type_index(agent_history),
        model.agent_reachable_distance(agent_history),
    )


def endpoints_per_centre(assignment, centre_count):
    counts = torch.zeros(centre_count, dtype=torch.long)
    return counts.index_add_(0, assignment, torch.ones_like(assignment))


def move_centres_to_assigned_means(endpoints, assignment, centres):
    totals = torch.zeros_like(centres).index_add_(0, assignment, endpoints)
    counts = endpoints_per_centre(assignment, centres.shape[0]).to(endpoints.dtype)
    means = totals / counts.clamp_min(1.0).unsqueeze(-1)
    return torch.where(counts.unsqueeze(-1) > 0.0, means, centres), counts


def reseed_empty_centres(endpoints, assignment, centres, counts):
    empty_centres = (counts == 0.0).nonzero(as_tuple=True)[0]
    if empty_centres.numel() == 0:
        return centres
    reseeded = centres.clone()
    splits_taken_from = torch.zeros_like(counts, dtype=torch.long)
    for empty_centre in empty_centres.tolist():
        busiest_centre = int((counts / (1.0 + splits_taken_from)).argmax())
        endpoints_of_busiest = endpoints[assignment == busiest_centre]
        spread_within_busiest = (
            endpoints_of_busiest - centres[busiest_centre]
        ).norm(dim=-1)
        spread_order = torch.argsort(spread_within_busiest, descending=True, stable=True)
        reseeded[empty_centre] = endpoints_of_busiest[
            spread_order[int(splits_taken_from[busiest_centre])]
        ]
        splits_taken_from[busiest_centre] += 1
    return reseeded


def spread_out_seed_centres(endpoints, centre_count, generator):
    first_seed = int(torch.randint(len(endpoints), (1,), generator=generator))
    centres = [endpoints[first_seed]]
    nearest_squared = (endpoints - centres[0]).pow(2).sum(dim=-1)
    while len(centres) < centre_count:
        probabilities = nearest_squared / nearest_squared.sum().clamp_min(1e-12)
        next_seed = int(torch.multinomial(probabilities, 1, generator=generator))
        centres.append(endpoints[next_seed])
        nearest_squared = torch.minimum(
            nearest_squared, (endpoints - centres[-1]).pow(2).sum(dim=-1)
        )
    return torch.stack(centres)


def fit_unit_anchors(endpoints, centre_count=model.QUERY_COUNT):
    generator = torch.Generator().manual_seed(SEED_GENERATOR_SEED)
    centres = spread_out_seed_centres(endpoints, centre_count, generator)
    initial_assignment = torch.cdist(endpoints, centres).argmin(dim=1)
    assignment = initial_assignment
    for iteration_count in range(1, MAXIMUM_ITERATIONS + 1):
        centres, counts = move_centres_to_assigned_means(endpoints, assignment, centres)
        centres = reseed_empty_centres(endpoints, assignment, centres, counts)
        next_assignment = torch.cdist(endpoints, centres).argmin(dim=1)
        if torch.equal(next_assignment, assignment):
            return centres, next_assignment, initial_assignment, iteration_count, True
        assignment = next_assignment
    return centres, assignment, initial_assignment, MAXIMUM_ITERATIONS, False


def minimum_pairwise_distance(anchors):
    separations = torch.cdist(anchors, anchors)
    separations.fill_diagonal_(float("inf"))
    return float(separations.min())


def largest_count_kept_apart_by_the_prune(endpoints, type_name):
    for centre_count in range(model.QUERY_COUNT, contract.NUM_PREDICTED_MODES - 1, -1):
        fitted = fit_unit_anchors(endpoints, centre_count)
        separation = minimum_pairwise_distance(fitted[0])
        print(
            f"  {type_name}: {centre_count} anchors, minimum pairwise {separation:.3f} m,"
            f" {'kept apart by' if separation >= model.PRUNE_DISTANCE_METRES else 'inside'}"
            f" the {model.PRUNE_DISTANCE_METRES} m prune radius"
        )
        if separation >= model.PRUNE_DISTANCE_METRES:
            return centre_count, fitted
    raise SystemExit(
        f"{type_name}: even {contract.NUM_PREDICTED_MODES} anchors sit inside the"
        f" {model.PRUNE_DISTANCE_METRES} m prune radius"
    )


def print_one_type(
    type_name, type_sample_count, fitted_anchors, fitted_counts,
    iteration_count, stopped_by_convergence,
):
    print(
        f"{type_name} | samples {type_sample_count} | {len(fitted_anchors)} anchors |"
        f" k-means ran {iteration_count} iterations, stopped by "
        f"{'an unchanged assignment' if stopped_by_convergence else f'the {MAXIMUM_ITERATIONS} iteration cap'}"
    )
    print(f"{'anchor':>7}{'x':>10}{'y':>10}{'distance':>10}{'angle deg':>11}{'share':>9}")
    for anchor_index in range(len(fitted_anchors)):
        offset_x, offset_y = fitted_anchors[anchor_index].tolist()
        print(
            f"{anchor_index:>7}{offset_x:>10.3f}{offset_y:>10.3f}"
            f"{math.hypot(offset_x, offset_y):>10.3f}"
            f"{math.degrees(math.atan2(offset_y, offset_x)):>11.1f}"
            f"{int(fitted_counts[anchor_index]) / type_sample_count:>9.1%}"
        )
    print(f"{'largest single share':<28}{int(fitted_counts.max()) / type_sample_count:>12.1%}")
    print(f"{'minimum pairwise distance':<28}{minimum_pairwise_distance(fitted_anchors):>12.3f}")
    print()


def main():
    staged_directory = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    endpoints_cache_path = staged_directory.parent / f"{staged_directory.name}_endpoints_cache.npz"

    start_seconds = time.perf_counter()
    cached_arrays = None
    if endpoints_cache_path.exists():
        with np.load(endpoints_cache_path) as endpoints_cache:
            cache_provenance = (
                endpoints_cache["provenance"] if "provenance" in endpoints_cache else None
            )
            try:
                contract.check_artifact_provenance(
                    cache_provenance, endpoints_cache_path, "Re-extracting endpoints now."
                )
                cached_arrays = {name: endpoints_cache[name] for name in
                                 ("endpoints", "predicted_type_index", "reachable_distance_metres")}
            except AssertionError as refusal:
                print(refusal)
    if cached_arrays is not None:
        endpoints = torch.from_numpy(cached_arrays["endpoints"])
        predicted_type_index = torch.from_numpy(cached_arrays["predicted_type_index"])
        reachable_distance_metres = torch.from_numpy(cached_arrays["reachable_distance_metres"])
        print(f"endpoints read from {endpoints_cache_path}")
    else:
        scenario_paths = sorted(staged_directory.glob("*.npz"))
        endpoints, predicted_type_index, reachable_distance_metres = metre_endpoints(scenario_paths)
        np.savez(
            endpoints_cache_path,
            endpoints=endpoints.numpy().astype(np.float32),
            predicted_type_index=predicted_type_index.numpy().astype(np.int64),
            reachable_distance_metres=reachable_distance_metres.numpy().astype(np.float32),
            provenance=contract.artifact_provenance("fit_anchors.py", staged_directory),
        )
        print(f"endpoints cached to {endpoints_cache_path}")
    elapsed_seconds = time.perf_counter() - start_seconds

    physically_reachable = endpoints.norm(dim=-1) <= reachable_distance_metres
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
        f" type, but " + ", ".join(starved_types) + f" over {len(endpoints)} endpoints of"
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

    unit_anchors = torch.zeros(contract.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2)
    anchor_counts = torch.zeros(contract.NUM_OBJECT_TYPES, dtype=torch.long)
    for type_index, type_name in enumerate(contract.PREDICTED_OBJECT_TYPES):
        type_endpoints = endpoints[predicted_type_index == type_index]
        centre_count, (
            centres, assignment, _, iteration_count, stopped_by_convergence
        ) = largest_count_kept_apart_by_the_prune(type_endpoints, type_name)
        fitted_counts = endpoints_per_centre(assignment, centre_count)
        share_order = torch.argsort(fitted_counts, descending=True, stable=True)
        fitted_anchors = centres[share_order]
        unit_anchors[type_index, :centre_count] = fitted_anchors
        anchor_counts[type_index] = centre_count
        print_one_type(
            type_name, type_endpoints.shape[0], fitted_anchors, fitted_counts[share_order],
            iteration_count, stopped_by_convergence,
        )

    np.savez(
        output_path,
        unit_anchors=unit_anchors.numpy().astype(np.float32),
        anchor_counts=anchor_counts.numpy().astype(np.int64),
        provenance=contract.artifact_provenance("fit_anchors.py", staged_directory),
    )
    print(
        f"wrote {output_path}, unit_anchors {tuple(unit_anchors.shape)} padded past each type's"
        f" count, anchor_counts {anchor_counts.tolist()} per {contract.PREDICTED_OBJECT_TYPES}"
        f" in that order, most-used anchor first"
    )


if __name__ == "__main__":
    main()
