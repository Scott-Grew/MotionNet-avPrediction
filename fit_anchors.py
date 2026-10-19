"""Step 2 of 5.

Clusters where training agents ended up after 8 s into 54 anchor endpoints per
object type.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

from womd import contract, loader, model

# Fixed, so a refit on the same endpoints gives the same anchors.
# Every other k-means setting is scikit-learn's default.
RANDOM_STATE_SEED = 0


def reachable_endpoints(paths: list[Path]) -> tuple[Tensor, Tensor]:
    """Every designated target's final valid future position (N, 2) in its
    own frame, and its predicted type (N,). Endpoints further away than the
    agent could have driven are left out; they stay training samples and
    only do not shape the anchors.
    """
    histories = []
    positions = []
    for path in paths:
        scenario = loader.read_scenario(path)
        eligible = loader.eligible_tracks(scenario, designated_only=True)
        if len(eligible) == 0:
            continue
        targets = loader.one_scene(scenario, eligible).targets
        for index in range(len(eligible)):
            valid_steps = np.flatnonzero(targets.future_mask[index])
            if valid_steps.size == 0:
                continue
            histories.append(targets.agent_history[index])
            positions.append(targets.future_positions[index, valid_steps[-1]])
    history = torch.from_numpy(np.stack(histories))
    endpoints = torch.from_numpy(np.stack(positions))
    type_index = model.predicted_type_index(history)
    reachable_metres = model.agent_reachable_distance(history)
    reachable = endpoints.norm(dim=-1) <= reachable_metres
    print(
        f"{len(endpoints)} endpoints, {int((~reachable).sum())} beyond"
        f" reachable distance left out of the fit"
    )
    return endpoints[reachable], type_index[reachable]


def fit_type(type_name: str, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Fits one object type's anchors (54, 2) with k-means and returns them
    most-used first, with how many endpoints each drew (54,).
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
    stop_reason = "convergence"
    if fitted.n_iter_ >= fitted.max_iter:
        stop_reason = "the iteration cap"
    separations = torch.cdist(centres, centres)
    separations.fill_diagonal_(float("inf"))
    print(
        f"{type_name} | samples {len(positions)}"
        f" | k-means ran {fitted.n_iter_} iterations, stopped by {stop_reason}"
        f" | largest share {int(counts.max()) / len(positions):.1%}"
        f" | closest two anchors {float(separations.min()):.3f} m"
    )
    return centres[order], counts[order]


def main() -> None:
    """Fits one set of 54 anchor endpoints per object type from the training
    futures and writes them with a provenance stamp.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()

    paths = sorted(arguments.staged_directory.glob("*.npz"))
    endpoints, type_index = reachable_endpoints(paths)
    per_type_anchors = []
    for index, type_name in enumerate(contract.PREDICTED_OBJECT_TYPES):
        anchors, _ = fit_type(type_name, endpoints[type_index == index])
        per_type_anchors.append(anchors)

    unit_anchors = torch.stack(per_type_anchors)
    provenance = contract.artifact_provenance(
        "fit_anchors.py", arguments.staged_directory
    )
    np.savez(
        arguments.output_path,
        unit_anchors=unit_anchors.numpy().astype(np.float32),
        provenance=provenance,
    )
    print(
        f"wrote {arguments.output_path}, one set of anchors per"
        f" {contract.PREDICTED_OBJECT_TYPES} in that order, most-used first"
    )


if __name__ == "__main__":
    main()
