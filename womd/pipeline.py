"""Streams loader samples into training batches through a shuffled multi-worker
DataLoader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    get_worker_info,
)

from womd import loader


class ScenarioSampleStream(IterableDataset):
    """Streams per-agent training samples, splitting scenario files across
    DataLoader workers.
    """

    def __init__(self, scenario_paths: list[Path], seed: int,
                 designated_targets_only: bool) -> None:
        """Stores the scenario paths, seed and target-filtering flag."""
        self.scenario_paths = scenario_paths
        self.seed = seed
        self.designated_targets_only = designated_targets_only

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """For this worker's scenario slice, shuffles scenarios and each one's
        eligible tracks with a per-worker seeded generator.
        """
        worker_info = get_worker_info()
        worker_index = worker_info.id if worker_info else 0
        worker_count = worker_info.num_workers if worker_info else 1
        worker_paths = self.scenario_paths[worker_index::worker_count]
        random_generator = np.random.default_rng(self.seed + worker_index)
        for scenario_index in random_generator.permutation(len(worker_paths)):
            scenario_arrays = loader.read_scenario(worker_paths[scenario_index])
            sample_track_indices = loader.eligible_track_indices(
                scenario_arrays["track_rows"],
                scenario_arrays["track_valid"],
                scenario_arrays["is_designated_target"],
                self.designated_targets_only,
            )
            for track_index in random_generator.permutation(
                    sample_track_indices):
                yield loader.build_sample(scenario_arrays, int(track_index))


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Builds a batch from build_batch and converts its numpy arrays to torch
    tensors.
    """
    batch = loader.build_batch(samples)
    return {name: torch.from_numpy(array) for name, array in batch.items()}


def batches(scenario_paths: list[Path], worker_count: int, batch_size: int,
            prefetch_batches: int, seed: int,
            designated_targets_only: bool) -> DataLoader:
    """Wraps ScenarioSampleStream in a DataLoader with the given worker count,
    batch size, and prefetching.
    """
    return DataLoader(
        ScenarioSampleStream(scenario_paths, seed, designated_targets_only),
        batch_size=batch_size,
        num_workers=worker_count,
        collate_fn=collate_samples,
        prefetch_factor=(prefetch_batches if worker_count > 0 else None),
        pin_memory=torch.cuda.is_available(),
    )
