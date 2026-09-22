"""Streams scene batches for training through a shuffled multi-worker
DataLoader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    get_worker_info,
)

from womd import loader


class SceneBatchStream(IterableDataset):
    """Streams finished batches that each hold exactly targets_per_batch
    predicted agents, splitting scenario files across DataLoader workers.
    """

    def __init__(self, scenario_paths: list[Path], seed: int,
                 designated_targets_only: bool, targets_per_batch: int) -> None:
        self.scenario_paths = scenario_paths
        self.seed = seed
        self.designated_targets_only = designated_targets_only
        self.targets_per_batch = targets_per_batch

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """For this worker's scenario slice, shuffles scenarios and each one's
        eligible tracks with a per-worker seeded generator.

        A scene whose targets do not all fit in the batch being filled is
        split, so the ones that fit close this batch and the rest open the
        next, and the scene is encoded once in each.
        """
        worker_info = get_worker_info()
        worker_index = worker_info.id if worker_info else 0
        worker_count = worker_info.num_workers if worker_info else 1
        worker_paths = self.scenario_paths[worker_index::worker_count]
        random_generator = np.random.default_rng(self.seed + worker_index)
        scene_samples = []
        free_slots = self.targets_per_batch
        for scenario_index in random_generator.permutation(len(worker_paths)):
            scenario_arrays = loader.read_scenario(worker_paths[scenario_index])
            waiting_track_indices = random_generator.permutation(
                loader.eligible_track_indices(
                    scenario_arrays["track_rows"],
                    scenario_arrays["track_valid"],
                    scenario_arrays["is_designated_target"],
                    self.designated_targets_only,
                )).tolist()
            while waiting_track_indices:
                taken_track_indices = waiting_track_indices[:free_slots]
                waiting_track_indices = waiting_track_indices[free_slots:]
                scene_samples.append(
                    loader.build_scene_sample(scenario_arrays,
                                              taken_track_indices))
                free_slots -= len(taken_track_indices)
                if free_slots:
                    continue
                yield loader.build_scene_batch(scene_samples)
                scene_samples = []
                free_slots = self.targets_per_batch
        if scene_samples:
            yield loader.build_scene_batch(scene_samples)


def batches(scenario_paths: list[Path], worker_count: int, batch_size: int,
            prefetch_batches: int, seed: int,
            designated_targets_only: bool) -> DataLoader:
    """Wraps SceneBatchStream in a DataLoader with the given worker count and
    prefetching.

    batch_size=None because the stream already yields finished batches; the
    DataLoader only converts their numpy arrays to torch tensors.
    """
    return DataLoader(
        SceneBatchStream(scenario_paths, seed, designated_targets_only,
                         batch_size),
        batch_size=None,
        num_workers=worker_count,
        prefetch_factor=(prefetch_batches if worker_count > 0 else None),
        pin_memory=torch.cuda.is_available(),
    )
