"""Streams scene batches for training through a shuffled multi-worker
DataLoader.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from womd import loader
from womd.loader import SceneBatch


class BatchSettings(NamedTuple):
    """How training batches are drawn from the staged scenarios."""

    batch_size: int  # predicted agents per batch
    worker_count: int
    prefetch_batches: int
    seed: int
    designated_only: bool


def torch_batch(batch: SceneBatch) -> SceneBatch:
    """The same batch with every numpy array turned into a torch tensor."""
    return batch.each(torch.from_numpy)


class SceneBatchStream(IterableDataset):
    """Streams finished batches that each hold exactly batch_size predicted
    agents, splitting scenario files across DataLoader workers.
    """

    def __init__(self, paths: list[Path], settings: BatchSettings) -> None:
        self.paths = paths
        self.settings = settings

    def __iter__(self) -> Iterator[SceneBatch]:
        """Shuffles this worker's scenarios and each one's eligible tracks with
        a per-worker seeded generator.

        A scene whose targets do not all fit in the batch being filled is
        split, so the ones that fit close this batch and the rest open the
        next, and the scene is encoded once in each.
        """
        worker = get_worker_info()
        worker_index = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        worker_paths = self.paths[worker_index::worker_count]
        generator = np.random.default_rng(self.settings.seed + worker_index)
        samples = []
        free_slots = self.settings.batch_size
        for scenario_index in generator.permutation(len(worker_paths)):
            scenario = loader.read_scenario(worker_paths[scenario_index])
            eligible = loader.eligible_tracks(
                scenario, self.settings.designated_only
            )
            waiting = generator.permutation(eligible).tolist()
            while waiting:
                taken = waiting[:free_slots]
                waiting = waiting[free_slots:]
                samples.append(loader.scene_sample(scenario, taken))
                free_slots -= len(taken)
                if free_slots:
                    continue
                yield loader.scene_batch(samples)
                samples = []
                free_slots = self.settings.batch_size
        if samples:
            yield loader.scene_batch(samples)


def batches(paths: list[Path], settings: BatchSettings) -> DataLoader:
    """Wraps SceneBatchStream in a DataLoader. batch_size is None because the
    stream already yields finished batches; the DataLoader only turns their
    numpy arrays into torch tensors.
    """
    prefetch = settings.prefetch_batches if settings.worker_count > 0 else None
    return DataLoader(
        SceneBatchStream(paths, settings),
        batch_size=None,
        num_workers=settings.worker_count,
        prefetch_factor=prefetch,
        pin_memory=torch.cuda.is_available(),
    )
