"""Streams predicted agents' inputs for training through a shuffled
multi-worker DataLoader, which stacks them into batches.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from womd import inputs, loader
from womd.inputs import TargetInputs


class BatchSettings(NamedTuple):
    batch_size: int  # predicted agents per batch
    worker_count: int
    prefetch_batches: int
    seed: int
    designated_only: bool


def torch_batch(batch: TargetInputs) -> TargetInputs:
    return batch.each(torch.from_numpy)


class TargetStream(IterableDataset):
    """Streams one TargetInputs per predicted agent, splitting scenario files
    across DataLoader workers.
    """

    def __init__(self, paths: list[Path], settings: BatchSettings) -> None:
        self.paths = paths
        self.settings = settings

    def __iter__(self) -> Iterator[TargetInputs]:
        """Shuffles this worker's scenarios and each one's eligible tracks with
        a per-worker seeded generator. A scene's map is chunked once for all
        of its targets.
        """
        worker = get_worker_info()
        worker_index = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        worker_paths = worker_share(self.paths, worker_index, worker_count)
        generator = np.random.default_rng(self.settings.seed + worker_index)
        designated_only = self.settings.designated_only
        for scenario_index in generator.permutation(len(worker_paths)):
            scenario = loader.read_scenario(worker_paths[scenario_index])
            eligible = loader.eligible_tracks(scenario, designated_only)
            chunks = inputs.scene_chunks(scenario)
            for target in generator.permutation(eligible):
                yield inputs.target_inputs(scenario, chunks, int(target))


def worker_share(paths: list[Path], index: int, count: int) -> list[Path]:
    return paths[index::count]


def steps_per_epoch(paths: list[Path], settings: BatchSettings) -> int:
    """Batches per epoch, the sum over workers of ceil(eligible targets /
    batch size), because each worker fills its own batches. Counting opens
    every staged file.
    """
    worker_count = max(settings.worker_count, 1)
    step_count = 0
    for index in range(worker_count):
        targets = 0
        for path in worker_share(paths, index, worker_count):
            scenario = loader.read_scenario(path)
            eligible = loader.eligible_tracks(
                scenario, settings.designated_only
            )
            targets += len(eligible)
        step_count += math.ceil(targets / settings.batch_size)
    return step_count


def batches(paths: list[Path], settings: BatchSettings) -> DataLoader:
    """Wraps TargetStream in a DataLoader, which stacks batch_size targets
    into one TargetInputs of torch tensors; each worker's last batch may be
    smaller.
    """
    prefetch = settings.prefetch_batches if settings.worker_count > 0 else None
    return DataLoader(
        TargetStream(paths, settings),
        batch_size=settings.batch_size,
        num_workers=settings.worker_count,
        prefetch_factor=prefetch,
        pin_memory=torch.cuda.is_available(),
    )
