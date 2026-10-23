"""Streams predicted agents' inputs for training through a shuffled
multi-worker DataLoader, which stacks them into batches.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import resource_tracker
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from womd import formats, views
from womd.views import TargetInputs

COUNT_REPORT_EVERY_FILES = 10000
COUNT_READ_THREADS = 128


class BatchSettings(NamedTuple):
    batch_size: int  # predicted agents per batch
    worker_count: int
    prefetch_batches: int
    seed: int
    designated_only: bool


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
        worker_paths = self.paths[worker_index::worker_count]
        generator = np.random.default_rng(self.settings.seed + worker_index)
        designated_only = self.settings.designated_only
        for scenario_index in generator.permutation(len(worker_paths)):
            scenario = formats.StagedFile.load(worker_paths[scenario_index])
            eligible = views.eligible_tracks(scenario, designated_only)
            chunks = views.scene_chunks(scenario)
            for target in generator.permutation(eligible):
                yield views.target_inputs(scenario, chunks, int(target))


def steps_per_epoch(paths: list[Path], settings: BatchSettings) -> int:
    """Batches per epoch, the sum over workers of ceil(eligible targets /
    batch size), because each worker fills its own batches. Counting reads
    three arrays from every staged file, on COUNT_READ_THREADS threads so
    the waits on each file's read overlap.
    """
    worker_count = max(settings.worker_count, 1)
    designated_only = itertools.repeat(settings.designated_only)
    start = time.perf_counter()
    targets_per_file = np.zeros(len(paths), dtype=np.int64)
    with ThreadPoolExecutor(COUNT_READ_THREADS) as executor:
        counts = executor.map(eligible_count, paths, designated_only)
        for file_index, count in enumerate(counts):
            targets_per_file[file_index] = count
            files_read = file_index + 1
            if files_read % COUNT_REPORT_EVERY_FILES == 0:
                seconds = time.perf_counter() - start
                print(
                    f"counted {files_read} of {len(paths)} files"
                    f" in {seconds:.0f} s"
                )
    step_count = 0
    for index in range(worker_count):
        targets = int(targets_per_file[index::worker_count].sum())
        step_count += math.ceil(targets / settings.batch_size)
    return step_count


def eligible_count(path: Path, designated_only: bool) -> int:
    tracks = formats.TrackArrays.load(path)
    eligible = views.eligible_tracks(tracks, designated_only)
    return len(eligible)


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
        worker_init_fn=keep_tracker_pipe_private,
    )


def keep_tracker_pipe_private(_worker_index: int) -> None:
    """Stops programs a worker starts from inheriting multiprocessing's
    resource-tracker pipe. On macOS each worker starts a torch_shm_manager,
    which would hold the pipe open while the main process holds a socket to
    it; the tracker then never sees the pipe close and the main process
    waits on the tracker forever at exit. The private attribute is read so
    that a worker without a tracker does not start one.
    """
    tracker_pipe = resource_tracker._resource_tracker._fd
    if tracker_pipe is not None:
        os.set_inheritable(tracker_pipe, False)
