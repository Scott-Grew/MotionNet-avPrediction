import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    get_worker_info,
)

from womd import loader


class ScenarioSampleStream(IterableDataset):
    def __init__(self, scenario_paths, seed, designated_targets_only):
        self.scenario_paths = scenario_paths
        self.seed = seed
        self.designated_targets_only = designated_targets_only

    def __iter__(self):
        worker_info = get_worker_info()
        worker_index = worker_info.id if worker_info else 0
        worker_count = worker_info.num_workers if worker_info else 1
        worker_paths = self.scenario_paths[worker_index::worker_count]
        random_generator = np.random.default_rng(
            self.seed + worker_index
        )
        for scenario_index in random_generator.permutation(
            len(worker_paths)
        ):
            scenario_arrays = loader.read_scenario(
                worker_paths[scenario_index]
            )
            sample_track_indices = loader.eligible_track_indices(
                scenario_arrays["track_rows"],
                scenario_arrays["track_valid"],
                scenario_arrays["is_designated_target"],
                self.designated_targets_only,
            )
            for track_index in random_generator.permutation(
                sample_track_indices
            ):
                yield loader.build_sample(
                    scenario_arrays, int(track_index)
                )


def collate_samples(samples):
    batch = loader.build_batch(samples)
    return {
        name: torch.from_numpy(array) for name, array in batch.items()
    }


def batches(
    scenario_paths,
    worker_count,
    batch_size,
    prefetch_batches,
    seed,
    designated_targets_only,
):
    return DataLoader(
        ScenarioSampleStream(
            scenario_paths, seed, designated_targets_only
        ),
        batch_size=batch_size,
        num_workers=worker_count,
        collate_fn=collate_samples,
        prefetch_factor=(
            prefetch_batches if worker_count > 0 else None
        ),
        pin_memory=torch.cuda.is_available(),
    )
