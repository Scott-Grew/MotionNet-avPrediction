import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

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
        random_generator = np.random.default_rng(self.seed + worker_index)
        for scenario_index in random_generator.permutation(len(worker_paths)):
            scenario_arrays = loader.read_scenario(worker_paths[scenario_index])
            sample_track_indices = loader.eligible_track_indices(
                scenario_arrays["track_rows"],
                scenario_arrays["track_valid"],
                scenario_arrays["is_designated_target"],
                self.designated_targets_only,
            )
            for track_index in random_generator.permutation(sample_track_indices):
                yield loader.build_sample(scenario_arrays, int(track_index))


def collate_samples(samples):
    batch = loader.build_batch(samples)
    return {name: torch.from_numpy(array) for name, array in batch.items()}


def track_sample_batches(staged_directory, batch_size, designated_targets_only, scenario_order=None):
    scenario_paths = sorted(staged_directory.glob("*.npz"))
    if scenario_order is not None:
        scenario_paths = [scenario_paths[scenario_index] for scenario_index in scenario_order]
    batch = []
    for scenario_path in scenario_paths:
        scenario_array = loader.read_scenario(scenario_path)
        for track_index in loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            designated_targets_only,
        ):
            track_index = int(track_index)
            batch.append((scenario_array, track_index, loader.build_sample(scenario_array, track_index)))
            if len(batch) < batch_size:
                continue
            yield batch
            batch = []
    if batch:
        yield batch


def batches(scenario_paths, worker_count, batch_size, prefetch_batches, seed, designated_targets_only):
    return DataLoader(
        ScenarioSampleStream(scenario_paths, seed, designated_targets_only),
        batch_size=batch_size,
        num_workers=worker_count,
        collate_fn=collate_samples,
        prefetch_factor=prefetch_batches if worker_count > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )
