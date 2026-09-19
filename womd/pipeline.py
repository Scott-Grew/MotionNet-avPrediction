import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    get_worker_info,
)

from womd import loader


class SceneBatchStream(IterableDataset):
    def __init__(
        self,
        scenario_paths,
        seed,
        designated_targets_only,
        targets_per_batch,
    ):
        self.scenario_paths = scenario_paths
        self.seed = seed
        self.designated_targets_only = designated_targets_only
        self.targets_per_batch = targets_per_batch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_index = worker_info.id if worker_info else 0
        worker_count = worker_info.num_workers if worker_info else 1
        worker_paths = self.scenario_paths[worker_index::worker_count]
        random_generator = np.random.default_rng(
            self.seed + worker_index
        )
        scene_samples = []
        free_target_slots = self.targets_per_batch
        for scenario_index in random_generator.permutation(
            len(worker_paths)
        ):
            scenario_arrays = loader.read_scenario(
                worker_paths[scenario_index]
            )
            waiting_track_indices = random_generator.permutation(
                loader.eligible_track_indices(
                    scenario_arrays["track_rows"],
                    scenario_arrays["track_valid"],
                    scenario_arrays["is_designated_target"],
                    self.designated_targets_only,
                )
            ).tolist()
            while waiting_track_indices:
                scene_samples.append(
                    loader.build_scene_sample(
                        scenario_arrays,
                        waiting_track_indices[:free_target_slots],
                    )
                )
                taken = len(scene_samples[-1]["targets"])
                waiting_track_indices = waiting_track_indices[taken:]
                free_target_slots -= taken
                if free_target_slots:
                    continue
                yield loader.build_scene_batch(scene_samples)
                scene_samples = []
                free_target_slots = self.targets_per_batch
        if scene_samples:
            yield loader.build_scene_batch(scene_samples)


def batches(
    scenario_paths,
    worker_count,
    batch_size,
    prefetch_batches,
    seed,
    designated_targets_only,
):
    return DataLoader(
        SceneBatchStream(
            scenario_paths, seed, designated_targets_only, batch_size
        ),
        batch_size=None,
        num_workers=worker_count,
        prefetch_factor=(
            prefetch_batches if worker_count > 0 else None
        ),
        pin_memory=torch.cuda.is_available(),
    )
