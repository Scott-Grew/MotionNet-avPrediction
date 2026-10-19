"""Step 1 of 5.

Reads raw Waymo shards and writes one .npz per scenario, in the scene frame, for
the later steps to read.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
from pathlib import Path

import tensorflow

from womd import store
from womd_protos import scenario_pb2

# A scenario whose logged timestamps stray this far from 10 Hz is
# counted as irregular in the staging report.
IRREGULAR_SPACING_SECONDS = 0.005


class Stager:
    """Stages shards into one output directory, skipping scenarios already
    staged there when skip_existing is set and refusing to overwrite them
    otherwise. It keeps what it wrote, what it skipped, and each written
    scenario's worst timestamp spacing deviation in seconds.
    """

    def __init__(self, output_directory: Path, skip_existing: bool) -> None:
        self.output_directory = output_directory
        self.skip_existing = skip_existing
        self.written: list[Path] = []
        self.skipped: list[Path] = []
        self.spacing_deviations: list[float] = []

    def shard(self, shard_path: Path) -> None:
        """Stages each scenario in one shard to its own .npz file; a scenario
        id seen twice is refused, since its file then already exists.
        """
        dataset = tensorflow.data.TFRecordDataset(str(shard_path))
        for record_bytes in dataset.as_numpy_iterator():
            scenario = scenario_pb2.Scenario.FromString(record_bytes)
            output_path = self.output_directory / f"{scenario.scenario_id}.npz"
            if output_path.exists():
                if not self.skip_existing:
                    raise FileExistsError(
                        f"output file already exists: {output_path}"
                    )
                self.skipped.append(output_path)
                continue
            store.staged_file(scenario).save(output_path)
            deviation = store.worst_spacing_deviation(scenario)
            self.spacing_deviations.append(deviation)
            self.written.append(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("shard_paths", type=Path, nargs="+")
    parser.add_argument("--skip-existing", action="store_true")
    arguments = parser.parse_args()

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    stager = Stager(
        arguments.output_directory, skip_existing=arguments.skip_existing
    )
    for shard_path in arguments.shard_paths:
        stager.shard(shard_path)
    irregular = sum(
        deviation >= IRREGULAR_SPACING_SECONDS
        for deviation in stager.spacing_deviations
    )
    worst = max(stager.spacing_deviations, default=0.0)
    if arguments.skip_existing:
        print(f"{len(stager.skipped)} scenarios already staged, skipped")
    print(
        f"{len(stager.written)} scenarios staged into"
        f" {arguments.output_directory}"
    )
    print(
        f"{irregular} with irregular timestep spacing"
        f" (worst deviation {worst:.4f} s)"
    )


if __name__ == "__main__":
    main()
