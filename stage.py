import womd.runtime_env
import argparse
from pathlib import Path

from womd import store, tfrecord
from womd_protos import scenario_pb2


def stage_shard(shard_path, output_directory, skip_existing):
    written_paths = []
    spacing_deviations = []
    skipped_paths = []
    for scenario in tfrecord.read_scenarios(
        shard_path, scenario_pb2.Scenario
    ):
        output_path = (
            Path(output_directory) / f"{scenario.scenario_id}.npz"
        )
        if output_path.exists():
            assert (
                skip_existing
            ), f"output file already exists: {output_path}"
            skipped_paths.append(output_path)
            continue
        spacing_deviations.append(
            store.write_scenario(scenario, output_path)
        )
        written_paths.append(output_path)
    return written_paths, spacing_deviations, skipped_paths


def stage_shards(shard_paths, output_directory, skip_existing=False):
    written_paths = []
    spacing_deviations = []
    skipped_paths = []
    for shard_path in shard_paths:
        shard_written_paths, shard_deviations, shard_skipped_paths = (
            stage_shard(shard_path, output_directory, skip_existing)
        )
        written_paths.extend(shard_written_paths)
        spacing_deviations.extend(shard_deviations)
        skipped_paths.extend(shard_skipped_paths)
    distinct_paths = set(written_paths)
    assert len(distinct_paths) == len(
        written_paths
    ), f"{len(written_paths) - len(distinct_paths)} colliding output paths"
    if skip_existing:
        print(
            f"{len(skipped_paths)} scenarios already staged, skipped"
        )
    return written_paths, spacing_deviations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("shard_paths", type=Path, nargs="+")
    parser.add_argument("--skip-existing", action="store_true")
    arguments = parser.parse_args()

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    written_paths, spacing_deviations = stage_shards(
        arguments.shard_paths,
        arguments.output_directory,
        arguments.skip_existing,
    )
    irregular_count = sum(
        deviation >= 0.005 for deviation in spacing_deviations
    )
    print(
        f"{len(written_paths)} scenarios staged into {arguments.output_directory}"
    )
    print(
        f"{irregular_count} with irregular timestep spacing"
        f" (worst deviation {max(spacing_deviations, default=0.0):.4f} s)"
    )


if __name__ == "__main__":
    main()
