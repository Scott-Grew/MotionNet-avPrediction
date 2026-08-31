import womd.runtime_env
import argparse
import subprocess
import sys
from pathlib import Path

WAYMO_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TREE_ROOT = Path(__file__).resolve().parent
CONTAINER_MOUNT_POINT = "/mnt"
CONTAINER_IMAGE_TAG = "waymo-scorer"


def container_path(host_path):
    resolved = Path(host_path).resolve()
    relative = resolved.relative_to(WAYMO_PROJECT_ROOT)
    return f"{CONTAINER_MOUNT_POINT}/{relative}"


def build_container_image():
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            CONTAINER_IMAGE_TAG,
            str(TREE_ROOT / "container"),
        ],
        check=True,
    )


def run_in_container(runner_arguments):
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{WAYMO_PROJECT_ROOT}:{CONTAINER_MOUNT_POINT}",
            CONTAINER_IMAGE_TAG,
            "python3",
            f"{CONTAINER_MOUNT_POINT}/{TREE_ROOT.name}/container/runner.py",
            *runner_arguments,
        ],
        check=True,
    )


def run_score(
    checkpoint_path, anchors_path, staged_directory, output_directory
):
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = output_directory / "predictions.npz"

    subprocess.run(
        [
            sys.executable,
            str(TREE_ROOT / "submit.py"),
            str(Path(checkpoint_path).resolve()),
            str(Path(staged_directory).resolve()),
            str(Path(anchors_path).resolve()),
            str(predictions_path),
        ],
        check=True,
    )

    build_container_image()
    run_in_container(
        [
            "score",
            container_path(predictions_path),
            container_path(staged_directory),
        ]
    )


def run_check_reader(shard_path, sample_count):
    build_container_image()
    run_in_container(
        [
            "check-reader",
            container_path(shard_path),
            str(sample_count),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("checkpoint_path", type=Path)
    score_parser.add_argument("anchors_path", type=Path)
    score_parser.add_argument("staged_directory", type=Path)
    score_parser.add_argument("output_directory", type=Path)

    check_reader_parser = subparsers.add_parser("check-reader")
    check_reader_parser.add_argument("shard_path", type=Path)
    check_reader_parser.add_argument("sample_count", type=int)

    arguments = parser.parse_args()

    if arguments.command == "score":
        run_score(
            arguments.checkpoint_path,
            arguments.anchors_path,
            arguments.staged_directory,
            arguments.output_directory,
        )
    elif arguments.command == "check-reader":
        run_check_reader(arguments.shard_path, arguments.sample_count)


if __name__ == "__main__":
    main()
