"""Step 5 of 5.

Host-side driver: writes predictions, then runs Waymo's own metrics inside a
linux/amd64 Docker container.
"""
from __future__ import annotations

import womd.runtime_env
import argparse
import subprocess
import sys
from pathlib import Path

# The repository is mounted into the scorer container at the mount
# point; the tag names the Docker image built from container/.
REPOSITORY_ROOT = Path(__file__).resolve().parent
CONTAINER_MOUNT_POINT = "/mnt"
CONTAINER_IMAGE_TAG = "waymo-scorer"


def container_path(host_path: Path | str) -> str:
    """Converts a host path to the same file inside the container.

    Only this repository is mounted, so the path must sit inside it.
    """
    resolved = Path(host_path).resolve()
    assert resolved.is_relative_to(REPOSITORY_ROOT), (
        f"{resolved} is outside {REPOSITORY_ROOT}, which is the"
        f" only folder the scorer container can see")
    relative = resolved.relative_to(REPOSITORY_ROOT)
    return f"{CONTAINER_MOUNT_POINT}/{relative}"


def build_container_image() -> None:
    """Builds the linux/amd64 scorer image, since the waymo-open-dataset package
    used inside it has no macOS build.
    """
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            CONTAINER_IMAGE_TAG,
            str(REPOSITORY_ROOT / "container"),
        ],
        check=True,
    )


def run_in_container(runner_arguments: list[str]) -> None:
    """Runs container/runner.py inside the scorer image with the project root
    mounted, forwarding runner_arguments to it.
    """
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{REPOSITORY_ROOT}:{CONTAINER_MOUNT_POINT}",
            CONTAINER_IMAGE_TAG,
            "python3",
            f"{CONTAINER_MOUNT_POINT}/container/runner.py",
            *runner_arguments,
        ],
        check=True,
    )


def run_score(checkpoint_path: Path, anchors_path: Path, staged_directory: Path,
              output_directory: Path) -> None:
    """Runs submit.py on the host to produce predictions, then scores them
    inside the container against Waymo's own metrics.
    """
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = output_directory / "predictions.npz"

    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "submit.py"),
            str(Path(checkpoint_path).resolve()),
            str(Path(staged_directory).resolve()),
            str(Path(anchors_path).resolve()),
            str(predictions_path),
        ],
        check=True,
    )

    build_container_image()
    run_in_container([
        "score",
        container_path(predictions_path),
        container_path(staged_directory),
    ])


def run_check_reader(shard_path: Path, sample_count: int) -> None:
    """Runs the container's field-by-field proto comparison against a shard on
    the host.
    """
    build_container_image()
    run_in_container([
        "check-reader",
        container_path(shard_path),
        str(sample_count),
    ])


def main() -> None:
    """Drives the container from the host: either scores a checkpoint's
    predictions or checks the vendored protos against Waymo's.
    """
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
