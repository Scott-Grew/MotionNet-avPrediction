"""Step 5 of 5.

Host-side driver that writes predictions, then runs Waymo's own metrics inside
a linux/amd64 Docker container.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import subprocess
import sys
from pathlib import Path

# The repository is mounted into the scorer container at the mount
# point; the tag names the Docker image built from container/.
REPOSITORY_ROOT = Path(__file__).resolve().parent
CONTAINER_MOUNT_POINT = "/mnt"
CONTAINER_IMAGE_TAG = "waymo-scorer"
PLATFORM = ["--platform", "linux/amd64"]


def container_path(host_path: Path | str) -> str:
    """The same file inside the container. Only this repository is mounted, so
    the path must sit inside it.
    """
    resolved = Path(host_path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(
            f"{resolved} is outside {REPOSITORY_ROOT},"
            " the only folder the container sees"
        )
    return f"{CONTAINER_MOUNT_POINT}/{resolved.relative_to(REPOSITORY_ROOT)}"


def build_container_image() -> None:
    """Builds the linux/amd64 scorer image, since the waymo-open-dataset package
    used inside it has no macOS build.
    """
    container_directory = str(REPOSITORY_ROOT / "container")
    command = [
        "docker",
        "build",
        *PLATFORM,
        "-t",
        CONTAINER_IMAGE_TAG,
        container_directory,
    ]
    subprocess.run(command, check=True)


def run_in_container(runner_arguments: list[str]) -> None:
    """Runs container/runner.py inside the scorer image with the project root
    mounted, forwarding runner_arguments to it.
    """
    mount = f"{REPOSITORY_ROOT}:{CONTAINER_MOUNT_POINT}"
    runner = f"{CONTAINER_MOUNT_POINT}/container/runner.py"
    command = [
        "docker",
        "run",
        "--rm",
        *PLATFORM,
        "-v",
        mount,
        CONTAINER_IMAGE_TAG,
        "python3",
        runner,
    ]
    subprocess.run(command + runner_arguments, check=True)


def run_score(arguments: argparse.Namespace) -> None:
    """Runs submit.py on the host to produce predictions, then scores them
    inside the container against Waymo's own metrics.
    """
    output_directory = Path(arguments.output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = output_directory / "predictions.npz"
    submit = [
        sys.executable,
        str(REPOSITORY_ROOT / "submit.py"),
        str(Path(arguments.checkpoint_path).resolve()),
        str(Path(arguments.staged_directory).resolve()),
        str(predictions_path),
    ]
    subprocess.run(submit, check=True)
    build_container_image()
    staged = container_path(arguments.staged_directory)
    run_in_container(["score", container_path(predictions_path), staged])


def run_check_reader(arguments: argparse.Namespace) -> None:
    """Runs the container's field-by-field proto comparison against a shard on
    the host.
    """
    build_container_image()
    shard = container_path(arguments.shard_path)
    run_in_container(["check-reader", shard, str(arguments.sample_count)])


def main() -> None:
    """Drives the container from the host, either scoring a checkpoint's
    predictions or checking the vendored protos against Waymo's.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("checkpoint_path", type=Path)
    score.add_argument("staged_directory", type=Path)
    score.add_argument("output_directory", type=Path)
    check_reader = subparsers.add_parser("check-reader")
    check_reader.add_argument("shard_path", type=Path)
    check_reader.add_argument("sample_count", type=int)
    arguments = parser.parse_args()
    if arguments.command == "score":
        run_score(arguments)
    else:
        run_check_reader(arguments)


if __name__ == "__main__":
    main()
