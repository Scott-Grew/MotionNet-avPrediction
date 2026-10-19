"""Saves and loads checkpoints and anchor files, refusing any artifact the
current code version did not produce.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from womd import contract
from womd.model import QUERY_COUNT

Checkpoint = dict[str, Any]
Device = str | torch.device
GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)


class TrainingObjects(NamedTuple):
    """The three stateful objects a training run carries between steps."""

    predictor: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scaler: GradScaler


class Progress(NamedTuple):
    """How far a run has got, as the checkpoint records it."""

    seed: int
    completed_epochs: int
    epoch_plan: dict[str, Any] | None


def parameter_fingerprint(model_state: dict[str, torch.Tensor]) -> str:
    """Hashes parameter names and shapes into one fingerprint used to detect a
    checkpoint saved under a different architecture.
    """
    description = ",".join(
        f"{name}:{tuple(tensor.shape)}"
        for name, tensor in sorted(model_state.items())
    )
    return hashlib.sha256(description.encode()).hexdigest()


def load_checkpoint(path: Path | str, device: Device = "cpu") -> Checkpoint:
    """Loads a checkpoint and checks its code version and parameter
    fingerprint against the working tree.
    """
    checkpoint = torch.load(path, map_location=device)
    code_version = checkpoint.get("code_version")
    if code_version != contract.STAGING_CODE_VERSION:
        raise ValueError(
            f"{path} is from code version {code_version!r}, this tree is"
            f" {contract.STAGING_CODE_VERSION!r}"
        )
    stamped = checkpoint.get("parameter_fingerprint")
    if stamped is not None and stamped != parameter_fingerprint(
        checkpoint["model_state"]
    ):
        raise ValueError(f"{path}: model_state does not match its fingerprint")
    return checkpoint


def load_anchor_file(path: Path | str) -> torch.Tensor:
    """Loads the fitted per-type anchors from a .npz file and checks their
    provenance and shape.
    """
    with np.load(path) as anchors_file:
        provenance = (
            anchors_file["provenance"] if "provenance" in anchors_file else None
        )
        contract.check_provenance(
            provenance, path, "Refit them with fit_anchors.py."
        )
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    expected_shape = (contract.NUM_OBJECT_TYPES, QUERY_COUNT, 2)
    if unit_anchors.shape != expected_shape:
        raise ValueError(
            f"{path} holds unit_anchors of shape {tuple(unit_anchors.shape)},"
            f" the model needs {expected_shape}"
        )
    return unit_anchors


def unwrapped(predictor: torch.nn.Module) -> torch.nn.Module:
    """The bare model under the multi-process wrapper, so a checkpoint holds
    plain parameter names whatever the process count.
    """
    if isinstance(predictor, DistributedDataParallel):
        return predictor.module
    return predictor


def checkpoint_of(trained: TrainingObjects, progress: Progress) -> Checkpoint:
    """Bundles the model, optimiser and scaler state, the run's progress and
    a parameter fingerprint.
    """
    model_state = unwrapped(trained.predictor).state_dict()
    return {
        "model_state": model_state,
        "optimizer_state": trained.optimizer.state_dict(),
        "gradient_scaler_state": trained.scaler.state_dict(),
        "completed_epochs": progress.completed_epochs,
        "seed": progress.seed,
        "code_version": contract.STAGING_CODE_VERSION,
        "parameter_fingerprint": parameter_fingerprint(model_state),
        "epoch_plan": progress.epoch_plan,
    }


def previous_path(checkpoint_path: Path) -> Path:
    """Where save_checkpoint keeps the prior checkpoint as a fallback."""
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".previous")


def save_checkpoint(checkpoint_path: Path, state: Checkpoint) -> None:
    """Writes to a temporary file, then swaps it in, so a crash never truncates
    the checkpoint; the prior one is kept as a fallback.
    """
    partial_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".partial"
    )
    torch.save(state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_path(checkpoint_path))
    partial_path.replace(checkpoint_path)
