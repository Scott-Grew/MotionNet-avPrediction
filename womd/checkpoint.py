"""Saves and loads checkpoints and anchor files, refusing any artifact the
current code version did not produce.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from womd import contract
from womd.model import QUERY_COUNT, MotionPredictor

Checkpoint = dict[str, Any]
Device = str | torch.device
GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
# The anchors live inside every checkpoint as a model buffer.
ANCHORS_KEY = "mode_decoder.unit_anchors"


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
    shapes = sorted(model_state.items())
    description = ",".join(f"{name}:{tuple(t.shape)}" for name, t in shapes)
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
    fingerprint = parameter_fingerprint(checkpoint["model_state"])
    if stamped is not None and stamped != fingerprint:
        raise ValueError(f"{path}: model_state does not match its fingerprint")
    return checkpoint


def load_predictor(path: Path | str) -> MotionPredictor:
    """The trained model in eval mode, built from the anchors its checkpoint
    holds, so no anchor file can disagree with the weights.
    """
    model_state = load_checkpoint(path)["model_state"]
    predictor = MotionPredictor(model_state[ANCHORS_KEY])
    predictor.load_state_dict(model_state)
    return predictor.eval()


def load_anchor_file(path: Path | str) -> torch.Tensor:
    """Loads the fitted per-type anchors from a .npz file and checks their
    provenance and shape; training starts from them.
    """
    with np.load(path) as anchors_file:
        stored = set(anchors_file.files)
        provenance = (
            anchors_file["provenance"] if "provenance" in stored else None
        )
        hint = "Refit them with fit_anchors.py."
        contract.check_provenance(provenance, path, hint)
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    expected_shape = (contract.NUM_OBJECT_TYPES, QUERY_COUNT, 2)
    if unit_anchors.shape != expected_shape:
        raise ValueError(
            f"{path} holds unit_anchors of shape {tuple(unit_anchors.shape)},"
            f" the model needs {expected_shape}"
        )
    return unit_anchors


def checkpoint_of(trained: TrainingObjects, progress: Progress) -> Checkpoint:
    """Bundles the model, optimiser and scaler state, the run's progress and
    a parameter fingerprint.
    """
    model_state = trained.predictor.state_dict()
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
    suffix = checkpoint_path.suffix + ".partial"
    partial_path = checkpoint_path.with_suffix(suffix)
    torch.save(state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_path(checkpoint_path))
    partial_path.replace(checkpoint_path)
