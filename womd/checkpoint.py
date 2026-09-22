"""Loads checkpoints and anchor files, refusing any artifact the current code
version did not produce.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from womd import contract
from womd.model import QUERY_COUNT


def parameter_fingerprint(model_state: dict[str, torch.Tensor]) -> str:
    """Hashes parameter names and shapes into one fingerprint used to detect a
    checkpoint saved under a different architecture.
    """
    parameter_description = ",".join(
        f"{name}:{tuple(tensor.shape)}"
        for name, tensor in sorted(model_state.items()))
    return hashlib.sha256(parameter_description.encode()).hexdigest()


def load_checkpoint_state(
        checkpoint_path: Path | str,
        map_location: str | torch.device = "cpu",
        allow_version_mismatch: bool = False) -> dict[str, Any]:
    """Loads a checkpoint and, unless overridden, verifies its code version and
    parameter fingerprint match the working tree.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_code_version = checkpoint.get("code_version")
    if not allow_version_mismatch:
        assert checkpoint_code_version == contract.STAGING_CODE_VERSION, (
            f"{checkpoint_path} is from code version"
            f" {checkpoint_code_version!r}, this tree is"
            f" {contract.STAGING_CODE_VERSION!r}; pass"
            f" allow_version_mismatch=True to load it anyway")
    stamped_fingerprint = checkpoint.get("parameter_fingerprint")
    if stamped_fingerprint is not None:
        recomputed_fingerprint = parameter_fingerprint(
            checkpoint["model_state"])
        assert stamped_fingerprint == recomputed_fingerprint, (
            f"{checkpoint_path}: model_state does not match its"
            f" parameter_fingerprint stamp")
    return checkpoint


def load_anchor_file(anchors_path: Path | str) -> torch.Tensor:
    """Loads the fitted per-type unit anchors from a .npz file and checks their
    provenance and shape before returning them.
    """
    with np.load(anchors_path) as anchors_file:
        contract.check_artifact_provenance(
            (anchors_file["provenance"]
             if "provenance" in anchors_file else None),
            anchors_path,
            "Refit them with fit_anchors.py.",
        )
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    assert unit_anchors.shape == (
        contract.NUM_OBJECT_TYPES,
        QUERY_COUNT,
        2,
    ), (f"{anchors_path} holds unit_anchors of shape"
        f" {tuple(unit_anchors.shape)}, but the model needs"
        f" ({contract.NUM_OBJECT_TYPES}, {QUERY_COUNT}, 2)."
        f" Re-run fit_anchors.py")
    return unit_anchors


def unwrapped(predictor: torch.nn.Module) -> torch.nn.Module:
    """The bare model under the multi-process and torch.compile wrappers, so
    a checkpoint holds plain parameter names however the run was launched.
    """
    if isinstance(predictor, DistributedDataParallel):
        predictor = predictor.module
    return getattr(predictor, "_orig_mod", predictor)


def checkpoint_state(
        predictor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gradient_scaler: torch.amp.GradScaler,
        *,
        seed: int,
        completed_epochs: int,
        epoch_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bundles the model, optimizer and scaler state, training progress and a
    parameter fingerprint into a checkpoint dict.
    """
    model_state = unwrapped(predictor).state_dict()
    return {
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": gradient_scaler.state_dict(),
        "completed_epochs": completed_epochs,
        "seed": seed,
        "code_version": contract.STAGING_CODE_VERSION,
        "parameter_fingerprint": parameter_fingerprint(model_state),
        "epoch_plan": epoch_plan,
    }


def save_checkpoint(checkpoint_path: Path, previous_checkpoint_path: Path,
                    state: dict[str, Any]) -> None:
    """Writes to a temporary file, then swaps it in, so a crash never truncates
    the checkpoint; the prior one is kept as a fallback.
    """
    partial_suffix = checkpoint_path.suffix + ".partial"
    partial_path = checkpoint_path.with_suffix(partial_suffix)
    torch.save(state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_checkpoint_path)
    partial_path.replace(checkpoint_path)
