"""Loads checkpoints and anchor files, refusing any artifact the current code
version did not produce.
"""
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
