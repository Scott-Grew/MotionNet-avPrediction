"""Saves and loads checkpoints and anchor files, refusing any artifact the
current code version did not produce.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from womd import contract
from womd.model import MotionPredictor

Checkpoint = dict[str, Any]
Device = str | torch.device
GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
# The anchors live inside every checkpoint as a model buffer.
ANCHORS_KEY = "mode_decoder.unit_anchors"


def load_checkpoint(path: Path | str, device: Device = "cpu") -> Checkpoint:
    """Loads a checkpoint and checks its code version against the working
    tree; load_state_dict then refuses any parameter of the wrong shape.
    """
    checkpoint = torch.load(path, map_location=device)
    code_version = checkpoint.get("code_version")
    if code_version != contract.STAGING_CODE_VERSION:
        raise ValueError(
            f"{path} is from code version {code_version!r}, this tree is"
            f" {contract.STAGING_CODE_VERSION!r}"
        )
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
    provenance; the model checks their shape.
    """
    with np.load(path) as anchors_file:
        stored = set(anchors_file.files)
        provenance = (
            anchors_file["provenance"] if "provenance" in stored else None
        )
        hint = "Refit them with fit_anchors.py."
        contract.check_provenance(provenance, path, hint)
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    return unit_anchors


def previous_path(checkpoint_path: Path) -> Path:
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
