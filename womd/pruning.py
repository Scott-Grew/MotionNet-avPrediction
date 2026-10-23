"""Cuts the model's 54 futures to the 6 Waymo scores and moves the dropped
futures' probability onto the kept ones.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from womd import layout

# Two kept futures may not end closer together than this; a chosen
# value.
PRUNE_DISTANCE_METRES = 2.5


class PrunedModes(NamedTuple):
    """The kept modes of each sample, most confident first, and how many of
    the slots were kept rather than backfilled.
    """

    trajectories: Tensor  # (batch, 6, steps, 2)
    confidence_logits: Tensor  # (batch, 6)
    kept_count: Tensor  # (batch,)


class PruneWalk:
    """Per sample, the modes kept so far with their endpoints, and the modes
    dropped so far, as the walk down each sample's confidence order fills
    the output slots.
    """

    def __init__(self, endpoints: Tensor) -> None:
        batch_size = endpoints.shape[0]
        slot_count = layout.NUM_PREDICTED_MODES
        device = endpoints.device
        self.slot_positions = torch.arange(slot_count, device=device)
        self.kept_indices = torch.zeros(
            batch_size, slot_count, dtype=torch.long, device=device
        )
        self.kept_endpoints = endpoints.new_zeros(batch_size, slot_count, 2)
        self.kept_slot_filled = torch.zeros_like(
            self.kept_indices, dtype=torch.bool
        )
        self.kept_count = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self.dropped_indices = torch.zeros_like(self.kept_indices)
        self.dropped_count = torch.zeros_like(self.kept_count)

    def next_slot(self, flagged: Tensor, filled: Tensor) -> Tensor:
        """A one-hot mask over the slots giving, for each flagged sample, the
        next slot it has not filled yet.
        """
        is_next = self.slot_positions[None, :] == filled[:, None]
        return flagged[:, None] & is_next

    def consider(self, candidate: Tensor, endpoint: Tensor) -> None:
        """Keeps each sample's candidate mode unless it ends within the prune
        distance of a kept one, in which case it is dropped.
        """
        separations = torch.cdist(endpoint, self.kept_endpoints).squeeze(1)
        near_a_kept_mode = separations < PRUNE_DISTANCE_METRES
        too_close = (near_a_kept_mode & self.kept_slot_filled).any(dim=-1)
        still_walking = self.kept_count < len(self.slot_positions)

        keeps = still_walking & ~too_close
        keep_slot = self.next_slot(keeps, self.kept_count)
        self.kept_indices = torch.where(
            keep_slot, candidate[:, None], self.kept_indices
        )
        self.kept_endpoints = torch.where(
            keep_slot[:, :, None], endpoint, self.kept_endpoints
        )
        self.kept_slot_filled = self.kept_slot_filled | keep_slot
        self.kept_count = self.kept_count + keeps.long()

        drops = still_walking & too_close
        drop_slot = self.next_slot(drops, self.dropped_count)
        self.dropped_indices = torch.where(
            drop_slot, candidate[:, None], self.dropped_indices
        )
        self.dropped_count = self.dropped_count + drops.long()

    def final_indices(self) -> Tensor:
        """The kept modes, then dropped modes most confident first for any
        sample that kept fewer than the slot count.
        """
        slots = self.slot_positions[None, :]
        backfill_positions = (slots - self.kept_count[:, None]).clamp(min=0)
        is_kept_slot = slots < self.kept_count[:, None]
        backfill = self.dropped_indices.gather(1, backfill_positions)
        return torch.where(is_kept_slot, self.kept_indices, backfill)


def prune_modes(trajectories: Tensor, logits: Tensor) -> PrunedModes:
    """Walks each sample's modes from most to least confident, keeps one unless
    it ends near a kept one, then backfills to 6. The walk always visits every
    mode, so its length is fixed and it exports to ONNX; once 6 are kept, the
    later modes change nothing.
    """
    mode_count = logits.shape[1]
    endpoints = trajectories[:, :, -1]
    confidence_order = torch.argsort(logits, dim=-1, descending=True)
    walk = PruneWalk(endpoints)
    for position in range(mode_count):
        candidate = confidence_order[:, position]
        selector = candidate[:, None, None].expand(-1, -1, 2)
        walk.consider(candidate, endpoints.gather(1, selector))

    final_indices = walk.final_indices()
    trailing_shape = trajectories.shape[2:]
    selector = final_indices[:, :, None, None].expand(-1, -1, *trailing_shape)
    return PrunedModes(
        trajectories.gather(1, selector),
        logits.gather(1, final_indices),
        walk.kept_count,
    )


def kept_confidences(paths: Tensor, logits: Tensor, kept: Tensor) -> Tensor:
    """Moves each original mode's softmax probability mass onto whichever kept
    mode has the nearest endpoint.
    """
    probabilities = torch.softmax(logits, dim=-1)
    separations = torch.cdist(paths[:, :, -1], kept[:, :, -1])
    nearest_kept = separations.argmin(dim=-1)
    aggregated = torch.zeros(
        kept.shape[:2], dtype=probabilities.dtype, device=probabilities.device
    )
    return aggregated.scatter_add(1, nearest_kept, probabilities)


class SubmittedModes(NamedTuple):
    trajectories: Tensor  # (batch, 6, 16, 2) metres, agent frame
    confidences: Tensor  # (batch, 6) summing to 1


def submitted_modes(trajectories: Tensor, logits: Tensor) -> SubmittedModes:
    """The model's 54 modes at 10 Hz cut to Waymo's 6 at 2 Hz."""
    pruned = prune_modes(trajectories, logits)
    kept = pruned.trajectories
    confidences = kept_confidences(trajectories, logits, kept)
    indices = layout.SUBMISSION_FUTURE_INDICES
    steps = torch.tensor(indices, device=trajectories.device)
    return SubmittedModes(kept.index_select(2, steps), confidences)
