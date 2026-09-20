"""Cuts the model's 54 futures to the 6 Waymo scores and moves the dropped
futures' probability onto the kept ones.
"""
import torch

from womd import contract

# Two kept futures may not end closer together than this; a chosen
# value.
PRUNE_DISTANCE_METRES = 2.5


def next_free_slot(is_flagged: torch.Tensor, filled_count: torch.Tensor,
                   slot_positions: torch.Tensor) -> torch.Tensor:
    """A one-hot mask over the output slots: for each flagged sample, the next
    slot it has not filled yet.
    """
    is_next_slot = slot_positions[None, :] == filled_count[:, None]
    return is_flagged[:, None] & is_next_slot


def prune_modes_batched_with_kept_count(
    trajectories: torch.Tensor, confidence_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walks each sample's modes from most to least confident, keeps one unless
    it ends near a kept one, then backfills to 6.
    """
    batch_size, mode_count = confidence_logits.shape
    slot_count = contract.NUM_PREDICTED_MODES
    device = trajectories.device
    endpoints = trajectories[:, :, -1]
    confidence_order = torch.argsort(confidence_logits, dim=-1, descending=True)
    slot_positions = torch.arange(slot_count, device=device)

    # Per sample: the modes kept so far, where they end, and the modes
    # dropped so far, each with a count of how many slots are filled.
    kept_indices = torch.zeros(batch_size,
                               slot_count,
                               dtype=torch.long,
                               device=device)
    kept_endpoints = torch.zeros_like(endpoints[:, :slot_count])
    kept_slot_filled = torch.zeros_like(kept_indices, dtype=torch.bool)
    kept_count = torch.zeros(batch_size, dtype=torch.long, device=device)
    dropped_indices = torch.zeros_like(kept_indices)
    dropped_count = torch.zeros_like(kept_count)

    for walk_position in range(mode_count):
        # Each sample's next candidate in its own confidence order.
        candidate_index = confidence_order[:, walk_position]
        candidate_endpoint = endpoints.gather(
            1, candidate_index[:, None, None].expand(-1, -1, 2))

        separations = torch.cdist(candidate_endpoint, kept_endpoints).squeeze(1)
        near_a_kept_mode = separations < PRUNE_DISTANCE_METRES
        too_close = (near_a_kept_mode & kept_slot_filled).any(dim=-1)
        still_walking = kept_count < slot_count

        keeps = still_walking & ~too_close
        keep_slot = next_free_slot(keeps, kept_count, slot_positions)
        kept_indices = torch.where(keep_slot, candidate_index[:, None],
                                   kept_indices)
        kept_endpoints = torch.where(keep_slot[:, :, None], candidate_endpoint,
                                     kept_endpoints)
        kept_slot_filled = kept_slot_filled | keep_slot
        kept_count = kept_count + keeps.long()

        drops = still_walking & too_close
        drop_slot = next_free_slot(drops, dropped_count, slot_positions)
        dropped_indices = torch.where(drop_slot, candidate_index[:, None],
                                      dropped_indices)
        dropped_count = dropped_count + drops.long()

        if bool((kept_count == slot_count).all()):
            break

    # A sample that kept fewer than 6 fills its remaining slots from
    # its dropped modes, most confident first.
    slots_past_the_kept = slot_positions[None, :] - kept_count[:, None]
    backfill_positions = slots_past_the_kept.clamp(min=0)
    is_kept_slot = slot_positions[None, :] < kept_count[:, None]
    backfill_indices = dropped_indices.gather(1, backfill_positions)
    final_indices = torch.where(is_kept_slot, kept_indices, backfill_indices)

    trailing_shape = trajectories.shape[2:]
    trajectory_selector = final_indices[:, :, None,
                                        None].expand(-1, -1, *trailing_shape)
    return (
        trajectories.gather(1, trajectory_selector),
        confidence_logits.gather(1, final_indices),
        kept_count,
    )


def prune_modes_batched(
        trajectories: torch.Tensor,
        confidence_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as prune_modes_batched_with_kept_count, without the kept-mode count.
    """
    kept_trajectories, kept_confidence_logits, _ = (
        prune_modes_batched_with_kept_count(trajectories, confidence_logits))
    return kept_trajectories, kept_confidence_logits


def aggregated_confidences(trajectories: torch.Tensor,
                           confidence_logits: torch.Tensor,
                           kept_trajectories: torch.Tensor) -> torch.Tensor:
    """Moves each original mode's softmax probability mass onto whichever kept
    mode has the nearest endpoint.
    """
    probabilities = torch.softmax(confidence_logits, dim=-1)
    separations = torch.cdist(trajectories[:, :, -1], kept_trajectories[:, :,
                                                                        -1])
    nearest_kept = separations.argmin(dim=-1)
    aggregated = torch.zeros(
        kept_trajectories.shape[:2],
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return aggregated.scatter_add(1, nearest_kept, probabilities)
