"""Plain reference versions of the anchors and the pruning rule, kept only for
the tests to compare against.
"""
import torch

from womd import contract
from womd.model import QUERY_COUNT
from womd.pruning import PRUNE_DISTANCE_METRES

ANCHOR_DIRECTION_COUNT = 9
ANCHOR_DISTANCE_COUNT = 6
assert ANCHOR_DIRECTION_COUNT * ANCHOR_DISTANCE_COUNT == QUERY_COUNT


def unit_anchor_offsets():
    """Builds the 54 unit-length anchor endpoints as a grid of 9 directions by 6
    distance fractions.
    """
    # Direction repeats slowest, distance fastest, so the flat
    # index matches ANCHOR_DIRECTION_COUNT * ANCHOR_DISTANCE_COUNT.
    direction_indices = torch.arange(ANCHOR_DIRECTION_COUNT).repeat_interleave(
        ANCHOR_DISTANCE_COUNT)
    distance_indices = torch.arange(ANCHOR_DISTANCE_COUNT).repeat(
        ANCHOR_DIRECTION_COUNT)
    angles = 2 * torch.pi * direction_indices / ANCHOR_DIRECTION_COUNT
    fractions = (distance_indices + 1) / ANCHOR_DISTANCE_COUNT
    return fractions.unsqueeze(-1) * torch.stack(
        [angles.cos(), angles.sin()], dim=-1)


def unit_anchor_offsets_per_type():
    """Repeats the same 54 unit anchors once per object type, giving each type
    the same starting anchor set.
    """
    return unit_anchor_offsets().repeat(contract.NUM_OBJECT_TYPES, 1, 1)


def prune_modes_single_sample(trajectories, confidence_logits):
    """The pruning walk for one sample, written as a plain loop; the batched
    version in womd/pruning.py must match it exactly.
    """
    kept_indices = []
    dropped_indices = []
    endpoints = trajectories[:, -1]
    for mode_index in torch.argsort(confidence_logits,
                                    descending=True).tolist():
        if kept_indices and bool((torch.cdist(
                endpoints[mode_index][None],
                endpoints[kept_indices],
        ) < PRUNE_DISTANCE_METRES).any()):
            dropped_indices.append(mode_index)
            continue
        kept_indices.append(mode_index)
        if len(kept_indices) == contract.NUM_PREDICTED_MODES:
            break
    # If suppression kept too few modes, pad back up to
    # NUM_PREDICTED_MODES with the highest-confidence dropped ones.
    kept_indices.extend(dropped_indices[:contract.NUM_PREDICTED_MODES -
                                        len(kept_indices)])
    kept = torch.tensor(kept_indices, device=trajectories.device)
    return trajectories[kept], confidence_logits[kept]
