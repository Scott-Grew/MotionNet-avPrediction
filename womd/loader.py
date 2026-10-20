"""Reads staged scenarios and picks the agents to predict. The scorer
container imports this file under Python 3.8. womd/inputs.py builds each
agent's model inputs from what it reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy import ndarray

from womd import contract, frame_ops
from womd.contract import StagedFile
from womd.frame_ops import Frame

HISTORY = slice(0, contract.HISTORY_STEPS)
FUTURE = slice(contract.CURRENT_STEP_INDEX + 1, None)
HEADING = slice(contract.AGENT_HEADING_COSINE, contract.AGENT_HEADING_SINE + 1)


class StagedScenario(NamedTuple):
    """One staged scenario in the scene frame, with T tracks, F map features
    and D map dots. The last field is derived on load.
    """

    scenario_id: str
    scene_frame: Frame  # the scene frame in the world
    track_ids: ndarray  # (T,)
    track_rows: ndarray  # (T, 91, 13)
    track_valid: ndarray  # (T, 91)
    is_designated_target: ndarray  # (T,)
    map_rows: ndarray  # (D, 32)
    feature_lengths: ndarray  # (F,) dots per feature
    polyline_signal_histories: ndarray  # (F, 11, 9)
    map_dot_polyline_index: ndarray  # (D,) feature each dot belongs to


def eligible_tracks(scenario: StagedScenario, designated_only: bool) -> ndarray:
    """Indices of tracks valid at the current step and of a predicted object
    type, optionally restricted to designated targets.
    """
    now_rows = scenario.track_rows[:, contract.CURRENT_STEP_INDEX]
    # The predicted types are the first NUM_OBJECT_TYPES columns of the type
    # one-hot; the column after them is TYPE_OTHER.
    type_one_hot = now_rows[:, contract.AGENT_TYPE]
    predicted_columns = type_one_hot[:, : contract.NUM_OBJECT_TYPES]
    is_predicted_type = predicted_columns.any(axis=1)
    now_valid = scenario.track_valid[:, contract.CURRENT_STEP_INDEX]
    is_eligible = now_valid & is_predicted_type
    if designated_only:
        is_eligible = is_eligible & scenario.is_designated_target
    return np.flatnonzero(is_eligible)


def agent_frame(rows: ndarray) -> Frame:
    """A track's own frame at the current step, from its rows (91, 13) in the
    scene frame.
    """
    now = rows[contract.CURRENT_STEP_INDEX]
    sine = now[contract.AGENT_HEADING_SINE]
    cosine = now[contract.AGENT_HEADING_COSINE]
    return Frame(now[contract.AGENT_POSITION], np.arctan2(sine, cosine))


def rows_to_frame(rows: ndarray, frame: Frame) -> ndarray:
    """Re-expresses agent rows' position, velocity and heading columns in
    another agent's frame.
    """
    moved = rows.copy()
    moved[..., contract.AGENT_POSITION] = frame_ops.positions_to_frame(
        rows[..., contract.AGENT_POSITION], frame
    )
    moved[..., contract.AGENT_VELOCITY] = frame_ops.directions_to_frame(
        rows[..., contract.AGENT_VELOCITY], frame.heading
    )
    # A heading's (cos, sin) is a unit direction, so it rotates like one.
    moved[..., HEADING] = frame_ops.directions_to_frame(
        rows[..., HEADING], frame.heading
    )
    return moved


def staged_scenario(staged: StagedFile) -> StagedScenario:
    """The staged arrays the model reads, with the feature each map dot
    belongs to derived from the feature lengths.
    """
    feature_lengths = staged.feature_lengths
    dot_polyline = np.repeat(np.arange(len(feature_lengths)), feature_lengths)
    scene_frame = Frame(staged.frame_origin, float(staged.frame_heading))
    return StagedScenario(
        scenario_id=str(staged.scenario_id),
        scene_frame=scene_frame,
        track_ids=staged.track_ids,
        track_rows=staged.track_rows,
        track_valid=staged.track_valid,
        is_designated_target=staged.is_designated_target,
        map_rows=staged.map_rows,
        feature_lengths=feature_lengths,
        polyline_signal_histories=staged.polyline_signal_histories,
        map_dot_polyline_index=dot_polyline,
    )


def read_scenario(path: Path | str) -> StagedScenario:
    return staged_scenario(StagedFile.load(path))


def designated_tracks(scenario: StagedScenario) -> ndarray:
    """The tracks Waymo designated for scoring; raises if any of them cannot
    be predicted.
    """
    designated_count = int(scenario.is_designated_target.sum())
    track_indices = eligible_tracks(scenario, designated_only=True)
    if len(track_indices) != designated_count:
        raise ValueError(
            f"{scenario.scenario_id} designates {designated_count} targets,"
            f" {len(track_indices)} can be predicted"
        )
    return track_indices
