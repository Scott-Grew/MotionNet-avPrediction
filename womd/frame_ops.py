"""Moves points, directions and headings between frames.

Each function takes a frame's origin and heading as measured in the frame the
points are in now. The three frames are drawn at the end of this file.
"""
from __future__ import annotations

import numpy as np


def rotation_matrix(heading: float,
                    dtype: np.dtype | type = np.float64) -> np.ndarray:
    """2D rotation matrix for the given heading angle, in radians."""
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=dtype)


def positions_to_frame(positions: np.ndarray, frame_origin: np.ndarray,
                       frame_heading: float) -> np.ndarray:
    """Moves points into the frame at frame_origin whose x axis points along
    frame_heading.
    """
    positions = np.asarray(positions)
    centred = positions - np.asarray(frame_origin, dtype=positions.dtype)
    return centred @ rotation_matrix(frame_heading, positions.dtype)


def positions_from_frame(frame_positions: np.ndarray, frame_origin: np.ndarray,
                         frame_heading: float) -> np.ndarray:
    """Inverse of positions_to_frame: moves points out of a frame, back into the
    one its origin and heading were measured in.
    """
    rotated = (np.asarray(frame_positions, dtype=np.float64)
               @ rotation_matrix(frame_heading).T)
    return rotated + np.asarray(frame_origin, dtype=np.float64)


def directions_to_frame(directions: np.ndarray,
                        frame_heading: float) -> np.ndarray:
    """Rotates direction vectors into a frame; a direction has no position, so
    no origin is subtracted.
    """
    directions = np.asarray(directions)
    return directions @ rotation_matrix(frame_heading, directions.dtype)


def headings_to_frame(headings: np.ndarray, frame_heading: float) -> np.ndarray:
    """Expresses headings relative to the frame heading, wrapped into [-pi, pi].
    """
    return wrap_to_pi(np.asarray(headings, dtype=np.float64) - frame_heading)


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wraps angles in radians into [-pi, pi]."""
    full_turn = 2.0 * np.pi
    shifted = np.asarray(angles, dtype=np.float64) + np.pi
    return shifted % full_turn - np.pi


# ------------------------------------------------------------------
# THE THREE FRAMES
#
#   world            Waymo's own coordinates
#     |  origin and heading of the self-driving car, current step
#     v
#   scene            staged files are stored here (womd/store.py)
#     |  origin and heading of one predicted agent, current step
#     v
#   agent            the model reads and predicts here
#
#   Predictions travel back up: agent -> scene -> world (submit.py).
# ------------------------------------------------------------------
