"""Moves points, directions and headings between frames.

A Frame is an origin and a heading measured in the frame the points are in
now. The three frames of the project are drawn at the end of this file.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Frame(NamedTuple):
    """A coordinate frame as seen from its parent frame."""

    origin: np.ndarray  # (2,) metres
    heading: float  # radians


def rotation_matrix(heading: float, dtype: type = np.float64) -> np.ndarray:
    """2D rotation matrix for the given heading angle, in radians."""
    cosine = np.cos(heading)
    sine = np.sin(heading)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=dtype)


def positions_to_frame(positions: np.ndarray, frame: Frame) -> np.ndarray:
    """Moves points into the frame whose x axis points along its heading."""
    positions = np.asarray(positions)
    centred = positions - np.asarray(frame.origin, dtype=positions.dtype)
    # A row vector p lands at (p - origin) R(heading), which is the column
    # form R(heading)^T (p - origin), a rotation by -heading.
    return centred @ rotation_matrix(frame.heading, positions.dtype)


def positions_from_frame(positions: np.ndarray, frame: Frame) -> np.ndarray:
    """Moves points out of a frame, back into the one its origin and heading
    were measured in, inverting positions_to_frame.
    """
    frame_positions = np.asarray(positions, dtype=np.float64)
    rotated = frame_positions @ rotation_matrix(frame.heading).T
    return rotated + np.asarray(frame.origin, dtype=np.float64)


def directions_to_frame(directions: np.ndarray, heading: float) -> np.ndarray:
    """Rotates direction vectors into a frame; a direction has no position, so
    no origin is subtracted.
    """
    directions = np.asarray(directions)
    return directions @ rotation_matrix(heading, directions.dtype)


def directions_from_frame(directions: np.ndarray, heading: float) -> np.ndarray:
    """Rotates direction vectors out of a frame, undoing directions_to_frame."""
    frame_directions = np.asarray(directions, dtype=np.float64)
    return frame_directions @ rotation_matrix(heading).T


def headings_to_frame(headings: np.ndarray, heading: float) -> np.ndarray:
    """Headings relative to the frame heading, wrapped to [-pi, pi)."""
    return wrap_to_pi(np.asarray(headings, dtype=np.float64) - heading)


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    # (a + pi) mod 2 pi - pi lands every angle in [-pi, pi).
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
#   Predictions travel back up, agent -> scene -> world (submit.py).
# ------------------------------------------------------------------
