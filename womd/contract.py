"""The fixed layout every step agrees on: sequence lengths, the columns of an
agent row and a map row, the two file formats passed between steps, and the
stamp that ties each file to the code version that wrote it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

# A scenario is 91 steps at 10 Hz, 11 of history ending at the
# current step and then the 80 future steps to predict.
HISTORY_STEPS = 11
FUTURE_STEPS = 80
CURRENT_STEP_INDEX = 10
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS
TIMESTEP_SECONDS = 0.1
FUTURE_HORIZON_SECONDS = FUTURE_STEPS * TIMESTEP_SECONDS
# Bounds how far an agent can travel inside the horizon, which
# sets the scale its anchors are fitted at.
MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED = 3.56

# Agent type vocabulary the model predicts, plus an "other"
# catch-all used only for staged agent features.
PREDICTED_OBJECT_TYPES = ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST")
NUM_OBJECT_TYPES = len(PREDICTED_OBJECT_TYPES)
AGENT_TYPES = PREDICTED_OBJECT_TYPES + ("TYPE_OTHER",)
NUM_AGENT_TYPES = len(AGENT_TYPES)

# Staging and model sizing constants, and the version stamp
# checked against each staged artifact's provenance record.
STAGING_CROP_RADIUS_METRES = 400.0
MAP_POINT_SPACING_METRES = 1.0
MAP_CHUNK_DOTS = 20
NUM_PREDICTED_MODES = 6
STAGING_CODE_VERSION = "2026-08-19-a"

# Waymo scores 16 of the 80 future steps, every fifth one, the
# first being scenario step 15, future index 4.
SUBMISSION_STEPS = 16
SUBMISSION_STEP_STRIDE = FUTURE_STEPS // SUBMISSION_STEPS
SUBMISSION_FIRST_FUTURE_INDEX = 15 - (CURRENT_STEP_INDEX + 1)
SUBMISSION_FUTURE_INDICES = tuple(
    range(SUBMISSION_FIRST_FUTURE_INDEX, FUTURE_STEPS, SUBMISSION_STEP_STRIDE)
)

# Input scales. Each is the standard deviation of that quantity
# as the model reads it, measured on every 60th training scenario.
# All distances share the map-position scale, so geometry keeps
# its proportions.
DISTANCE_NORMALISER_METRES = 65.8
VELOCITY_NORMALISER_METRES_PER_SECOND = 3.9
DIMENSION_NORMALISER_METRES = 1.6
# Measured over lane dots that carry a posted limit above zero.
SPEED_LIMIT_NORMALISER_MILES_PER_HOUR = 12.9

# Column layout of one agent feature row, in order. Position, heading
# as cosine and sine, velocity, box size, one-hot type, is-SDC flag.
AGENT_POSITION = slice(0, 2)
AGENT_HEADING_COSINE = 2
AGENT_HEADING_SINE = 3
AGENT_VELOCITY = slice(4, 6)
AGENT_DIMENSIONS = slice(6, 8)
AGENT_TYPE = slice(8, 8 + NUM_AGENT_TYPES)
AGENT_IS_SDC = AGENT_TYPE.stop
AGENT_FEATURE_DIM = AGENT_IS_SDC + 1

# Traffic signal state vocabulary, one-hot per history step in a
# lane's signal history.
TRAFFIC_SIGNAL_STATES = (
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_ARROW_STOP",
    "LANE_STATE_ARROW_CAUTION",
    "LANE_STATE_ARROW_GO",
    "LANE_STATE_STOP",
    "LANE_STATE_CAUTION",
    "LANE_STATE_GO",
    "LANE_STATE_FLASHING_STOP",
    "LANE_STATE_FLASHING_CAUTION",
)
NUM_TRAFFIC_SIGNAL_STATES = len(TRAFFIC_SIGNAL_STATES)
SIGNAL_HISTORY_SHAPE = (HISTORY_STEPS, NUM_TRAFFIC_SIGNAL_STATES)

# Vocabularies for the one-hot blocks of a map dot row, the kind of
# feature a dot belongs to and then the lane, line and edge types.
MAP_POLYLINE_KINDS = (
    "lane",
    "road_line",
    "road_edge",
    "stop_sign",
    "crosswalk",
    "speed_bump",
    "driveway",
)
NUM_MAP_POLYLINE_KINDS = len(MAP_POLYLINE_KINDS)

LANE_TYPES = (
    "TYPE_UNDEFINED",
    "TYPE_FREEWAY",
    "TYPE_SURFACE_STREET",
    "TYPE_BIKE_LANE",
)
NUM_LANE_TYPES = len(LANE_TYPES)

ROAD_LINE_TYPES = (
    "TYPE_UNKNOWN",
    "TYPE_BROKEN_SINGLE_WHITE",
    "TYPE_SOLID_SINGLE_WHITE",
    "TYPE_SOLID_DOUBLE_WHITE",
    "TYPE_BROKEN_SINGLE_YELLOW",
    "TYPE_BROKEN_DOUBLE_YELLOW",
    "TYPE_SOLID_SINGLE_YELLOW",
    "TYPE_SOLID_DOUBLE_YELLOW",
    "TYPE_PASSING_DOUBLE_YELLOW",
)

ROAD_EDGE_TYPES = (
    "TYPE_UNKNOWN",
    "TYPE_ROAD_EDGE_BOUNDARY",
    "TYPE_ROAD_EDGE_MEDIAN",
)
# Road lines and road edges share one boundary-type one-hot,
# lines first.
BOUNDARY_TYPES = ROAD_LINE_TYPES + ROAD_EDGE_TYPES
NUM_BOUNDARY_TYPES = len(BOUNDARY_TYPES)

# Column layout of one map dot row, 32 columns, in order. Columns
# that do not apply to a dot's kind stay zero.
MAP_POSITION = slice(0, 2)
MAP_DIRECTION = slice(2, 4)
MAP_KIND = slice(4, 4 + NUM_MAP_POLYLINE_KINDS)
MAP_LANE_TYPE = slice(MAP_KIND.stop, MAP_KIND.stop + NUM_LANE_TYPES)
MAP_SPEED_LIMIT = MAP_LANE_TYPE.stop
MAP_BOUNDARY_TYPE = slice(
    MAP_SPEED_LIMIT + 1, MAP_SPEED_LIMIT + 1 + NUM_BOUNDARY_TYPES
)
MAP_STOP_POINT = slice(MAP_BOUNDARY_TYPE.stop, MAP_BOUNDARY_TYPE.stop + 2)
# The two crossing codes stay last because the map encoder slices
# them off the end of the row and one-hot encodes them.
MAP_LEFT_BOUNDARY_CROSSING = MAP_STOP_POINT.stop
MAP_RIGHT_BOUNDARY_CROSSING = MAP_LEFT_BOUNDARY_CROSSING + 1
MAP_FEATURE_DIM = MAP_RIGHT_BOUNDARY_CROSSING + 1
# A crossing code is 0 for no boundary, else 1 + the road line
# type of the boundary on that side of the lane.
NUM_BOUNDARY_CROSSING_CODES = len(ROAD_LINE_TYPES) + 1

# One signal history per polyline, flattened to a one-hot signal
# state for each history step.
POLYLINE_SIGNAL_DIM = HISTORY_STEPS * NUM_TRAFFIC_SIGNAL_STATES

# The two sides a lane boundary can be on.
LANE_SIDES = ("left", "right")


def artifact_provenance(producer: str, source: Path | str) -> str:
    """The JSON stamp saved into each artifact, naming the code version, the
    script that wrote it and what it was made from.
    """
    stamp = {
        "code_version": STAGING_CODE_VERSION,
        "producer": producer,
        "source": str(source),
    }
    return json.dumps(stamp)


def check_provenance(stamp: Any, path: Path | str, hint: str) -> dict[str, str]:
    """Raises if an artifact has no stamp or was written by another code
    version; hint says how to regenerate it.
    """
    if stamp is None:
        raise ValueError(f"{path} has no provenance stamp. {hint}")
    provenance = json.loads(str(stamp))
    if provenance["code_version"] != STAGING_CODE_VERSION:
        raise ValueError(
            f"{path} is from code version {provenance['code_version']!r},"
            f" this tree is {STAGING_CODE_VERSION!r}. {hint}"
        )
    return provenance


class StagedFile(NamedTuple):
    """One staged scenario as stage.py writes it, every array in the scene
    frame. T tracks, F map features, D map dots, 91 steps. Files staged by
    older code, data/staged among them, carry five extra lane-topology arrays
    and no provenance stamp; load reads only these fields and checks the stamp
    when present.
    """

    track_rows: np.ndarray  # (T, 91, 13) float32, agent rows
    track_valid: np.ndarray  # (T, 91) bool, step observed
    track_ids: np.ndarray  # (T,) int64
    is_designated_target: np.ndarray  # (T,) bool, Waymo scores it
    is_object_of_interest: np.ndarray  # (T,) bool
    map_rows: np.ndarray  # (D, 32) float32, dots grouped by feature
    feature_lengths: np.ndarray  # (F,) int64, dots per feature
    feature_ids: np.ndarray  # (F,) int64
    feature_is_interpolating: np.ndarray  # (F,) bool
    polyline_signal_histories: np.ndarray  # (F, 11, 9) float32
    frame_origin: np.ndarray  # (2,) float32, the scene frame in the world
    frame_heading: np.ndarray  # () float32
    scenario_id: np.ndarray  # () str
    provenance: np.ndarray | None = None  # () str

    def save(self, path: Path) -> None:
        """Writes to a .partial path and renames it, so a killed process
        never leaves a half-written file in place.
        """
        partial_path = path.with_suffix(path.suffix + ".partial")
        with open(partial_path, "wb") as partial_file:
            np.savez_compressed(partial_file, **self._asdict())
        partial_path.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> StagedFile:
        with np.load(path) as staged:
            stored = set(staged.files)
            named = [name for name in cls._fields if name in stored]
            arrays = {name: staged[name] for name in named}
        if "provenance" in arrays:
            hint = "Restage the directory with stage.py."
            check_provenance(arrays["provenance"], path, hint)
        dot_count = int(arrays["feature_lengths"].sum())
        row_count = len(arrays["map_rows"])
        if dot_count != row_count:
            raise ValueError(
                f"{path}: feature_lengths sum to {dot_count}"
                f" over {row_count} map rows"
            )
        return cls(**arrays)


class Submission(NamedTuple):
    """Six world-frame futures per designated target at 2 Hz, one row per
    target, as submit.py writes them and the container scores them.
    """

    scenario_id: np.ndarray  # (targets,) str
    track_id: np.ndarray  # (targets,) int64
    world_trajectories: np.ndarray  # (targets, 6, 16, 2) float64 metres
    confidences: np.ndarray  # (targets, 6) float64, summing to 1
    provenance: np.ndarray  # () str

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self._asdict())

    @classmethod
    def load(cls, path: Path | str) -> Submission:
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        hint = "Regenerate the predictions with submit.py."
        check_provenance(arrays.get("provenance"), path, hint)
        return cls(**arrays)
