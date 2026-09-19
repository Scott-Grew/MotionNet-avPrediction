# Sequence layout: history/future step counts, the current-step
# index within a scenario, and the physical timestep.
HISTORY_STEPS = 11
FUTURE_STEPS = 80
CURRENT_STEP_INDEX = 10
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS
TIMESTEP_SECONDS = 0.1
FUTURE_HORIZON_SECONDS = FUTURE_STEPS * TIMESTEP_SECONDS
MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED = 3.56

# Agent type vocabulary the model predicts, plus an "other"
# catch-all used only for staged agent features.
PREDICTED_OBJECT_TYPES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)
NUM_OBJECT_TYPES = len(PREDICTED_OBJECT_TYPES)
AGENT_TYPES = PREDICTED_OBJECT_TYPES + ("TYPE_OTHER",)
NUM_AGENT_TYPES = len(AGENT_TYPES)

# Staging and model sizing constants, and the version stamp
# checked against every staged artifact's provenance record.
STAGING_CROP_RADIUS_METRES = 400.0
MAP_POINT_SPACING_METRES = 1.0
MAP_CHUNK_DOTS = 20
NUM_PREDICTED_MODES = 6
STAGING_CODE_VERSION = "2026-08-19-a"

# Evenly spaced indices into the 80 future steps that get
# reported at submission time.
SUBMISSION_STEPS = 16
SUBMISSION_FIRST_SCENARIO_STEP = 15
SUBMISSION_STEP_STRIDE = FUTURE_STEPS // SUBMISSION_STEPS
SUBMISSION_FIRST_FUTURE_INDEX = SUBMISSION_FIRST_SCENARIO_STEP - (
    CURRENT_STEP_INDEX + 1
)
SUBMISSION_FUTURE_INDICES = tuple(
    range(
        SUBMISSION_FIRST_FUTURE_INDEX,
        FUTURE_STEPS,
        SUBMISSION_STEP_STRIDE,
    )
)
assert len(SUBMISSION_FUTURE_INDICES) == SUBMISSION_STEPS
assert SUBMISSION_FUTURE_INDICES[-1] == FUTURE_STEPS - 1

# Scale factors that bring model inputs into roughly unit range.
DISTANCE_NORMALISER_METRES = 65.8
VELOCITY_NORMALISER_METRES_PER_SECOND = 3.9
DIMENSION_NORMALISER_METRES = 1.6
SPEED_LIMIT_NORMALISER_MILES_PER_HOUR = 12.9

# Column layout of one agent feature row: position, heading as
# cosine/sine, velocity, box size, one-hot type, is-SDC flag.
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

# Map polyline kind vocabulary, one-hot per map dot row.
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

# Column layout of one map dot row: position, direction, one-hot
# kind. MAP_FEATURE_DIM grows below as more columns are appended.
MAP_POSITION = slice(0, 2)
MAP_DIRECTION = slice(2, 4)
MAP_KIND = slice(4, 4 + NUM_MAP_POLYLINE_KINDS)
MAP_FEATURE_DIM = MAP_KIND.stop
POLYLINE_SIGNAL_DIM = HISTORY_STEPS * NUM_TRAFFIC_SIGNAL_STATES

# Lane type vocabulary.
LANE_TYPES = (
    "TYPE_UNDEFINED",
    "TYPE_FREEWAY",
    "TYPE_SURFACE_STREET",
    "TYPE_BIKE_LANE",
)
NUM_LANE_TYPES = len(LANE_TYPES)

# Extends the map dot row with a one-hot lane type and a speed
# limit column.
MAP_LANE_TYPE = slice(MAP_KIND.stop, MAP_KIND.stop + NUM_LANE_TYPES)
MAP_SPEED_LIMIT = MAP_LANE_TYPE.stop
MAP_FEATURE_DIM = MAP_SPEED_LIMIT + 1

# Road line marking type vocabulary.
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

# Road edge type vocabulary, appended after road line types to
# form one combined boundary type vocabulary.
ROAD_EDGE_TYPES = (
    "TYPE_UNKNOWN",
    "TYPE_ROAD_EDGE_BOUNDARY",
    "TYPE_ROAD_EDGE_MEDIAN",
)
BOUNDARY_TYPES = ROAD_LINE_TYPES + ROAD_EDGE_TYPES
NUM_BOUNDARY_TYPES = len(BOUNDARY_TYPES)

# Extends the map dot row with a one-hot boundary type, a stop
# point, and left/right lane boundary crossing codes.
MAP_BOUNDARY_TYPE = slice(
    MAP_SPEED_LIMIT + 1, MAP_SPEED_LIMIT + 1 + NUM_BOUNDARY_TYPES
)
MAP_STOP_POINT = slice(
    MAP_BOUNDARY_TYPE.stop, MAP_BOUNDARY_TYPE.stop + 2
)
MAP_LEFT_BOUNDARY_CROSSING = MAP_STOP_POINT.stop
MAP_RIGHT_BOUNDARY_CROSSING = MAP_LEFT_BOUNDARY_CROSSING + 1
MAP_FEATURE_DIM = MAP_RIGHT_BOUNDARY_CROSSING + 1
NUM_BOUNDARY_CROSSING_CODES = len(ROAD_LINE_TYPES) + 1

# The two sides a lane boundary can be on.
LANE_SIDES = ("left", "right")


# Builds the JSON provenance stamp saved into every staged
# artifact, recording the code version that produced it.
def artifact_provenance(producer, source):
    import json

    return json.dumps(
        {
            "code_version": STAGING_CODE_VERSION,
            "producer": producer,
            "source": str(source),
        }
    )


# Raises if a loaded artifact has no provenance stamp, or was
# produced under a different code version than the working tree.
def check_artifact_provenance(
    stored_provenance, artifact_path, regenerate_hint
):
    import json

    assert stored_provenance is not None, (
        f"{artifact_path} carries no provenance stamp, so what produced it is unrecorded."
        f" {regenerate_hint}"
    )
    provenance = json.loads(str(stored_provenance))
    assert provenance["code_version"] == STAGING_CODE_VERSION, (
        f"{artifact_path} was produced under code_version {provenance['code_version']!r}"
        f" by {provenance['producer']} from {provenance['source']}, but the working tree is at"
        f" {STAGING_CODE_VERSION!r}. {regenerate_hint}"
    )
    return provenance
