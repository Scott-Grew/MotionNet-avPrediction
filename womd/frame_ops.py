import numpy as np


# 2D rotation matrix for the given heading angle, in radians.
def rotation_matrix(heading, dtype=np.float64):
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=dtype)


# Rotates world-frame points into a frame centred on the agent
# with x pointing along its heading.
def positions_to_agent_frame(
    world_positions, frame_origin, frame_heading
):
    positions = np.asarray(world_positions)
    centred = positions - np.asarray(
        frame_origin, dtype=positions.dtype
    )
    return centred @ rotation_matrix(frame_heading, positions.dtype)


# Inverse of positions_to_agent_frame: maps agent-frame points
# back into the world frame.
def positions_to_world_frame(
    agent_positions, frame_origin, frame_heading
):
    rotated = (
        np.asarray(agent_positions, dtype=np.float64)
        @ rotation_matrix(frame_heading).T
    )
    return rotated + np.asarray(frame_origin, dtype=np.float64)


# Rotates direction vectors (no translation) into the agent
# frame.
def directions_to_agent_frame(world_directions, frame_heading):
    directions = np.asarray(world_directions)
    return directions @ rotation_matrix(
        frame_heading, directions.dtype
    )


# Expresses world headings relative to the frame heading,
# wrapped into [-pi, pi].
def headings_to_agent_frame(world_headings, frame_heading):
    return wrap_to_pi(
        np.asarray(world_headings, dtype=np.float64) - frame_heading
    )


# Wraps angles in radians into [-pi, pi].
def wrap_to_pi(angles):
    return (np.asarray(angles, dtype=np.float64) + np.pi) % (
        2.0 * np.pi
    ) - np.pi
