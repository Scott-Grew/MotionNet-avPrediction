"""The network. Every agent and map chunk in a scene becomes a token and the
tokens self-attend once per scene; each predicted agent then reads them
through its own crop and 54 anchored futures are decoded.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from womd import contract
from womd.loader import MapArrays, SceneBatch
from womd.view import target_view

# Network size, chosen values. Tokens, queries and attention layers
# share one hidden width.
HIDDEN_DIM = 192
FEEDFORWARD_DIM = 4 * HIDDEN_DIM
ATTENTION_HEAD_COUNT = 4
SCENE_ATTENTION_ROUNDS = 6
DECODER_ROUNDS = 6

# The decoder predicts one future per anchor, 54 per object type;
# womd/pruning.py later keeps the 6 that Waymo scores.
QUERY_COUNT = 54
# The customary transformer initialisation scale, used for the
# learned queries.
QUERY_INITIALISATION_SCALE = 0.02
# Predicted log sigma is clamped to 0.2 m .. 148 m; without a
# floor the likelihood rewards ever-narrower Gaussians.
MINIMUM_LOG_STANDARD_DEVIATION = -1.609
MAXIMUM_LOG_STANDARD_DEVIATION = 5.0

# torch's own pre-norm transformer layers, with no dropout.
TRANSFORMER_LAYER_SETTINGS = {
    "d_model": HIDDEN_DIM,
    "nhead": ATTENTION_HEAD_COUNT,
    "dim_feedforward": FEEDFORWARD_DIM,
    "dropout": 0.0,
    "activation": "relu",
    "batch_first": True,
    "norm_first": True,
}


class ModePredictions(NamedTuple):
    """What the decoder returns per mode, a path and its per-step log sigma in
    metres, a confidence logit and the anchor it grew from.
    """

    trajectories: Tensor  # (batch, 54, 80, 2)
    log_standard_deviation: Tensor  # (batch, 54, 80, 2)
    confidence_logits: Tensor  # (batch, 54)
    anchors: Tensor  # (batch, 54, 2)


class Memory(NamedTuple):
    """Each target's view of its scene: its own history token first, then its
    scene's tokens, and a mask of which of those it may attend to.
    """

    tokens: Tensor  # (targets, 1 + agents + chunks, hidden)
    present: Tensor  # (targets, 1 + agents + chunks)


class Pooled(NamedTuple):
    """One token per map chunk slot, and whether any dot filled the slot."""

    tokens: Tensor  # (batch, chunks, hidden)
    present: Tensor  # (batch, chunks)


def agent_feature_divisors() -> Tensor:
    """Per-feature scale so position (metres), velocity (m/s) and dimensions
    (metres) are normalised before they enter the network.
    """
    divisors = torch.ones(contract.AGENT_FEATURE_DIM)
    divisors[contract.AGENT_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.AGENT_VELOCITY] = (
        contract.VELOCITY_NORMALISER_METRES_PER_SECOND
    )
    divisors[contract.AGENT_DIMENSIONS] = contract.DIMENSION_NORMALISER_METRES
    return divisors


def map_feature_divisors() -> Tensor:
    """Per-feature scale for map rows, position and stop point in metres and
    speed limit in miles per hour.
    """
    divisors = torch.ones(contract.MAP_FEATURE_DIM)
    divisors[contract.MAP_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.MAP_SPEED_LIMIT] = (
        contract.SPEED_LIMIT_NORMALISER_MILES_PER_HOUR
    )
    divisors[contract.MAP_STOP_POINT] = contract.DISTANCE_NORMALISER_METRES
    return divisors


def two_layer_mlp(input_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
    )


class AgentHistoryEncoder(nn.Module):
    """Encodes one agent's history steps into a single hidden-dim token. The
    MLP reads the whole window at once, each step's features plus one
    validity flag per step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_divisors", agent_feature_divisors())
        self.network = two_layer_mlp(
            contract.HISTORY_STEPS * (contract.AGENT_FEATURE_DIM + 1)
        )

    def forward(self, history: Tensor, mask: Tensor) -> Tensor:
        """History (..., steps, features) and mask (..., steps) in, one (...,
        hidden) token out; invalid steps are zeroed.
        """
        step_is_valid = mask.unsqueeze(-1).to(history.dtype)
        masked_history = history / self.feature_divisors * step_is_valid
        with_validity = torch.cat([masked_history, step_is_valid], dim=-1)
        return self.network(with_validity.flatten(start_dim=-2))


class MapDotEncoder(nn.Module):
    """Encodes one map dot (one metre of a polyline) into a hidden-dim token.
    The two crossing codes are category numbers, not quantities, so each is
    widened to a one-hot before the MLP sees it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_divisors", map_feature_divisors())
        self.register_buffer(
            "crossing_code_rows",
            torch.eye(contract.NUM_BOUNDARY_CROSSING_CODES),
        )
        plain_columns = contract.MAP_LEFT_BOUNDARY_CROSSING
        one_hot_columns = 2 * contract.NUM_BOUNDARY_CROSSING_CODES
        self.network = two_layer_mlp(plain_columns + one_hot_columns)

    def forward(self, map_rows: Tensor) -> Tensor:
        """Map rows (..., 32) in, one (..., hidden) token per dot out."""
        crossing = contract.MAP_LEFT_BOUNDARY_CROSSING
        scaled = map_rows / self.feature_divisors
        plain = scaled[..., :crossing]
        crossing_codes = map_rows[..., crossing:].long()
        one_hot = self.crossing_code_rows[crossing_codes].flatten(start_dim=-2)
        features = torch.cat([plain, one_hot.to(scaled.dtype)], dim=-1)
        return self.network(features)


def pool_chunks(dots: Tensor, scene_map: MapArrays) -> Pooled:
    """Max-pools each scene's dot tokens (S, D, hidden) into its chunk tokens
    (S, C, hidden); a chunk with no dots is absent.
    """
    scene_count, dot_count, hidden = dots.shape
    chunk_count = scene_map.chunk_pose.shape[1]
    # Padding dots and empty chunks start at -inf, so amax never selects
    # them, and a chunk no dot reaches stays at -inf.
    padding = ~scene_map.dot_present[..., None]
    real_dots = dots.masked_fill(padding, float("-inf")).flatten(0, 1)
    # Scene s, chunk c is row s * C + c once scenes and chunks are
    # flattened; a scatter along the first axis exports to ONNX.
    scenes = torch.arange(scene_count, device=dots.device)
    rows = scenes[:, None] * chunk_count + scene_map.dot_chunk
    selector = rows.flatten()[:, None].expand(-1, hidden)
    tokens = dots.new_full((scene_count * chunk_count, hidden), float("-inf"))
    tokens = tokens.scatter_reduce(0, selector, real_dots, reduce="amax")
    tokens = tokens.view(scene_count, chunk_count, hidden)
    present = tokens[..., 0] > float("-inf")
    tokens = torch.where(present[..., None], tokens, torch.zeros_like(tokens))
    return Pooled(tokens, present)


def initialise_attention(attention: nn.MultiheadAttention) -> None:
    """Re-draws torch's fused query-key-value projection from the uniform range
    three separate nn.Linear layers would start in.
    """
    bound = 1.0 / math.sqrt(HIDDEN_DIM)
    nn.init.uniform_(attention.in_proj_weight, -bound, bound)
    nn.init.uniform_(attention.in_proj_bias, -bound, bound)
    nn.init.uniform_(attention.out_proj.bias, -bound, bound)


def scene_attention_layer() -> nn.TransformerEncoderLayer:
    layer = nn.TransformerEncoderLayer(**TRANSFORMER_LAYER_SETTINGS)
    initialise_attention(layer.self_attn)
    return layer


def decoder_round_layer() -> nn.TransformerDecoderLayer:
    """One decoder round of self-attention over the modes, cross-attention to
    the scene tokens and feedforward.
    """
    layer = nn.TransformerDecoderLayer(**TRANSFORMER_LAYER_SETTINGS)
    initialise_attention(layer.self_attn)
    initialise_attention(layer.multihead_attn)
    return layer


class SceneEncoder(nn.Module):
    """Encodes each scene once, in the scene frame, then gives every predicted
    agent its own view of those tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.agent_encoder = AgentHistoryEncoder()
        self.scene_agent_encoder = AgentHistoryEncoder()
        self.map_encoder = MapDotEncoder()
        self.signal_projection = nn.Linear(
            contract.POLYLINE_SIGNAL_DIM, HIDDEN_DIM, bias=False
        )
        # A token pose is (x, y, cosine, sine), and only x and y are metres
        # that need scaling.
        self.pose_projection = nn.Linear(4, HIDDEN_DIM)
        metres = contract.DISTANCE_NORMALISER_METRES
        self.register_buffer(
            "pose_divisors", torch.tensor([metres, metres, 1.0, 1.0])
        )
        self.layers = nn.ModuleList(
            scene_attention_layer() for _ in range(SCENE_ATTENTION_ROUNDS)
        )

    def scene_tokens(self, batch: SceneBatch) -> tuple[Tensor, Tensor]:
        """Tokens (scenes, agents + chunks, hidden), in that order, after
        self-attention, and a mask of which tokens are real.
        """
        scene = batch.scene
        scene_map = batch.map
        agent_motion = self.scene_agent_encoder(
            scene.agent_history, scene.agent_history_mask
        )
        agent_signals = scene.agent_signal_history.flatten(start_dim=-2)
        agent_tokens = agent_motion + self.signal_projection(agent_signals)

        dot_embeddings = self.map_encoder(scene_map.rows)
        pooled = pool_chunks(dot_embeddings, scene_map)
        chunk_signals = scene_map.chunk_signal_history.flatten(start_dim=-2)
        map_tokens = pooled.tokens + self.signal_projection(chunk_signals)

        tokens = torch.cat([agent_tokens, map_tokens], dim=1)
        agent_present = scene.agent_history_mask.any(dim=-1)
        present = torch.cat([agent_present, pooled.present], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=~present)
        return tokens, present

    def forward(self, batch: SceneBatch) -> Memory:
        """Each target's memory: its own history token, then its scene's tokens
        with each token's pose in the target's frame added, masked to its
        crop.
        """
        tokens, present = self.scene_tokens(batch)
        targets = batch.targets
        own_motion = self.agent_encoder(
            targets.agent_history, targets.agent_history_mask
        )
        own_signals = targets.agent_signal_history.flatten(start_dim=-2)
        own_token = (
            own_motion + self.signal_projection(own_signals)
        ).unsqueeze(1)
        own_present = torch.ones(
            own_token.shape[:2], dtype=torch.bool, device=own_token.device
        )

        # A scene with several targets is repeated once per target, each copy
        # seen from that target's frame and masked to its crop.
        view = target_view(batch)
        pose_embedding = self.pose_projection(view.pose / self.pose_divisors)
        scene_tokens = tokens.index_select(0, targets.scene_index)
        target_tokens = scene_tokens + pose_embedding
        scene_present = present.index_select(0, targets.scene_index)
        view_present = scene_present & view.visible
        memory_tokens = torch.cat([own_token, target_tokens], dim=1)
        memory_present = torch.cat([own_present, view_present], dim=1)
        return Memory(memory_tokens, memory_present)


def predicted_type_index(history: Tensor) -> Tensor:
    """Each agent's predicted type index, from the one-hot object type at the
    current history step.
    """
    type_one_hot = history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE]
    return type_one_hot[:, : contract.NUM_OBJECT_TYPES].argmax(dim=-1)


def agent_reachable_distance(history: Tensor) -> Tensor:
    """Upper bound on how far an agent could travel over the future horizon at
    constant maximum acceleration from its current speed.
    """
    speed = history[
        :, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY
    ].norm(dim=-1)
    horizon = contract.FUTURE_HORIZON_SECONDS
    acceleration = contract.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED
    # d = v T + a T^2 / 2 for speed v, horizon T and acceleration a.
    return speed * horizon + 0.5 * acceleration * horizon**2


def anchor_lines(anchors: Tensor) -> Tensor:
    """Each anchor endpoint (..., 2) as a constant-speed straight line from
    the agent, (..., 80, 2): future step k sits at k / 80 of the endpoint.
    The decoder adds its offsets to these lines and the loss assigns each
    logged future to the nearest one, so both must use this one rule.
    """
    steps = torch.arange(1, contract.FUTURE_STEPS + 1, device=anchors.device)
    ramp = steps.to(anchors.dtype) / contract.FUTURE_STEPS
    return anchors[..., None, :] * ramp[:, None]


class ModeDecoder(nn.Module):
    """Decodes the 54 anchor-conditioned trajectory modes and their confidences
    from the memory built by SceneEncoder. Each query is a learned vector plus
    an embedding of its anchor endpoint, so each query starts tied to one
    anchor.
    """

    def __init__(self, unit_anchors: Tensor) -> None:
        super().__init__()
        expected_shape = (contract.NUM_OBJECT_TYPES, QUERY_COUNT, 2)
        if unit_anchors.shape != expected_shape:
            raise ValueError(
                f"unit_anchors has shape {tuple(unit_anchors.shape)},"
                f" not {expected_shape}"
            )
        self.queries = nn.Parameter(
            torch.randn(QUERY_COUNT, HIDDEN_DIM) * QUERY_INITIALISATION_SCALE
        )
        # Anchor endpoints in metres, despite the name, which saved
        # checkpoints and anchor files depend on.
        self.register_buffer(
            "unit_anchors", unit_anchors.detach().clone().float()
        )
        self.anchor_projection = nn.Linear(2, HIDDEN_DIM)
        self.scene_norm = nn.LayerNorm(HIDDEN_DIM)
        self.rounds = nn.ModuleList(
            decoder_round_layer() for _ in range(DECODER_ROUNDS)
        )
        self.head_norm = nn.LayerNorm(HIDDEN_DIM)
        # Per future step, an x, y offset and an x, y log sigma.
        self.trajectory_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, 2 * contract.FUTURE_STEPS * 2),
        )
        # There is no final bias because one bias shared by all modes
        # cancels in the softmax over modes and could never be trained.
        self.confidence_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, 1, bias=False),
        )

    def forward(self, memory: Memory, type_index: Tensor) -> ModePredictions:
        """Trajectories and log sigmas (batch, modes, steps, 2) in metres,
        confidence logits (batch, modes), and the anchors used.
        """
        batch_size = memory.tokens.shape[0]
        anchors = self.unit_anchors[type_index]
        queries = self.queries + self.anchor_projection(
            anchors / contract.DISTANCE_NORMALISER_METRES
        )
        normed_tokens = self.scene_norm(memory.tokens)
        for decoder_round in self.rounds:
            queries = decoder_round(
                queries, normed_tokens, memory_key_padding_mask=~memory.present
            )
        queries = self.head_norm(queries)

        # Cast up so the cumulative sum below runs in float32 even under
        # mixed precision.
        head_shape = (batch_size, QUERY_COUNT, 2, contract.FUTURE_STEPS, 2)
        head_output = self.trajectory_head(queries).float().view(head_shape)
        step_displacements, log_sigma = head_output.unbind(dim=2)
        # A mode is its anchor's straight constant-speed line plus the running
        # sum of the predicted per-step offsets.
        lines = anchor_lines(anchors)
        trajectories = step_displacements.cumsum(dim=-2) + lines
        log_sigma = log_sigma.clamp(
            MINIMUM_LOG_STANDARD_DEVIATION, MAXIMUM_LOG_STANDARD_DEVIATION
        )
        confidence_logits = self.confidence_head(queries).squeeze(-1)
        return ModePredictions(
            trajectories, log_sigma, confidence_logits, anchors
        )


class MotionPredictor(nn.Module):
    """Top-level model, the scene encoder's memory feeding the mode decoder."""

    def __init__(self, unit_anchors: Tensor) -> None:
        super().__init__()
        self.scene_encoder = SceneEncoder()
        self.mode_decoder = ModeDecoder(unit_anchors)

    @property
    def unit_anchors(self) -> Tensor:
        return self.mode_decoder.unit_anchors

    def forward(self, batch: SceneBatch) -> ModePredictions:
        """The scene encoder and mode decoder for one batch, with anchors
        chosen by each agent's predicted type.
        """
        memory = self.scene_encoder(batch)
        type_index = predicted_type_index(batch.targets.agent_history)
        return self.mode_decoder(memory, type_index)


# ------------------------------------------------------------------
# DATA FLOW for S scenes holding B predicted agents, hidden width 192
#
#   once per scene, in the scene frame.
#
#   scene_agent_history  --MLP-->      A tokens  \
#                                                 >  (S, A+C, 192)
#   map_rows  --MLP, max-pool-->       C tokens  /
#   (each token also gets its traffic-light history)
#                          |
#            6 x pre-norm self-attention layer
#                          |
#                     scene tokens
#                          |
#   once per predicted agent.
#
#   its scene's tokens, each + an embedding of that token's pose
#   in the agent's frame, masked to the agent's crop
#   agent_history  --MLP-->  1 own token, placed first
#                          |
#                  (B, 1+A+C, 192)
#                          |
#   54 queries, each a learned vector + its anchor's embedding
#                          |
#            6 x [ self-attention over the 54 modes,
#                  cross-attention to the agent's tokens,
#                  feedforward ]
#                          |
#            +-------------+--------------+
#            |                            |
#     trajectory head              confidence head
#     offsets and log sigma,       logits (B, 54)
#     (B, 54, 80, 2) each
#            |
#     path = the anchor's straight line + running sum of offsets
# ------------------------------------------------------------------
