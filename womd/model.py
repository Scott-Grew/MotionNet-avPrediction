"""The network. Every agent and map chunk in a scene becomes a token and the
tokens self-attend once per scene; each predicted agent then reads them
through its own crop and 54 anchored futures are decoded.
"""
from __future__ import annotations

from collections import namedtuple
import math

import torch
from torch import nn

from womd import contract

# What the decoder returns per mode, a path and its per-step log sigma
# in metres, a confidence logit and the anchor it grew from.
ModePredictions = namedtuple(
    "ModePredictions",
    "trajectories log_standard_deviation confidence_logits anchors",
)

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


def agent_feature_divisors() -> torch.Tensor:
    """Per-feature scale so position (metres), velocity (m/s) and dimensions
    (metres) are normalised before they enter the network.
    """
    divisors = torch.ones(contract.AGENT_FEATURE_DIM)
    divisors[contract.AGENT_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.AGENT_VELOCITY] = (
        contract.VELOCITY_NORMALISER_METRES_PER_SECOND)
    divisors[contract.AGENT_DIMENSIONS] = contract.DIMENSION_NORMALISER_METRES
    return divisors


def map_feature_divisors() -> torch.Tensor:
    """Per-feature scale for map rows, position and stop point in metres and
    speed limit in miles per hour.
    """
    divisors = torch.ones(contract.MAP_FEATURE_DIM)
    divisors[contract.MAP_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.MAP_SPEED_LIMIT] = (
        contract.SPEED_LIMIT_NORMALISER_MILES_PER_HOUR)
    divisors[contract.MAP_STOP_POINT] = contract.DISTANCE_NORMALISER_METRES
    return divisors


class AgentHistoryEncoder(nn.Module):
    """Encodes one agent's history steps into a single hidden-dim token."""

    def __init__(self) -> None:
        """The MLP reads the whole history window at once, each step's features
        plus one validity flag per step.
        """
        super().__init__()
        input_width = contract.HISTORY_STEPS * (contract.AGENT_FEATURE_DIM + 1)
        self.register_buffer("feature_divisors", agent_feature_divisors())
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, agent_history: torch.Tensor,
                agent_history_mask: torch.Tensor) -> torch.Tensor:
        """History (..., steps, features) and mask (..., steps) in, one (...,
        hidden) token out; invalid steps are zeroed.
        """
        step_is_valid = agent_history_mask.unsqueeze(-1).to(agent_history.dtype)
        masked_history = agent_history / self.feature_divisors * step_is_valid
        history_with_validity = torch.cat([masked_history, step_is_valid],
                                          dim=-1).flatten(start_dim=-2)
        return self.network(history_with_validity)


class MapDotEncoder(nn.Module):
    """Encodes one map dot (one metre of a polyline) into a hidden-dim token;
    boundary crossing codes go in one-hot.
    """

    def __init__(self) -> None:
        """The two crossing codes are category numbers, not quantities, so each
        is widened to a one-hot before the MLP sees it.
        """
        super().__init__()
        self.register_buffer("feature_divisors", map_feature_divisors())
        self.register_buffer(
            "crossing_code_rows",
            torch.eye(contract.NUM_BOUNDARY_CROSSING_CODES),
        )
        plain_column_count = contract.MAP_LEFT_BOUNDARY_CROSSING
        one_hot_column_count = 2 * contract.NUM_BOUNDARY_CROSSING_CODES
        input_width = plain_column_count + one_hot_column_count
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, map_rows: torch.Tensor) -> torch.Tensor:
        """Map rows (dots, features) in, (dots, hidden) out.

        The dots of all samples in the batch share the one leading axis.
        """
        scaled_rows = map_rows / self.feature_divisors
        crossing_codes = map_rows[:,
                                  contract.MAP_LEFT_BOUNDARY_CROSSING:].long()
        crossing_one_hot = self.crossing_code_rows[crossing_codes].flatten(
            start_dim=-2)
        return self.network(
            torch.cat(
                [
                    scaled_rows[:, :contract.MAP_LEFT_BOUNDARY_CROSSING],
                    crossing_one_hot.to(scaled_rows.dtype),
                ],
                dim=-1,
            ))


def pool_dots_to_chunk_tokens(
        dot_embeddings: torch.Tensor, dot_chunk_slot: torch.Tensor,
        batch_size: int, max_chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Max-pools dot embeddings (dots, hidden) into chunk tokens (batch, chunks,
    hidden); a chunk slot with no dots is absent.
    """
    hidden_width = dot_embeddings.shape[-1]
    # Slots start at -inf so amax pooling never selects an empty
    # slot; a slot untouched by any dot stays at -inf.
    tokens = torch.full(
        (batch_size * max_chunks, hidden_width),
        float("-inf"),
        dtype=dot_embeddings.dtype,
        device=dot_embeddings.device,
    )
    tokens = tokens.scatter_reduce(
        0,
        dot_chunk_slot.unsqueeze(-1).expand(-1, hidden_width),
        dot_embeddings,
        reduce="amax",
        include_self=True,
    )
    chunk_present = tokens[:, 0] > float("-inf")
    tokens = torch.where(
        chunk_present.unsqueeze(-1),
        tokens,
        torch.zeros_like(tokens),
    )
    return (
        tokens.view(batch_size, max_chunks, hidden_width),
        chunk_present.view(batch_size, max_chunks),
    )


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
        self.signal_projection = nn.Linear(contract.POLYLINE_SIGNAL_DIM,
                                           HIDDEN_DIM,
                                           bias=False)
        # A token pose is (x, y, cosine, sine), and only x and y are
        # metres that need scaling.
        self.pose_projection = nn.Linear(4, HIDDEN_DIM)
        self.register_buffer(
            "pose_divisors",
            torch.tensor([
                contract.DISTANCE_NORMALISER_METRES,
                contract.DISTANCE_NORMALISER_METRES,
                1.0,
                1.0,
            ]),
        )
        self.layers = nn.ModuleList(
            scene_attention_layer() for _ in range(SCENE_ATTENTION_ROUNDS))

    def scene_tokens(
            self,
            batch: dict[str,
                        torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns tokens (scenes, agents + chunks, hidden), in that order,
        after self-attention, and a mask of which tokens are real.
        """
        agent_signals = batch["scene_agent_signal_history"].flatten(
            start_dim=-2)
        agent_motion = self.scene_agent_encoder(
            batch["scene_agent_history"], batch["scene_agent_history_mask"])
        agent_tokens = agent_motion + self.signal_projection(agent_signals)

        map_tokens, map_present = pool_dots_to_chunk_tokens(
            self.map_encoder(batch["map_rows"]),
            batch["map_dot_chunk_slot"],
            agent_tokens.shape[0],
            batch["map_chunk_signal_history"].shape[1],
        )
        map_tokens = map_tokens + self.signal_projection(
            batch["map_chunk_signal_history"].flatten(start_dim=-2))

        tokens = torch.cat([agent_tokens, map_tokens], dim=1)
        agent_present = batch["scene_agent_history_mask"].any(dim=-1)
        token_present = torch.cat([agent_present, map_present], dim=1)
        token_absent = ~token_present
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=token_absent)
        return tokens, token_present

    def forward(
            self,
            batch: dict[str,
                        torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns each target's memory (targets, 1 + agents + chunks, hidden):
        its own history token, then its scene's tokens with each token's pose in
        the target's frame added, and a mask hiding tokens outside its crop.
        """
        tokens, token_present = self.scene_tokens(batch)

        own_signals = batch["agent_signal_history"].flatten(start_dim=-2)
        own_motion = self.agent_encoder(batch["agent_history"],
                                        batch["agent_history_mask"])
        own_token = own_motion + self.signal_projection(own_signals)
        own_token = own_token.unsqueeze(1)
        own_present = torch.ones(
            own_token.shape[:2],
            dtype=torch.bool,
            device=own_token.device,
        )

        # A scene with several targets is repeated once per target.
        scene_of_target = batch["target_scene_index"]
        pose_embedding = self.pose_projection(batch["token_pose"] /
                                              self.pose_divisors)
        target_view = tokens[scene_of_target] + pose_embedding
        view_present = token_present[scene_of_target] & batch["token_visible"]
        return (
            torch.cat([own_token, target_view], dim=1),
            torch.cat([own_present, view_present], dim=1),
        )


def predicted_type_index(agent_history: torch.Tensor) -> torch.Tensor:
    """Reads the one-hot object-type feature at the current history step and
    returns each agent's predicted type index.
    """
    type_one_hot = agent_history[:, contract.CURRENT_STEP_INDEX,
                                 contract.AGENT_TYPE]
    return type_one_hot[:, :contract.NUM_OBJECT_TYPES].argmax(dim=-1)


def agent_reachable_distance(agent_history: torch.Tensor) -> torch.Tensor:
    """Upper bound on how far an agent could travel over the future horizon at
    constant maximum acceleration from its current speed.
    """
    current_speed = agent_history[:, contract.CURRENT_STEP_INDEX,
                                  contract.AGENT_VELOCITY].norm(dim=-1)
    horizon = contract.FUTURE_HORIZON_SECONDS
    acceleration = contract.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED
    # d = v T + a T^2 / 2 for speed v, horizon T and acceleration a.
    return current_speed * horizon + 0.5 * acceleration * horizon**2


class ModeDecoder(nn.Module):
    """Decodes the 54 anchor-conditioned trajectory modes and their confidences
    from the memory built by SceneEncoder.
    """

    def __init__(self, unit_anchors: torch.Tensor) -> None:
        """Each query is a learned vector plus an embedding of its anchor
        endpoint, so each query starts out tied to one anchor.
        """
        super().__init__()
        assert unit_anchors.shape == (
            contract.NUM_OBJECT_TYPES,
            QUERY_COUNT,
            2,
        ), (f"unit_anchors has shape {tuple(unit_anchors.shape)};"
            f" the decoder holds one {QUERY_COUNT}-anchor set per"
            f" predicted object type")
        self.queries = nn.Parameter(
            torch.randn(QUERY_COUNT, HIDDEN_DIM) * QUERY_INITIALISATION_SCALE)
        # Anchor endpoints in metres, despite the name, which saved
        # checkpoints and anchor files depend on.
        self.register_buffer("unit_anchors",
                             unit_anchors.detach().clone().float())
        self.anchor_projection = nn.Linear(2, HIDDEN_DIM)
        self.scene_norm = nn.LayerNorm(HIDDEN_DIM)
        self.rounds = nn.ModuleList(
            decoder_round_layer() for _ in range(DECODER_ROUNDS))
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
        step_numbers = torch.arange(1,
                                    contract.FUTURE_STEPS + 1,
                                    dtype=torch.float32)
        self.register_buffer(
            "anchor_ramp",
            step_numbers / contract.FUTURE_STEPS,
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor, token_present: torch.Tensor,
                predicted_type_index: torch.Tensor) -> ModePredictions:
        """Returns trajectories and log sigmas (batch, modes, steps, 2) in
        metres, confidence logits (batch, modes), and the anchors.
        """
        batch_size = tokens.shape[0]
        selected_unit_anchors = self.unit_anchors[predicted_type_index]
        queries = self.queries + self.anchor_projection(
            selected_unit_anchors / contract.DISTANCE_NORMALISER_METRES)
        normed_tokens = self.scene_norm(tokens)
        token_absent = ~token_present
        for decoder_round in self.rounds:
            queries = decoder_round(queries,
                                    normed_tokens,
                                    memory_key_padding_mask=token_absent)

        queries = self.head_norm(queries)
        # Cast up so the cumulative sum below runs in float32 even
        # under mixed precision.
        head_output = (self.trajectory_head(queries).float().view(
            batch_size, QUERY_COUNT, 2, contract.FUTURE_STEPS, 2))
        step_displacements, log_standard_deviation = head_output.unbind(dim=2)

        # A mode is its anchor's straight constant-speed line plus
        # the running sum of the predicted per-step offsets.
        anchor_lines = (selected_unit_anchors[:, :, None, :] *
                        self.anchor_ramp[None, None, :, None])
        trajectories = step_displacements.cumsum(dim=-2) + anchor_lines
        log_standard_deviation = log_standard_deviation.clamp(
            MINIMUM_LOG_STANDARD_DEVIATION,
            MAXIMUM_LOG_STANDARD_DEVIATION,
        )
        confidence_logits = self.confidence_head(queries).squeeze(-1)

        return ModePredictions(
            trajectories,
            log_standard_deviation,
            confidence_logits,
            selected_unit_anchors,
        )


class MotionPredictor(nn.Module):
    """Top-level model, the scene encoder's memory feeding the mode decoder."""

    def __init__(self, unit_anchors: torch.Tensor) -> None:
        super().__init__()
        self.scene_encoder = SceneEncoder()
        self.mode_decoder = ModeDecoder(unit_anchors)

    @property
    def unit_anchors(self) -> torch.Tensor:
        return self.mode_decoder.unit_anchors

    def predict(self, batch: dict[str, torch.Tensor]) -> ModePredictions:
        """Runs the full scene encoder and mode decoder pipeline for one batch,
        selecting anchors by each agent's predicted type.
        """
        tokens, token_present = self.scene_encoder(batch)
        return self.mode_decoder(
            tokens,
            token_present,
            predicted_type_index(batch["agent_history"]),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        with_likelihood_outputs: bool = False
    ) -> ModePredictions | tuple[torch.Tensor, torch.Tensor]:
        """Returns everything predict() does for training, or just the
        trajectories and confidence logits for inference and export.

        Training calls this rather than predict() so that a multi-process
        wrapper around the model sees the call.
        """
        predictions = self.predict(batch)
        if with_likelihood_outputs:
            return predictions
        return predictions.trajectories, predictions.confidence_logits


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
