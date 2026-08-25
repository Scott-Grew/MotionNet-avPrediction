import hashlib

import torch
from torch import nn

from womd import contract

HIDDEN_DIM = 128

def agent_feature_divisors():
    divisors = torch.ones(contract.AGENT_FEATURE_DIM)
    divisors[contract.AGENT_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.AGENT_VELOCITY] = contract.VELOCITY_NORMALISER_METRES_PER_SECOND
    divisors[contract.AGENT_DIMENSIONS] = contract.DIMENSION_NORMALISER_METRES
    return divisors

def map_feature_divisors():
    divisors = torch.ones(contract.MAP_FEATURE_DIM)
    divisors[contract.MAP_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.MAP_SPEED_LIMIT] = contract.SPEED_LIMIT_NORMALISER_MILES_PER_HOUR
    divisors[contract.MAP_STOP_POINT] = contract.DISTANCE_NORMALISER_METRES
    return divisors

class AgentHistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_width = contract.HISTORY_STEPS * (contract.AGENT_FEATURE_DIM + 1)
        self.register_buffer("feature_divisors", agent_feature_divisors())
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, agent_history, agent_history_mask):
        validity = agent_history_mask.unsqueeze(-1).to(agent_history.dtype)
        masked_history = agent_history / self.feature_divisors * validity
        flattened = torch.cat([masked_history, validity], dim=-1).flatten(start_dim=-2)
        return self.network(flattened)

class MapDotEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_divisors", map_feature_divisors())
        self.register_buffer(
            "crossing_code_rows", torch.eye(contract.NUM_BOUNDARY_CROSSING_CODES)
        )
        input_width = (contract.MAP_FEATURE_DIM - 2) + 2 * contract.NUM_BOUNDARY_CROSSING_CODES
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, map_rows):
        scaled = map_rows / self.feature_divisors
        crossing_codes = map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:].long()
        crossing_one_hot = self.crossing_code_rows[crossing_codes].flatten(start_dim=-2)
        return self.network(torch.cat(
            [scaled[:, :contract.MAP_LEFT_BOUNDARY_CROSSING], crossing_one_hot.to(scaled.dtype)],
            dim=-1,
        ))

def pool_dots_to_polyline_tokens(dot_embeddings, dot_polyline_slot, batch_size, max_polylines):
    hidden_width = dot_embeddings.shape[-1]
    tokens = torch.full(
        (batch_size * max_polylines, hidden_width), float("-inf"),
        dtype=dot_embeddings.dtype, device=dot_embeddings.device,
    )
    tokens = tokens.scatter_reduce(
        0, dot_polyline_slot.unsqueeze(-1).expand(-1, hidden_width),
        dot_embeddings, reduce="amax", include_self=True,
    )
    polyline_present = tokens[:, 0] > float("-inf")
    tokens = torch.where(polyline_present.unsqueeze(-1), tokens, torch.zeros_like(tokens))
    return (
        tokens.view(batch_size, max_polylines, hidden_width),
        polyline_present.view(batch_size, max_polylines),
    )

ATTENTION_HEAD_COUNT = 4
SCENE_ATTENTION_ROUNDS = 6
FEEDFORWARD_DIM = 4 * HIDDEN_DIM

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert HIDDEN_DIM % ATTENTION_HEAD_COUNT == 0
        self.query_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.key_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.value_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def split_heads(self, projected):
        batch_size, token_count, _ = projected.shape
        per_head = HIDDEN_DIM // ATTENTION_HEAD_COUNT
        return projected.view(batch_size, token_count, ATTENTION_HEAD_COUNT, per_head).transpose(1, 2)

    def forward(self, query_tokens, key_value_tokens, key_present):
        queries = self.split_heads(self.query_projection(query_tokens))
        keys = self.split_heads(self.key_projection(key_value_tokens))
        values = self.split_heads(self.value_projection(key_value_tokens))
        attended = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=key_present[:, None, None, :]
        )
        merged = attended.transpose(1, 2).flatten(start_dim=-2)
        return self.output_projection(merged)

class SceneAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(HIDDEN_DIM)
        self.attention = MultiHeadAttention()
        self.feedforward_norm = nn.LayerNorm(HIDDEN_DIM)
        self.feedforward = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, HIDDEN_DIM),
        )

    def forward(self, tokens, token_present):
        normed = self.attention_norm(tokens)
        tokens = tokens + self.attention(normed, normed, token_present)
        return tokens + self.feedforward(self.feedforward_norm(tokens))

class SceneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent_encoder = AgentHistoryEncoder()
        self.neighbour_encoder = AgentHistoryEncoder()
        self.map_encoder = MapDotEncoder()
        self.signal_projection = nn.Linear(contract.POLYLINE_SIGNAL_DIM, HIDDEN_DIM, bias=False)
        self.layers = nn.ModuleList(SceneAttentionLayer() for _ in range(SCENE_ATTENTION_ROUNDS))

    def forward(self, batch):
        agent_token = (
            self.agent_encoder(batch["agent_history"], batch["agent_history_mask"])
            + self.signal_projection(batch["agent_signal_history"].flatten(start_dim=-2))
        ).unsqueeze(1)
        neighbour_tokens = self.neighbour_encoder(
            batch["neighbour_history"], batch["neighbour_history_mask"]
        ) + self.signal_projection(batch["neighbour_signal_history"].flatten(start_dim=-2))
        map_tokens, map_present = pool_dots_to_polyline_tokens(
            self.map_encoder(batch["map_rows"]), batch["map_dot_polyline_slot"],
            agent_token.shape[0], int(batch["max_polylines_in_batch"]),
        )
        map_tokens = map_tokens + self.signal_projection(
            batch["map_chunk_signal_history"].flatten(start_dim=-2)
        )

        tokens = torch.cat([agent_token, neighbour_tokens, map_tokens], dim=1)
        agent_present = torch.ones(agent_token.shape[:2], dtype=torch.bool, device=tokens.device)
        neighbour_present = batch["neighbour_history_mask"].any(dim=-1)
        token_present = torch.cat([agent_present, neighbour_present, map_present], dim=1)

        for layer in self.layers:
            tokens = layer(tokens, token_present)
        return tokens, token_present

QUERY_COUNT = 54
PRUNE_DISTANCE_METRES = 2.5
ANCHOR_DIRECTION_COUNT = 9
ANCHOR_DISTANCE_COUNT = 6
assert ANCHOR_DIRECTION_COUNT * ANCHOR_DISTANCE_COUNT == QUERY_COUNT
MINIMUM_LOG_STANDARD_DEVIATION = -1.609
MAXIMUM_LOG_STANDARD_DEVIATION = 5.0

def unit_anchor_offsets():
    direction_indices = torch.arange(ANCHOR_DIRECTION_COUNT).repeat_interleave(ANCHOR_DISTANCE_COUNT)
    distance_indices = torch.arange(ANCHOR_DISTANCE_COUNT).repeat(ANCHOR_DIRECTION_COUNT)
    angles = 2 * torch.pi * direction_indices / ANCHOR_DIRECTION_COUNT
    fractions = (distance_indices + 1) / ANCHOR_DISTANCE_COUNT
    return fractions.unsqueeze(-1) * torch.stack([angles.cos(), angles.sin()], dim=-1)

def unit_anchor_offsets_per_type():
    return unit_anchor_offsets().repeat(contract.NUM_OBJECT_TYPES, 1, 1)

def predicted_type_index(agent_history):
    type_one_hot = agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE]
    return type_one_hot[:, : contract.NUM_OBJECT_TYPES].argmax(dim=-1)

def agent_reachable_distance(agent_history):
    current_speed = agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY].norm(dim=-1)
    return (
        current_speed * contract.FUTURE_HORIZON_SECONDS
        + 0.5 * contract.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED * contract.FUTURE_HORIZON_SECONDS ** 2
    )

class ModeDecoder(nn.Module):
    def __init__(self, unit_anchors):
        super().__init__()
        assert unit_anchors.shape == (contract.NUM_OBJECT_TYPES, QUERY_COUNT, 2), (
            f"unit_anchors has shape {tuple(unit_anchors.shape)}; the decoder holds one"
            f" {QUERY_COUNT}-anchor set per predicted object type"
        )
        self.queries = nn.Parameter(torch.randn(QUERY_COUNT, HIDDEN_DIM) * 0.02)
        self.register_buffer("unit_anchors", unit_anchors.detach().clone().float())
        self.anchor_projection = nn.Linear(2, HIDDEN_DIM)
        self.scene_norm = nn.LayerNorm(HIDDEN_DIM)
        self.query_norm = nn.LayerNorm(HIDDEN_DIM)
        self.attention = MultiHeadAttention()
        self.trajectory_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, 2 * contract.FUTURE_STEPS * 2),
        )
        self.confidence_head = nn.Linear(HIDDEN_DIM, 1)
        self.register_buffer(
            "anchor_ramp",
            torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32) / contract.FUTURE_STEPS,
            persistent=False,
        )

    def forward(self, tokens, token_present, predicted_type_index):
        batch_size = tokens.shape[0]
        selected_unit_anchors = self.unit_anchors[predicted_type_index]
        queries = self.queries + self.anchor_projection(
            selected_unit_anchors / contract.DISTANCE_NORMALISER_METRES
        )
        queries = queries + self.attention(
            self.query_norm(queries), self.scene_norm(tokens), token_present
        )
        head_output = self.trajectory_head(queries).float().view(
            batch_size, QUERY_COUNT, 2, contract.FUTURE_STEPS, 2
        )
        step_offsets, log_standard_deviation = head_output.unbind(dim=2)
        trajectories = (
            step_offsets + selected_unit_anchors[:, :, None, :] * self.anchor_ramp[None, None, :, None]
        )
        log_standard_deviation = log_standard_deviation.clamp(
            MINIMUM_LOG_STANDARD_DEVIATION, MAXIMUM_LOG_STANDARD_DEVIATION
        )
        confidence_logits = self.confidence_head(queries).squeeze(-1)
        return trajectories, log_standard_deviation, confidence_logits, selected_unit_anchors

def prune_modes(trajectories, confidence_logits):
    kept_indices = []
    dropped_indices = []
    endpoints = trajectories[:, -1]
    for mode_index in torch.argsort(confidence_logits, descending=True).tolist():
        if kept_indices and bool(
            (torch.cdist(endpoints[mode_index][None], endpoints[kept_indices]) < PRUNE_DISTANCE_METRES).any()
        ):
            dropped_indices.append(mode_index)
            continue
        kept_indices.append(mode_index)
        if len(kept_indices) == contract.NUM_PREDICTED_MODES:
            break
    kept_indices.extend(dropped_indices[: contract.NUM_PREDICTED_MODES - len(kept_indices)])
    kept = torch.tensor(kept_indices, device=trajectories.device)
    return trajectories[kept], confidence_logits[kept]

def prune_modes_batched_with_kept_count(trajectories, confidence_logits):
    batch_size, mode_count = confidence_logits.shape
    device = trajectories.device
    endpoints = trajectories[:, :, -1]
    confidence_order = torch.argsort(confidence_logits, dim=-1, descending=True)
    slot_positions = torch.arange(contract.NUM_PREDICTED_MODES, device=device)

    kept_indices = torch.zeros(batch_size, contract.NUM_PREDICTED_MODES, dtype=torch.long, device=device)
    kept_endpoints = torch.zeros_like(endpoints[:, : contract.NUM_PREDICTED_MODES])
    kept_slot_filled = torch.zeros(batch_size, contract.NUM_PREDICTED_MODES, dtype=torch.bool, device=device)
    kept_count = torch.zeros(batch_size, dtype=torch.long, device=device)
    dropped_indices = torch.zeros_like(kept_indices)
    dropped_count = torch.zeros_like(kept_count)

    for walk_position in range(mode_count):
        candidate_index = confidence_order[:, walk_position]
        candidate_endpoint = endpoints.gather(1, candidate_index[:, None, None].expand(-1, -1, 2))
        separations = torch.cdist(candidate_endpoint, kept_endpoints).squeeze(1)
        too_close = ((separations < PRUNE_DISTANCE_METRES) & kept_slot_filled).any(dim=-1)
        still_walking = kept_count < contract.NUM_PREDICTED_MODES

        keeps = still_walking & ~too_close
        keep_slot = keeps[:, None] & (slot_positions[None, :] == kept_count[:, None])
        kept_indices = torch.where(keep_slot, candidate_index[:, None], kept_indices)
        kept_endpoints = torch.where(keep_slot[:, :, None], candidate_endpoint, kept_endpoints)
        kept_slot_filled = kept_slot_filled | keep_slot
        kept_count = kept_count + keeps.long()

        drops = still_walking & too_close
        drop_slot = drops[:, None] & (slot_positions[None, :] == dropped_count[:, None])
        dropped_indices = torch.where(drop_slot, candidate_index[:, None], dropped_indices)
        dropped_count = dropped_count + drops.long()

        if bool((kept_count == contract.NUM_PREDICTED_MODES).all()):
            break

    backfill_positions = (slot_positions[None, :] - kept_count[:, None]).clamp(min=0)
    final_indices = torch.where(
        slot_positions[None, :] < kept_count[:, None],
        kept_indices,
        dropped_indices.gather(1, backfill_positions),
    )
    trajectory_selector = final_indices[:, :, None, None].expand(-1, -1, *trajectories.shape[2:])
    return (
        trajectories.gather(1, trajectory_selector),
        confidence_logits.gather(1, final_indices),
        kept_count,
    )

def prune_modes_batched(trajectories, confidence_logits):
    kept_trajectories, kept_confidence_logits, _ = prune_modes_batched_with_kept_count(
        trajectories, confidence_logits
    )
    return kept_trajectories, kept_confidence_logits

def aggregated_confidences(trajectories, confidence_logits, kept_trajectories):
    probabilities = torch.softmax(confidence_logits, dim=-1)
    separations = torch.cdist(trajectories[:, :, -1], kept_trajectories[:, :, -1])
    nearest_kept = separations.argmin(dim=-1)
    aggregated = torch.zeros(
        kept_trajectories.shape[:2], dtype=probabilities.dtype, device=probabilities.device
    )
    return aggregated.scatter_add(1, nearest_kept, probabilities)

class MotionPredictor(nn.Module):
    def __init__(self, unit_anchors):
        super().__init__()
        self.scene_encoder = SceneEncoder()
        self.mode_decoder = ModeDecoder(unit_anchors)

    @property
    def unit_anchors(self):
        return self.mode_decoder.unit_anchors

    def predict(self, batch):
        tokens, token_present = self.scene_encoder(batch)
        return self.mode_decoder(tokens, token_present, predicted_type_index(batch["agent_history"]))

    def forward(self, batch):
        trajectories, _, confidence_logits, _ = self.predict(batch)
        return trajectories, confidence_logits

def parameter_fingerprint(model_state):
    parameter_description = ",".join(
        f"{name}:{tuple(tensor.shape)}" for name, tensor in sorted(model_state.items())
    )
    return hashlib.sha256(parameter_description.encode()).hexdigest()

def load_checkpoint_state(checkpoint_path, map_location="cpu", allow_version_mismatch=False):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_code_version = checkpoint.get("code_version")
    if not allow_version_mismatch:
        assert checkpoint_code_version == contract.STAGING_CODE_VERSION, (
            f"{checkpoint_path} was saved under code_version {checkpoint_code_version!r}, but the"
            f" working tree is at contract.STAGING_CODE_VERSION {contract.STAGING_CODE_VERSION!r}."
            f" Loading it with strict=True would silently accept a byte-compatible but"
            f" architecturally different state dict. Pass allow_version_mismatch=True to load it"
            f" anyway."
        )
    stamped_fingerprint = checkpoint.get("parameter_fingerprint")
    if stamped_fingerprint is not None:
        recomputed_fingerprint = parameter_fingerprint(checkpoint["model_state"])
        assert stamped_fingerprint == recomputed_fingerprint, (
            f"{checkpoint_path} carries parameter_fingerprint {stamped_fingerprint} but its"
            f" model_state hashes to {recomputed_fingerprint}: the state dict was altered after"
            f" the stamp was written."
        )
    return checkpoint

def load_anchor_file(anchors_path):
    import numpy as np
    with np.load(anchors_path) as anchors_file:
        contract.check_artifact_provenance(
            anchors_file["provenance"] if "provenance" in anchors_file else None,
            anchors_path, "Refit them with fit_anchors.py.",
        )
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    assert unit_anchors.shape == (contract.NUM_OBJECT_TYPES, QUERY_COUNT, 2), (
        f"{anchors_path} holds unit_anchors of shape {tuple(unit_anchors.shape)}, but the model"
        f" needs ({contract.NUM_OBJECT_TYPES}, {QUERY_COUNT}, 2). Re-run fit_anchors.py"
    )
    return unit_anchors
