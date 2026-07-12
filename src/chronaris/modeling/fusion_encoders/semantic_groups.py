"""Semantic-group observation encoding for high-dimensional vehicle streams."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


VEHICLE_SEMANTIC_GROUPS = (
    "attitude_and_rate",
    "acceleration_and_load",
    "speed_and_altitude",
    "control_input",
    "status_mass_and_event",
    "other_numeric",
)

_GROUP_KEYWORDS = {
    "attitude_and_rate": (
        "pitch", "roll", "yaw", "attitude", "angular", "omega",
        "俯仰", "滚转", "偏航", "姿态", "角速度",
    ),
    "acceleration_and_load": (
        "accel", "acceleration", "load_factor", "normal_load", "load_g", "g_force", "overload",
        "加速度", "过载", "载荷因数",
    ),
    "speed_and_altitude": (
        "speed", "velocity", "altitude", "height", "vertical", "airspeed",
        "速度", "高度", "垂直速度", "空速",
    ),
    "control_input": (
        "control", "stick", "pedal", "throttle", "rudder", "aileron",
        "elevator", "操纵", "驾驶杆", "脚蹬", "油门", "舵面",
    ),
    "status_mass_and_event": (
        "status", "state", "mass", "weight", "event", "gear", "flap",
        "fuel", "状态", "质量", "重量", "事件", "起落架", "襟翼", "燃油",
    ),
}


@dataclass(frozen=True, slots=True)
class VehicleSemanticGroupMap:
    feature_names: tuple[str, ...]
    group_indices: tuple[tuple[str, tuple[int, ...]], ...]
    mapping_sha256: str

    @property
    def groups(self) -> Mapping[str, tuple[int, ...]]:
        return dict(self.group_indices)

    def to_manifest(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "group_indices": {
                name: list(indices) for name, indices in self.group_indices
            },
            "mapping_sha256": self.mapping_sha256,
        }


def build_vehicle_semantic_group_map(
    feature_names: tuple[str, ...],
    *,
    field_labels: Mapping[str, str] | None = None,
) -> VehicleSemanticGroupMap:
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("vehicle feature names must be non-empty and unique")
    labels = field_labels or {}
    grouped: dict[str, list[int]] = {name: [] for name in VEHICLE_SEMANTIC_GROUPS}
    for index, feature_name in enumerate(feature_names):
        text = f"{feature_name} {labels.get(feature_name, '')}".casefold()
        group = "other_numeric"
        for candidate in VEHICLE_SEMANTIC_GROUPS[:-1]:
            if any(keyword.casefold() in text for keyword in _GROUP_KEYWORDS[candidate]):
                group = candidate
                break
        grouped[group].append(index)
    group_indices = tuple(
        (name, tuple(grouped[name])) for name in VEHICLE_SEMANTIC_GROUPS
    )
    payload = json.dumps(
        {
            "feature_names": list(feature_names),
            "field_labels": {name: labels.get(name, "") for name in feature_names},
            "groups": {name: list(indices) for name, indices in group_indices},
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return VehicleSemanticGroupMap(
        feature_names=feature_names,
        group_indices=group_indices,
        mapping_sha256=hashlib.sha256(payload).hexdigest(),
    )


class SemanticGroupedObservationEncoder(nn.Module):
    """Encode each vehicle semantic group before masked group aggregation."""

    def __init__(
        self,
        group_map: VehicleSemanticGroupMap,
        *,
        embedding_dim: int,
        group_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or group_hidden_dim <= 0:
            raise ValueError("semantic group dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("semantic group dropout is invalid")
        self.group_map = group_map
        self.feature_dim = len(group_map.feature_names)
        self.embedding_dim = embedding_dim
        self.group_hidden_dim = group_hidden_dim
        self.group_encoders = nn.ModuleDict()
        for group_name, indices in group_map.group_indices:
            input_dim = max(1, len(indices)) * 3
            self.group_encoders[group_name] = nn.Sequential(
                nn.Linear(input_dim, group_hidden_dim),
                nn.LayerNorm(group_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(group_hidden_dim, group_hidden_dim),
            )
        self.group_missing_tokens = nn.Parameter(
            torch.zeros(len(VEHICLE_SEMANTIC_GROUPS), group_hidden_dim)
        )
        self.group_gate = nn.Linear(group_hidden_dim, 1)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(group_hidden_dim),
            nn.Linear(group_hidden_dim, embedding_dim),
        )

    def forward(
        self,
        values: torch.Tensor,
        feature_valid_mask: torch.Tensor,
        observation_age_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError("semantic grouped values have invalid shape")
        if feature_valid_mask.shape != values.shape:
            raise ValueError("semantic grouped mask shape mismatch")
        if observation_age_s is None:
            observation_age_s = torch.zeros_like(values)
        if observation_age_s.shape != values.shape:
            raise ValueError("semantic grouped observation age shape mismatch")
        tokens = []
        availability = []
        for group_index, (group_name, indices) in enumerate(
            self.group_map.group_indices
        ):
            if indices:
                index = torch.tensor(indices, dtype=torch.long, device=values.device)
                group_values = values.index_select(-1, index)
                group_mask = feature_valid_mask.index_select(-1, index)
                group_age = observation_age_s.index_select(-1, index)
                normalized_age = torch.where(
                    group_mask,
                    torch.log1p(group_age.clamp(min=0.0, max=30.0))
                    / torch.log(values.new_tensor(31.0)),
                    torch.zeros_like(group_age),
                )
                encoder_input = torch.cat(
                    (
                        group_values,
                        group_mask.to(group_values.dtype),
                        normalized_age,
                    ),
                    dim=-1,
                )
                token = self.group_encoders[group_name](encoder_input)
                available = group_mask.any(dim=-1)
            else:
                token = values.new_zeros(
                    (*values.shape[:2], self.group_hidden_dim)
                )
                available = torch.zeros(
                    values.shape[:2], dtype=torch.bool, device=values.device
                )
            missing = self.group_missing_tokens[group_index].view(1, 1, -1)
            tokens.append(torch.where(available.unsqueeze(-1), token, missing))
            availability.append(available)
        stacked = torch.stack(tokens, dim=2)
        available_groups = torch.stack(availability, dim=2)
        logits = self.group_gate(stacked).squeeze(-1)
        any_available = available_groups.any(dim=-1, keepdim=True)
        masked_logits = logits.masked_fill(
            ~available_groups,
            torch.finfo(logits.dtype).min,
        )
        safe_logits = torch.where(any_available, masked_logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1).masked_fill(~available_groups, 0.0)
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(
            denominator > 0,
            weights / denominator.clamp_min(torch.finfo(weights.dtype).eps),
            torch.full_like(weights, 1.0 / len(VEHICLE_SEMANTIC_GROUPS)),
        )
        pooled = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        return self.output_projection(pooled)
