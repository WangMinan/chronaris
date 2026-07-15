"""Matched-clean Dingxin input and time-shortcut controls."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from chronaris.evaluation.application_tasks.task_stability_contracts import stable_sha256
from chronaris.representation import DualStreamObservationBatch


def build_clean_input_contract(
    *,
    role_path: str,
    vehicle_raw_to_index: Mapping[str, Mapping[str, int]],
    vehicle_channel_count: int,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Identify the same train-independent time-like fields removed by task 1B."""

    roles = pd.read_csv(role_path)
    text = (
        roles[
            [
                "feature_name",
                "source_field",
                "display_label",
                "unit_hint",
                "semantic_category",
            ]
        ]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    time_like = set(
        roles.loc[
            roles["allowed_in_maneuver_input"].astype(bool)
            & text.str.contains(r"time|timestamp|rtc|时间", regex=True),
            "feature_name",
        ].astype(str)
    )
    removed = tuple(
        sorted(
            {
                int(index)
                for mapping in vehicle_raw_to_index.values()
                for name, index in mapping.items()
                if str(name) in time_like
            }
        )
    )
    if not removed or len(removed) >= vehicle_channel_count:
        raise ValueError("matched-clean time-field exclusion is empty or removes all channels")
    payload = {
        "explicit_query_or_block_position_used": False,
        "view_pilot_or_sortie_identity_used": False,
        "time_like_allowed_field_count": len(time_like),
        "time_like_present_channel_count": len(removed),
        "time_like_present_channels_sha256": stable_sha256(removed),
        "vehicle_channel_count_before": int(vehicle_channel_count),
        "vehicle_channel_count_after": int(vehicle_channel_count - len(removed)),
        "removal_mode": "zero_value_false_feature_mask_recomputed_point_mask",
    }
    payload["input_feature_contract_sha256"] = stable_sha256(payload)
    return removed, payload


def mask_time_like_vehicle_channels(
    batch: DualStreamObservationBatch,
    removed_indices: Sequence[int],
) -> DualStreamObservationBatch:
    """Remove explicit time fields without changing checkpoint tensor shapes."""

    indices = torch.as_tensor(
        tuple(int(value) for value in removed_indices),
        dtype=torch.long,
        device=batch.vehicle_values.device,
    )
    values = batch.vehicle_values.clone()
    feature_mask = batch.vehicle_feature_mask.clone()
    age = batch.vehicle_observation_age_s.clone()
    values.index_fill_(-1, indices, 0.0)
    feature_mask.index_fill_(-1, indices, False)
    age.index_fill_(-1, indices, 0.0)
    return replace(
        batch,
        vehicle_values=values,
        vehicle_feature_mask=feature_mask,
        vehicle_observation_age_s=age,
        vehicle_point_mask=feature_mask.any(dim=-1),
    )


def clean_guarded_provider(
    base_provider,
    *,
    allowed_sample_ids: Sequence[str],
    removed_indices: Sequence[int],
):
    """Build a cached provider that rejects every sample outside inner roles."""

    allowed = set(str(value) for value in allowed_sample_ids)
    cache = {}
    audit = {"request_count": 0, "cache_hit_count": 0, "forbidden_request_count": 0}

    def provider(sample_ids):
        identifiers = tuple(str(value) for value in sample_ids)
        audit["request_count"] += 1
        if not set(identifiers) <= allowed:
            audit["forbidden_request_count"] += 1
            raise PermissionError("matched-clean provider rejected outer or unknown samples")
        if identifiers in cache:
            audit["cache_hit_count"] += 1
            return cache[identifiers]
        batch = mask_time_like_vehicle_channels(base_provider(identifiers), removed_indices)
        cache[identifiers] = batch
        return batch

    return provider, audit


def normalized_phase(contexts: pd.DataFrame, sample_ids: Sequence[str]) -> np.ndarray:
    """Return identity-free within-sortie flight progress for diagnostics only."""

    lookup = contexts.set_index("context_id")
    values = []
    for sample_id in sample_ids:
        row = lookup.loc[str(sample_id)]
        sortie = contexts[contexts["sortie_id"].astype(str) == str(row["sortie_id"])]
        low = float(sortie["end_offset_ms"].min())
        high = float(sortie["end_offset_ms"].max())
        values.append((float(row["end_offset_ms"]) - low) / max(high - low, 1.0))
    return np.asarray(values, dtype=np.float64)


def phase_residualize(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    train_phase: Sequence[float],
    validation_phase: Sequence[float],
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove train-fitted polynomial flight-progress components from features."""

    left = np.asarray(train, dtype=np.float64).reshape(len(train), -1)
    right = np.asarray(validation, dtype=np.float64).reshape(len(validation), -1)
    polynomial = PolynomialFeatures(degree=3, include_bias=True)
    train_time = polynomial.fit_transform(np.asarray(train_phase).reshape(-1, 1))
    validation_time = polynomial.transform(np.asarray(validation_phase).reshape(-1, 1))
    time_scaler = StandardScaler().fit(train_time)
    model = Ridge(alpha=alpha).fit(time_scaler.transform(train_time), left)
    return (
        (left - model.predict(time_scaler.transform(train_time))).astype(np.float32),
        (right - model.predict(time_scaler.transform(validation_time))).astype(np.float32),
    )
