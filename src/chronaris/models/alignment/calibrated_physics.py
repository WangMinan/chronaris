"""Explicit, train-calibrated kinematic relations in physical units."""

from dataclasses import asdict, dataclass, replace
import math
from typing import Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class KinematicRelation:
    source: str
    target: str
    component: str
    unit: str
    provenance: str
    source_multiplier: float = 1.0
    target_multiplier: float = 1.0
    residual_scale: float = 1.0

    def __post_init__(self):
        if not all((self.source, self.target, self.component, self.unit, self.provenance)):
            raise ValueError("kinematic relation requires names, units and provenance")
        if not all(math.isfinite(v) and v > 0 for v in (
            self.source_multiplier, self.target_multiplier, self.residual_scale,
        )):
            raise ValueError("kinematic conversion and residual scales must be finite and positive")


SIMULATION_RELATIONS = tuple(
    KinematicRelation(f"vehicle.{source}", f"vehicle.{target}", component, unit,
                      "aviation_dual_stream vehicle_g1/vehicle_g2 generation equations")
    for source, target, component, unit in (
        ("speed_mps", "longitudinal_acc_mps2", "vehicle_rigid_body_translation", "m/s^2"),
        ("altitude_m", "vertical_speed_mps", "vehicle_rigid_body_vertical", "m/s"),
        ("roll_rad", "roll_rate_rps", "vehicle_rigid_body_rotation", "rad/s"),
        ("pitch_rad", "pitch_rate_rps", "vehicle_rigid_body_rotation", "rad/s"),
        ("yaw_rad", "yaw_rate_rps", "vehicle_rigid_body_rotation", "rad/s"),
    )
)


def relation_samples(values, times, mask, feature_names, relation):
    indices = {name: index for index, name in enumerate(feature_names)}
    if relation.source not in indices or relation.target not in indices:
        raise ValueError("kinematic relation references absent fields")
    source, target = indices[relation.source], indices[relation.target]
    dt = times[:, 1:] - times[:, :-1]
    valid = mask[:, 1:, source] & mask[:, :-1, source] & mask[:, 1:, target] & (dt > 0)
    source_values = values[..., source] * relation.source_multiplier
    rhs = values[:, 1:, target] * relation.target_multiplier
    derivative = (source_values[:, 1:] - source_values[:, :-1]) / dt.to(values.dtype).clamp_min(1e-9)
    return (derivative - rhs)[valid], rhs[valid]


def fit_physics_calibration(normalizer, provider, *, train_sample_ids: Sequence[str],
                            vehicle_feature_names, relations: Sequence[KinematicRelation], batch_size=4):
    """Fit relation scales using only the normalizer's exact training membership."""
    ids = tuple(train_sample_ids)
    if not ids or set(ids) != set(normalizer.fit_sample_ids) or len(ids) != len(set(ids)):
        raise ValueError("physics calibration must match the normalizer's unique training samples")
    if batch_size <= 0:
        raise ValueError("physics calibration batch size must be positive")
    parts = {relation: ([], []) for relation in relations}
    for offset in range(0, len(ids) if parts else 0, batch_size):
        selected = ids[offset:offset + batch_size]
        batch = provider(selected)
        if tuple(batch.sample_ids) != selected:
            raise ValueError("physics calibration provider changed sample membership")
        for relation in relations:
            residual, rhs = relation_samples(
                batch.vehicle_values.double(), batch.vehicle_timestamps_s.double(),
                batch.vehicle_feature_mask & batch.vehicle_point_mask.unsqueeze(-1),
                vehicle_feature_names, relation,
            )
            if residual.numel():
                parts[relation][0].append(residual.detach().cpu())
                parts[relation][1].append(rhs.detach().cpu())
    calibrated, unavailable = [], []
    for relation, (residual_parts, rhs_parts) in parts.items():
        if not residual_parts:
            unavailable.append({**asdict(relation), "reason": "no_training_derivative_pairs"})
            continue
        iqrs = []
        for values in (torch.cat(residual_parts), torch.cat(rhs_parts)):
            if not torch.isfinite(values).all():
                raise ValueError("non-finite training kinematic observations")
            q = torch.quantile(values, torch.tensor([0.25, 0.75], dtype=values.dtype))
            iqrs.append(float(q[1] - q[0]))
        calibrated.append(asdict(replace(relation, residual_scale=max(*iqrs, 1e-3))))
    return {
        "format": "chronaris.kinematic_calibration.v4",
        "fit_sample_hash": normalizer.fit_sample_hash,
        "fit_sample_ids": list(ids), "feature_names": list(vehicle_feature_names),
        "center": normalizer.vehicle.center.tolist(), "scale": normalizer.vehicle.scale.tolist(),
        "relations": calibrated, "unavailable_relations": unavailable,
    }


def calibrated_kinematic_losses(reconstruction, stream, calibration, *, huber_delta=1.0):
    if calibration.get("format") != "chronaris.kinematic_calibration.v4":
        raise ValueError("unsupported kinematic calibration format")
    if tuple(calibration["feature_names"]) != tuple(stream.feature_names):
        raise ValueError("kinematic calibration feature order changed")
    center = reconstruction.new_tensor(calibration["center"])
    scale = reconstruction.new_tensor(calibration["scale"])
    if center.shape != reconstruction.shape[-1:] or scale.shape != center.shape:
        raise ValueError("kinematic normalization shape mismatch")
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("invalid kinematic normalization statistics")
    physical = reconstruction.float() * scale.float() + center.float()
    mask = stream.feature_valid_mask & stream.mask.unsqueeze(-1)
    rows = []
    for spec in calibration["relations"]:
        relation = KinematicRelation(**spec)
        residual, _rhs = relation_samples(physical, stream.offsets_s, mask, stream.feature_names, relation)
        loss = (F.huber_loss(residual / relation.residual_scale, torch.zeros_like(residual), delta=huber_delta)
                if residual.numel() else reconstruction.sum() * 0)
        rows.append((relation.component, loss, int(residual.numel())))
    return rows


def observation_anchor_loss(output, batch):
    losses, count = [], 0
    for name in ("physiology", "vehicle"):
        stream, decoded = getattr(batch, name), getattr(output, name).reconstructions
        valid = stream.feature_valid_mask & stream.mask.unsqueeze(-1)
        if valid.any():
            losses.append(F.huber_loss(decoded[valid], stream.values[valid], delta=1.0))
            count += int(valid.sum())
    return (torch.stack(losses).mean() if losses else output.vehicle.reconstructions.sum() * 0), count
