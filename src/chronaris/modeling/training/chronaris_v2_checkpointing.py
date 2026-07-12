"""Checkpoint payload and recovery helpers for Chronaris v2 training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch

from chronaris.modeling.fusion_encoders.chronaris_v2 import V2_SUBSPACE_SLICES
from chronaris.representation.contracts import RepresentationContractError


@dataclass(frozen=True, slots=True)
class ChronarisV2TrainingResult:
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    best_epoch: int
    completed_epochs: int
    stopped_early: bool
    best_public_selection_loss: float
    training_elapsed_s: float
    epoch_rows: tuple[Mapping[str, object], ...]
    gradient_rows: tuple[Mapping[str, object], ...]


def build_v2_checkpoint_payload(**values):
    encoder = values["encoder"]
    backbone = encoder.backbone
    return {
        "format": "chronaris.common_pretraining_checkpoint.v2",
        "training_status": "running",
        "method_name": "chronaris",
        "architecture_version": "v2",
        "protocol_sha256": values["protocol_hash"],
        "encoder_state_dict": encoder.state_dict(),
        "common_head_state_dict": values["common_heads"].state_dict(),
        "head_state_dict": values["common_heads"].state_dict(),
        "v2_head_state_dict": values["v2_heads"].state_dict(),
        "optimizer_state_dict": values["optimizer"].state_dict(),
        "normalizer": dict(values["normalizer"].to_manifest()),
        "fold": values["fold"].to_dict(),
        "candidate_config": asdict(values["candidate"]),
        "config": asdict(values["config"]),
        "encoder_manifest": dict(encoder.config_manifest()),
        "physiology_feature_names": list(backbone.config.physiology_feature_names),
        "vehicle_feature_names": list(backbone.config.vehicle_feature_names),
        "vehicle_field_labels": [list(value) for value in backbone.config.field_labels],
        "subspace_slices": {
            name: list(bounds) for name, bounds in V2_SUBSPACE_SLICES.items()
        },
        "semantic_group_mapping_sha256": backbone.semantic_group_map.mapping_sha256,
        "physics_mapping_sha256": backbone.physics_mapping_sha256,
        "lag_config": dict(backbone.lag_config_manifest()),
        "normalization_inverse_transform": dict(values["normalizer"].to_manifest()),
        "physiology_teacher_manifest": values["physiology_teacher_manifest"],
        "initialization_manifest": values["initialization_manifest"],
        "best_epoch": values["best_epoch"],
        "completed_epochs": values["completed_epochs"],
        "best_public_selection_loss": values["best_score"],
        "stopped_early": False,
        "training_elapsed_s": values["elapsed"],
        "epoch_rows": list(values["epoch_rows"]),
        "gradient_rows": list(values["gradient_rows"]),
        "conflict_history": list(values["controller"].conflict_history),
        "pcgrad_active": values["controller"].use_pcgrad,
        "parameter_count": encoder.parameter_count,
        "head_parameter_count": sum(
            parameter.numel()
            for parameter in (
                *values["common_heads"].parameters(),
                *values["v2_heads"].parameters(),
            )
        ),
        "training_rows": [],
        "augmentation_rows": [],
        "label_used_for_encoder_training": False,
        "simulation_oracle_opened": False,
        "locked_test_opened": False,
        "selection_uses_public_pretext_only": True,
        "early_stopping_uses_public_pretext_only": True,
    }


def v2_protocol_hash(**payload):
    code_digest = hashlib.sha256()
    module_root = Path(__file__).parent
    for name in (
        "chronaris_v2_training.py",
        "chronaris_v2_checkpointing.py",
        "chronaris_v2_distillation.py",
        "chronaris_v2_objectives.py",
    ):
        code_digest.update(name.encode())
        code_digest.update((module_root / name).read_bytes())
    payload["code_sha256"] = code_digest.hexdigest()
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save_v2_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_v2_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "chronaris.common_pretraining_checkpoint.v2":
        raise RepresentationContractError("unsupported Chronaris v2 checkpoint")
    if payload.get("architecture_version") != "v2":
        raise RepresentationContractError("Chronaris v2 checkpoint version mismatch")
    return payload


def v2_training_result(payload, best_path, last_path, *, status):
    return ChronarisV2TrainingResult(
        status=status,
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        best_epoch=int(payload["best_epoch"]),
        completed_epochs=int(payload["completed_epochs"]),
        stopped_early=bool(payload["stopped_early"]),
        best_public_selection_loss=float(payload["best_public_selection_loss"]),
        training_elapsed_s=float(payload["training_elapsed_s"]),
        epoch_rows=tuple(payload["epoch_rows"]),
        gradient_rows=tuple(payload["gradient_rows"]),
    )
