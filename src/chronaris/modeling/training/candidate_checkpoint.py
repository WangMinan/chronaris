"""Checkpoint and protocol helpers for candidate training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from chronaris.modeling.training.rng import (
    canonical_training_state_sha256,
    capture_rng_state,
)
from chronaris.representation.contracts import RepresentationContractError


def build_candidate_checkpoint_payload(**values):
    encoder = values.pop("encoder")
    heads = values.pop("heads")
    optimizer = values.pop("optimizer")
    rng_state = capture_rng_state()
    payload = {
        "format": "chronaris.common_pretraining_checkpoint.v2",
        "training_status": "running",
        "method_name": values["method_name"],
        "protocol_sha256": values["protocol_hash"],
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict": heads.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer": dict(values["normalizer"].to_manifest()),
        "fold": values["fold"].to_dict(),
        "config": asdict(values["config"]),
        "augmentation_policy": asdict(values["policy"]),
        "candidate_config": asdict(values["candidate"]),
        "physiology_feature_names": list(values["physiology_feature_names"]),
        "vehicle_feature_names": list(values["vehicle_feature_names"]),
        "vehicle_field_labels": [list(value) for value in values["vehicle_field_labels"]],
        "encoder_manifest": dict(encoder.config_manifest()),
        "seed": values["config"].seed,
        "epoch": values["completed_epochs"],
        "best_epoch": values["best_epoch"],
        "completed_epochs": values["completed_epochs"],
        "best_validation_losses": dict(values["best_losses"]),
        "best_public_selection_loss": values["best_score"],
        "stopped_early": values["stopped_early"],
        "step_count": values["step_count"],
        "training_elapsed_s": values["elapsed"],
        "parameter_count": encoder.parameter_count,
        "head_parameter_count": sum(p.numel() for p in heads.parameters()),
        "epoch_rows": list(values["epoch_rows"]),
        "training_rows": list(values["training_rows"]),
        "augmentation_rows": list(values["augmentation_rows"]),
        "label_used_for_encoder_training": False,
        "simulation_oracle_opened": False,
        "selection_uses_public_pretext_only": True,
        "selection_weights": dict(values["selection_weights"]),
        "transfer_source": values.get("transfer_source"),
        "transfer_initialization": values.get("transfer_initialization"),
        "training_device_history": list(values["device_history"]),
        "augmentation_device": "cpu",
        "chronaris_fusion_kind": values["chronaris_fusion_kind"],
        "chronaris_variant": values["chronaris_variant"],
        "chronaris_lag_aware_weight": values["chronaris_lag_aware_weight"],
        "chronaris_mechanism_enabled": values["chronaris_mechanism_enabled"],
        "chronaris_auxiliary_enabled": values["chronaris_mechanism_enabled"],
        "chronaris_explicit_shift_enabled": values[
            "chronaris_explicit_shift_enabled"
        ],
        "chronaris_explicit_shift_weight": values["chronaris_explicit_shift_weight"],
        "chronaris_event_pair_weight": values["chronaris_event_pair_weight"],
        "early_stopping_uses_public_pretext_only": True,
        "rng_state": rng_state,
    }
    shift_head = values.get("explicit_time_shift_head")
    payload["explicit_time_shift_head_state_dict"] = (
        shift_head.state_dict() if shift_head is not None else None
    )
    payload["canonical_training_state_sha256"] = canonical_training_state_sha256(
        payload["encoder_state_dict"],
        payload["head_state_dict"],
        payload["explicit_time_shift_head_state_dict"],
        payload["optimizer_state_dict"],
        rng_state,
    )
    return payload


def candidate_protocol_hash(**payload) -> str:
    digest = hashlib.sha256()
    for name in ("candidate_screen.py", "candidate_mechanisms.py", "pretext.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    payload["code_sha256"] = digest.hexdigest()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def candidate_checkpoint_is_compatible(
    payload,
    *,
    candidate,
    config,
    policy,
    fold,
    normalizer,
    physiology_feature_names,
    vehicle_feature_names,
    vehicle_field_labels,
    transfer_source,
    chronaris_fusion_kind,
    chronaris_variant,
    chronaris_lag_aware_weight,
    chronaris_mechanism_enabled,
    chronaris_explicit_shift_enabled,
    chronaris_explicit_shift_weight,
    chronaris_event_pair_weight,
) -> bool:
    return all(
        (
            payload.get("candidate_config") == asdict(candidate),
            training_configs_match_ignoring_device(
                payload.get("config", {}), asdict(config)
            ),
            payload.get("augmentation_policy") == asdict(policy),
            payload.get("fold") == fold.to_dict(),
            payload.get("normalizer", {}).get("transform_sha256")
            == normalizer.to_manifest().get("transform_sha256"),
            payload.get("physiology_feature_names") == list(physiology_feature_names),
            payload.get("vehicle_feature_names") == list(vehicle_feature_names),
            payload.get("vehicle_field_labels")
            == [list(value) for value in vehicle_field_labels],
            payload.get("label_used_for_encoder_training") is False,
            payload.get("simulation_oracle_opened") is False,
            payload.get("transfer_source") == transfer_source,
            payload.get("chronaris_fusion_kind", "multiscale") == chronaris_fusion_kind,
            payload.get("chronaris_variant", "full") == chronaris_variant,
            float(payload.get("chronaris_lag_aware_weight", 0.0))
            == chronaris_lag_aware_weight,
            bool(payload.get("chronaris_mechanism_enabled", False))
            == chronaris_mechanism_enabled,
            bool(payload.get("chronaris_explicit_shift_enabled", False))
            == chronaris_explicit_shift_enabled,
            float(payload.get("chronaris_explicit_shift_weight", 0.0))
            == chronaris_explicit_shift_weight,
            float(payload.get("chronaris_event_pair_weight", 0.0))
            == chronaris_event_pair_weight,
        )
    )


def load_candidate_payload(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") not in {
        "chronaris.common_pretraining_checkpoint.v1",
        "chronaris.common_pretraining_checkpoint.v2",
    }:
        raise RepresentationContractError("unsupported candidate screen checkpoint")
    return payload


def atomic_save_candidate(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_configs_match_ignoring_device(stored, expected) -> bool:
    stored_values = dict(stored)
    expected_values = dict(expected)
    stored_values.pop("device", None)
    expected_values.pop("device", None)
    return stored_values == expected_values
