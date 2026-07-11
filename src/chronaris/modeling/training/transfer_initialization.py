"""Schema-safe partial initialization from a completed task-agnostic checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class TransferSourceDescriptor:
    checkpoint_path: str
    checkpoint_sha256: str
    method_name: str
    seed: int
    source_physiology_feature_count: int
    source_vehicle_feature_count: int

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransferInitializationManifest:
    source: TransferSourceDescriptor
    copied_tensor_count: int
    skipped_shape_tensor_count: int
    missing_source_tensor_count: int
    copied_element_count: int
    target_element_count: int
    copied_element_fraction: float
    copied_tensor_names: tuple[str, ...]
    skipped_shape_tensor_names: tuple[str, ...]
    missing_source_tensor_names: tuple[str, ...]
    schema_specific_layers_reinitialized: bool

    def to_dict(self):
        payload = asdict(self)
        payload["source"] = self.source.to_dict()
        return payload


def describe_transfer_source(
    checkpoint_path: str | Path,
    *,
    expected_method: str,
    expected_seed: int | None = None,
) -> TransferSourceDescriptor:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "chronaris.common_pretraining_checkpoint.v1":
        raise ValueError("unsupported transfer source checkpoint")
    if payload.get("training_status") != "completed":
        raise ValueError("transfer source checkpoint is incomplete")
    if payload.get("label_used_for_encoder_training") is not False:
        raise ValueError("transfer source checkpoint used downstream labels")
    if str(payload.get("method_name")) != expected_method:
        raise ValueError("transfer source method mismatch")
    seed = int(payload["seed"])
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("transfer source seed mismatch")
    return TransferSourceDescriptor(
        checkpoint_path=str(path),
        checkpoint_sha256=sha256_file(path),
        method_name=expected_method,
        seed=seed,
        source_physiology_feature_count=len(payload["physiology_feature_names"]),
        source_vehicle_feature_count=len(payload["vehicle_feature_names"]),
    )


def initialize_encoder_from_transfer_source(
    encoder,
    checkpoint_path: str | Path,
    *,
    expected_method: str,
    expected_seed: int | None = None,
) -> TransferInitializationManifest:
    source = describe_transfer_source(
        checkpoint_path,
        expected_method=expected_method,
        expected_seed=expected_seed,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_state = payload["encoder_state_dict"]
    target_state = encoder.state_dict()
    copied = []
    skipped_shape = []
    missing = []
    copied_elements = 0
    for name, target in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            missing.append(name)
            continue
        if tuple(source_tensor.shape) != tuple(target.shape):
            skipped_shape.append(name)
            continue
        target_state[name] = source_tensor.to(
            device=target.device,
            dtype=target.dtype,
        )
        copied.append(name)
        copied_elements += target.numel()
    if not copied:
        raise ValueError("transfer source has no shape-compatible encoder tensors")
    encoder.load_state_dict(target_state, strict=True)
    total_elements = sum(value.numel() for value in target_state.values())
    return TransferInitializationManifest(
        source=source,
        copied_tensor_count=len(copied),
        skipped_shape_tensor_count=len(skipped_shape),
        missing_source_tensor_count=len(missing),
        copied_element_count=copied_elements,
        target_element_count=total_elements,
        copied_element_fraction=copied_elements / total_elements,
        copied_tensor_names=tuple(copied),
        skipped_shape_tensor_names=tuple(skipped_shape),
        missing_source_tensor_names=tuple(missing),
        schema_specific_layers_reinitialized=bool(skipped_shape or missing),
    )
