"""Reuse frozen pretraining checkpoints on fixed-grid downstream contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from chronaris.modeling.fusion_encoders import (
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
)
from chronaris.modeling.training import (
    TRAINABLE_FUSION_METHODS,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import (
    CheckpointRegistry,
    load_fusion_stream_batch,
    select_observation_batch,
    validate_fusion_method_alignment,
    write_fusion_stream_batch,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


APPLICATION_METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)


def load_frozen_pretraining_adapters(
    *,
    pretraining_heavy_root: str | Path,
    pretraining_compact_root: str | Path,
):
    heavy_root = Path(pretraining_heavy_root)
    registry = CheckpointRegistry(
        Path(pretraining_compact_root) / "checkpoint_registry.json"
    )
    fold_ids = {record.fold.fold_id for record in registry.records.values()}
    if len(fold_ids) != 1 or len(registry.records) != 6:
        raise ValueError("pretraining registry must contain one complete six-method fold")
    fold_id = next(iter(fold_ids))
    adapters = {}
    checkpoint_rows = []
    for method_name in TRAINABLE_FUSION_METHODS:
        record = registry.require(method_name, fold_id)
        expected_path = heavy_root / "checkpoints" / method_name / "best.pt"
        if Path(record.checkpoint_path).resolve() != expected_path.resolve():
            raise ValueError("pretraining registry checkpoint path mismatch")
        encoder, _heads, normalizer, payload = load_common_pretraining_checkpoint(
            expected_path
        )
        adapters[method_name] = TrainedFusionAdapter(
            encoder=encoder,
            normalizer=normalizer,
            fold_id=fold_id,
            checkpoint_sha256=record.checkpoint_sha256,
        )
        checkpoint_rows.append(
            {
                "method_name": method_name,
                "checkpoint_path": str(expected_path),
                "checkpoint_sha256": record.checkpoint_sha256,
                "fit_sample_hash": record.fit_sample_hash,
                "training_status": payload["training_status"],
                "label_used_for_encoder_training": False,
            }
        )
    naive_record = registry.require("naive_time_sync", fold_id)
    naive_path = heavy_root / "checkpoints" / "naive_time_sync" / "best.pt"
    if sha256_file(naive_path) != naive_record.checkpoint_sha256:
        raise ValueError("naive time-sync checkpoint hash mismatch")
    adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
        encoder=load_naive_time_sync_checkpoint(naive_path),
        fold_id=fold_id,
        checkpoint_sha256=naive_record.checkpoint_sha256,
    )
    checkpoint_rows.append(
        {
            "method_name": "naive_time_sync",
            "checkpoint_path": str(naive_path),
            "checkpoint_sha256": naive_record.checkpoint_sha256,
            "fit_sample_hash": naive_record.fit_sample_hash,
            "training_status": "unsupervised_transform_only",
            "label_used_for_encoder_training": False,
        }
    )
    return adapters, tuple(checkpoint_rows), fold_id


def export_application_context_representations(
    *,
    adapters,
    batch,
    role_sample_ids: Mapping[str, tuple[str, ...]],
    output_root: str | Path,
    resume: bool,
    batch_size: int | None = None,
    require_valid_mask_match: bool = True,
):
    if batch_size is not None and batch_size <= 0:
        raise ValueError("application representation batch size must be positive")
    root = Path(output_root)
    outputs = {method: {} for method in APPLICATION_METHODS}
    rows = []
    for method_name in APPLICATION_METHODS:
        adapter = adapters[method_name]
        for role in ("train", "validation", "held_out"):
            role_batch = select_observation_batch(batch, role_sample_ids[role])
            destination = root / method_name / role
            status = "completed"
            output = None
            if resume and (destination / "fusion_stream.npz").exists() and (
                destination / "representation_manifest.json"
            ).exists():
                try:
                    candidate = load_fusion_stream_batch(destination)
                except (OSError, ValueError):
                    candidate = None
                if candidate is not None and (
                    candidate.method_name == method_name
                    and candidate.sample_ids == role_batch.sample_ids
                    and candidate.checkpoint_sha256 == adapter.checkpoint_sha256
                ):
                    output = candidate
                    status = "resumed"
            if output is None:
                output = _encode_in_batches(
                    adapter,
                    role_batch,
                    batch_size=batch_size,
                )
                write_fusion_stream_batch(
                    output,
                    root=destination,
                    export_role=f"application_{role}",
                )
                output = load_fusion_stream_batch(destination)
            outputs[method_name][role] = output
            rows.append(
                {
                    "method_name": method_name,
                    "role": role,
                    "status": status,
                    "sample_count": len(output.sample_ids),
                    "checkpoint_sha256": output.checkpoint_sha256,
                    "representation_sha256": sha256_file(
                        destination / "fusion_stream.npz"
                    ),
                    "output_root": str(destination),
                }
            )
    alignment_hashes = {
        role: validate_fusion_method_alignment(
            [outputs[method][role] for method in APPLICATION_METHODS],
            require_valid_mask_match=require_valid_mask_match,
        )
        for role in ("train", "validation", "held_out")
    }
    return outputs, rows, alignment_hashes


def _encode_in_batches(adapter, batch, *, batch_size):
    if batch_size is None or len(batch.sample_ids) <= batch_size:
        return adapter(batch)
    outputs = []
    for offset in range(0, len(batch.sample_ids), batch_size):
        ids = batch.sample_ids[offset : offset + batch_size]
        outputs.append(adapter(select_observation_batch(batch, ids)))
    first = outputs[0]
    if any(
        output.method_name != first.method_name
        or output.fold_id != first.fold_id
        or output.checkpoint_sha256 != first.checkpoint_sha256
        for output in outputs[1:]
    ):
        raise ValueError("batched application representation lineage changed")
    return type(first)(
        sample_ids=tuple(value for output in outputs for value in output.sample_ids),
        timestamps_s=torch.cat([output.timestamps_s for output in outputs], dim=0),
        sequence_embedding=torch.cat(
            [output.sequence_embedding for output in outputs], dim=0
        ),
        valid_mask=torch.cat([output.valid_mask for output in outputs], dim=0),
        pooled_embedding=torch.cat(
            [output.pooled_embedding for output in outputs], dim=0
        ),
        method_name=first.method_name,
        fold_id=first.fold_id,
        checkpoint_sha256=first.checkpoint_sha256,
        source_sample_hashes=tuple(
            value for output in outputs for value in output.source_sample_hashes
        ),
    )
