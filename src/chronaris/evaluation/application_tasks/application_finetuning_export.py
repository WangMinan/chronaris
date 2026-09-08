"""Label-aware representation export for the auxiliary fine-tuning family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from chronaris.evaluation.application_tasks.application_finetuning import (
    FINETUNING_FORMAT,
    EndToEndApplicationModel,
)
from chronaris.representation import (
    DualStreamObservationBatch,
    FusionStreamBatch,
    select_observation_batch,
    write_fusion_stream_batch,
    load_fusion_stream_batch,
)
from chronaris.representation.contracts import RepresentationContractError
from chronaris.modeling.training.candidate_validation import _load_batch
from chronaris.modeling.training.candidate_checkpoint import is_development_snapshot
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def load_frozen_application_encoder(checkpoint, *, route, fold, device, allow_diagnostic_snapshot=False):
    """Load only encoder weights while retaining label and initialization lineage."""
    from chronaris.modeling.training import load_common_pretraining_checkpoint
    if route not in {"self_supervised", "task_guided"}:
        raise ValueError("unknown frozen encoder route")
    if route == "self_supervised":
        encoder, _, normalizer, payload = load_common_pretraining_checkpoint(checkpoint, device=device,
            allow_diagnostic_snapshot=allow_diagnostic_snapshot)
        if payload["fold"] != fold.to_dict():
            raise RepresentationContractError("frozen encoder data roles changed")
        return encoder, normalizer, payload
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    snapshot = allow_diagnostic_snapshot and is_development_snapshot(payload)
    if (payload.get("format") != FINETUNING_FORMAT or payload.get("label_used_for_encoder_training") is not True
        or (payload.get("training_status") != "completed" and not snapshot)):
        raise RepresentationContractError("task-guided encoder requires a completed supervised checkpoint")
    roles = {role: list(getattr(fold, role + "_sample_ids")) for role in ("train", "validation", "held_out")}
    if payload["role_sample_ids"] != roles or payload["fold_id"] != fold.fold_id:
        raise RepresentationContractError("task-guided encoder data roles changed")
    if sha256_file(payload["source_checkpoint_path"]) != payload["source_checkpoint_sha256"]:
        raise RepresentationContractError("task-guided initialization checkpoint changed")
    encoder, _, normalizer, source = load_common_pretraining_checkpoint(payload["source_checkpoint_path"], device=device)
    if (source["fold"] != fold.to_dict() or source["seed"] != payload["seed"]
        or encoder.method_name != payload["method_name"] or payload["normalizer"] != normalizer.to_manifest()):
        raise RepresentationContractError("task-guided encoder initialization lineage changed")
    encoder.load_state_dict({name.removeprefix("encoder."): value for name, value in payload["model_state_dict"].items()
                             if name.startswith("encoder.")}, strict=True)
    return encoder, normalizer, payload


def export_loaded_application_encoder(*, encoder, normalizer, checkpoint, provider, fold, root,
                                      export_roles, export_prefix, label_used_for_encoder_training=False,
                                      batch_size=4):
    """Shared masked export for an already validated frozen encoder and role manifest."""
    from chronaris.modeling.training import TrainedFusionAdapter
    from chronaris.representation.oof_export import _concatenate_fusion_batches
    if batch_size < 1 or not export_roles or len(set(export_roles)) != len(export_roles) or set(export_roles) - {"train", "validation", "held_out"}:
        raise ValueError("invalid frozen export batch size or roles")
    from chronaris.modeling.fusion_encoders import NaiveTimeSyncEncoder, NaiveTimeSyncFusionAdapter
    adapter = (NaiveTimeSyncFusionAdapter(encoder=encoder, fold_id=fold.fold_id, checkpoint_sha256=sha256_file(checkpoint))
        if isinstance(encoder, NaiveTimeSyncEncoder) else TrainedFusionAdapter(encoder=encoder, normalizer=normalizer,
            fold_id=fold.fold_id, checkpoint_sha256=sha256_file(checkpoint)))
    outputs = {}
    for role in export_roles:
        ids = getattr(fold, role + "_sample_ids")
        if not ids:
            continue
        path = Path(root) / role
        manifest_path = path / "representation_manifest.json"
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text())
            output = load_fusion_stream_batch(path)
            if (output.sample_ids != ids or output.checkpoint_sha256 != adapter.checkpoint_sha256
                or output.fold_id != fold.fold_id or output.method_name != encoder.method_name
                or metadata["label_used_for_encoder_training"] is not label_used_for_encoder_training
                or metadata["export_role"] != f"{export_prefix}_{role}"):
                raise RepresentationContractError("frozen representation provenance changed")
        else:
            output = _concatenate_fusion_batches([adapter(provider(ids[i:i + batch_size])) for i in range(0, len(ids), batch_size)])
            if output.sample_ids != ids:
                raise RepresentationContractError("frozen export provider returned other samples")
            write_fusion_stream_batch(output, root=path, export_role=f"{export_prefix}_{role}",
                                     label_used_for_encoder_training=label_used_for_encoder_training)
        outputs[role] = output
    return outputs


def export_finetuned_application_representations(
    *,
    model: EndToEndApplicationModel,
    checkpoint_path: str | Path,
    batch: DualStreamObservationBatch,
    role_sample_ids: Mapping[str, Sequence[str]],
    output_root: str | Path,
    batch_size: int = 128,
    batch_provider=None,
    export_roles: tuple[str, ...] = ("train", "validation", "held_out"),
    allow_diagnostic_snapshot: bool = False,
) -> Mapping[str, FusionStreamBatch]:
    if (batch is None) == (batch_provider is None):
        raise ValueError("fine-tuned export requires exactly one observation source")
    if not export_roles or len(set(export_roles)) != len(export_roles) or not set(export_roles) <= {"train", "validation", "held_out"}:
        raise ValueError("invalid fine-tuned export roles")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    snapshot = allow_diagnostic_snapshot and is_development_snapshot(payload)
    if snapshot and not set(export_roles) <= {"train", "validation"}:
        raise RepresentationContractError("diagnostic snapshots cannot export confirmation roles")
    if (
        payload.get("format") != FINETUNING_FORMAT
        or (payload.get("training_status") != "completed" and not snapshot)
    ):
        raise RepresentationContractError(
            "fine-tuned representation checkpoint is incomplete"
        )
    expected_labels = model.encoder is not None
    if payload.get("label_used_for_encoder_training") is not expected_labels:
        raise RepresentationContractError("fine-tuned checkpoint label provenance differs from encoder kind")
    expected_roles = {key: list(value) for key, value in role_sample_ids.items()}
    if payload.get("role_sample_ids") != expected_roles:
        raise RepresentationContractError("fine-tuned export role lineage changed")
    if model.normalizer is not None and model.normalizer.to_manifest() != payload.get("normalizer"):
        raise RepresentationContractError("fine-tuned export normalization changed")
    if model.encoder is not None:
        model.encoder.load_state_dict({name.removeprefix("encoder."): value
            for name, value in payload["model_state_dict"].items() if name.startswith("encoder.")}, strict=True)
    model.eval()
    checkpoint_hash = sha256_file(checkpoint_path)
    outputs = {}
    for role in export_roles:
        chunks = []
        ids = tuple(role_sample_ids[role])
        if not ids:
            continue
        for offset in range(0, len(ids), batch_size):
            raw = _load_batch(batch, batch_provider, ids[offset : offset + batch_size])
            with torch.inference_mode():
                sequence, valid = model.encode_with_mask(raw)
                sequence = sequence.masked_fill(~valid.unsqueeze(-1), 0).detach().cpu()
                valid = valid.detach().cpu()
            chunks.append((raw, sequence, valid))
        sequence = torch.cat([value for _raw, value, _mask in chunks], dim=0)
        valid = torch.cat([mask for _raw, _value, mask in chunks], dim=0)
        raw_ids = tuple(value for raw, _value, _mask in chunks for value in raw.sample_ids)
        source_hashes = tuple(
            value for raw, _value, _mask in chunks for value in raw.source_sample_hashes
        )
        timestamps = torch.cat(
            [raw.query_timestamps_s for raw, _value, _mask in chunks], dim=0
        )
        output = FusionStreamBatch(
            sample_ids=raw_ids,
            timestamps_s=timestamps,
            sequence_embedding=sequence,
            valid_mask=valid,
            pooled_embedding=sequence.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1),
            method_name=model.method_name,
            fold_id=payload["fold_id"],
            checkpoint_sha256=checkpoint_hash,
            source_sample_hashes=source_hashes,
        )
        write_fusion_stream_batch(
            output,
            root=Path(output_root) / model.method_name / role,
            export_role=f"end_to_end_{role}",
            label_used_for_encoder_training=expected_labels,
        )
        outputs[role] = output
    return outputs
