"""Label-aware representation export for the auxiliary fine-tuning family."""

from __future__ import annotations

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
)
from chronaris.representation.contracts import RepresentationContractError
from chronaris.modeling.training.candidate_validation import _load_batch
from chronaris.modeling.training.candidate_checkpoint import is_development_snapshot
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


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
