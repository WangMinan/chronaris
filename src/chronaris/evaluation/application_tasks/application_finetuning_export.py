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
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def export_finetuned_application_representations(
    *,
    model: EndToEndApplicationModel,
    checkpoint_path: str | Path,
    batch: DualStreamObservationBatch,
    role_sample_ids: Mapping[str, Sequence[str]],
    output_root: str | Path,
    batch_size: int = 128,
) -> Mapping[str, FusionStreamBatch]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != FINETUNING_FORMAT
        or payload.get("training_status") != "completed"
    ):
        raise RepresentationContractError(
            "fine-tuned representation checkpoint is incomplete"
        )
    if payload.get("label_used_for_encoder_training") is not True:
        raise RepresentationContractError("fine-tuned checkpoint must declare label use")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    checkpoint_hash = sha256_file(checkpoint_path)
    outputs = {}
    for role in ("train", "validation", "held_out"):
        chunks = []
        ids = tuple(role_sample_ids[role])
        for offset in range(0, len(ids), batch_size):
            raw = select_observation_batch(batch, ids[offset : offset + batch_size])
            with torch.inference_mode():
                sequence = model.encode(raw).detach().cpu()
            chunks.append((raw, sequence))
        sequence = torch.cat([value for _raw, value in chunks], dim=0)
        raw_ids = tuple(value for raw, _value in chunks for value in raw.sample_ids)
        source_hashes = tuple(
            value for raw, _value in chunks for value in raw.source_sample_hashes
        )
        timestamps = torch.cat(
            [raw.query_timestamps_s for raw, _value in chunks], dim=0
        )
        output = FusionStreamBatch(
            sample_ids=raw_ids,
            timestamps_s=timestamps,
            sequence_embedding=sequence,
            valid_mask=torch.ones(sequence.shape[:2], dtype=torch.bool),
            pooled_embedding=sequence.mean(dim=1),
            method_name=model.method_name,
            fold_id=f"simulation_g1_to_g2_end_to_end__seed_{payload['seed']}",
            checkpoint_sha256=checkpoint_hash,
            source_sample_hashes=source_hashes,
        )
        write_fusion_stream_batch(
            output,
            root=Path(output_root) / model.method_name / role,
            export_role=f"end_to_end_{role}",
            label_used_for_encoder_training=True,
        )
        outputs[role] = output
    return outputs
