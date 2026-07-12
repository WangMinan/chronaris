"""Teacher-target and initialization contracts for Chronaris v2 repair."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import torch

from chronaris.modeling.training.chronaris_v2_checkpointing import load_v2_checkpoint
from chronaris.modeling.training.chronaris_v2_objectives import (
    ChronarisV2ObjectiveHeads,
)
from chronaris.modeling.training.pretext import CommonPretextHeadBundle
from chronaris.modeling.training.pretraining_encoders import TrainableFusionEncoder
from chronaris.representation import FusionStreamBatch, TrainOnlyRobustNormalizer
from chronaris.representation.contracts import RepresentationContractError


def physiology_teacher_manifest(
    targets: FusionStreamBatch | None,
) -> Mapping[str, object] | None:
    if targets is None:
        return None
    digest = hashlib.sha256()
    digest.update("\0".join(targets.sample_ids).encode())
    digest.update(
        targets.sequence_embedding.detach().cpu().contiguous().numpy().tobytes()
    )
    digest.update(targets.valid_mask.detach().cpu().contiguous().numpy().tobytes())
    return {
        "method_name": targets.method_name,
        "fold_id": targets.fold_id,
        "checkpoint_sha256": targets.checkpoint_sha256,
        "target_sha256": digest.hexdigest(),
        "sample_count": len(targets.sample_ids),
        "query_count": int(targets.sequence_embedding.shape[1]),
        "representation_dim": int(targets.sequence_embedding.shape[2]),
        "task_labels_opened": False,
    }


def initialization_manifest(
    checkpoint: str | Path | None,
    *,
    normalizer: TrainOnlyRobustNormalizer,
) -> Mapping[str, object] | None:
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    payload = load_v2_checkpoint(path)
    if payload.get("training_status") != "completed":
        raise RepresentationContractError("v2 initialization checkpoint is incomplete")
    if any(
        bool(payload.get(key))
        for key in (
            "label_used_for_encoder_training",
            "simulation_oracle_opened",
            "locked_test_opened",
        )
    ):
        raise RepresentationContractError(
            "v2 initialization checkpoint contains forbidden evidence"
        )
    if payload.get("normalizer") != dict(normalizer.to_manifest()):
        raise RepresentationContractError(
            "v2 initialization normalizer differs from current train fold"
        )
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "source_protocol_sha256": str(payload["protocol_sha256"]),
        "source_candidate_id": str(payload["candidate_config"]["candidate_id"]),
        "task_labels_opened": False,
        "simulation_oracle_opened": False,
        "locked_test_opened": False,
    }


def initialize_v2_repair_candidate(
    checkpoint: str | Path,
    *,
    encoder: TrainableFusionEncoder,
    common_heads: CommonPretextHeadBundle,
    v2_heads: ChronarisV2ObjectiveHeads,
    device: str,
) -> None:
    payload = load_v2_checkpoint(Path(checkpoint), device=device)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    common_heads.load_state_dict(payload["common_head_state_dict"], strict=True)
    incompatible = v2_heads.load_state_dict(
        payload["v2_head_state_dict"],
        strict=False,
    )
    expected_missing = (
        {
            "physiology_teacher_projection.weight",
            "physiology_teacher_projection.bias",
        }
        if v2_heads.physiology_teacher_projection is not None
        else set()
    )
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RepresentationContractError(
            "v2 repair initialization changed non-teacher objective parameters"
        )


def select_physiology_teacher_targets(
    targets: FusionStreamBatch | None,
    sample_ids: Sequence[str],
    *,
    device: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if targets is None:
        return None, None
    index_by_id = {
        sample_id: index for index, sample_id in enumerate(targets.sample_ids)
    }
    missing = tuple(sample_id for sample_id in sample_ids if sample_id not in index_by_id)
    if missing:
        raise RepresentationContractError(
            f"physiology teacher targets are missing samples: {missing[:3]}"
        )
    indices = torch.tensor(
        [index_by_id[sample_id] for sample_id in sample_ids],
        dtype=torch.long,
        device=targets.sequence_embedding.device,
    )
    sequence = targets.sequence_embedding.index_select(0, indices).to(device)
    valid_mask = targets.valid_mask.index_select(0, indices).to(device)
    return sequence, valid_mask


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
