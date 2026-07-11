"""Public-pretext confirmation for already-selected encoder checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.candidate_screen import PUBLIC_SELECTION_WEIGHTS
from chronaris.modeling.training.common_pretraining import (
    load_common_pretraining_checkpoint,
)
from chronaris.modeling.training.pretext import CommonPretextWeights
from chronaris.representation import (
    AugmentationPolicy,
    DualStreamObservationBatch,
    apply_augmentation_realizations,
    build_batch_augmentation_realizations,
    build_common_pretext_targets,
    build_lag_discrimination_inputs,
    move_common_pretext_targets,
    select_observation_batch,
)
from chronaris.representation.contracts import RepresentationContractError


def confirm_selected_pretext_checkpoint(
    checkpoint_path: str | Path,
    *,
    batch: DualStreamObservationBatch,
    sample_ids: Sequence[str],
    batch_size: int,
    seed: int,
    device: str = "cpu",
    augmentation_policy: AugmentationPolicy | None = None,
) -> Mapping[str, object]:
    """Evaluate one frozen selection on a reserved model-visible profile."""

    ids = tuple(sample_ids)
    if not ids or batch_size <= 0:
        raise ValueError("candidate confirmation requires samples and positive batch size")
    policy = augmentation_policy or AugmentationPolicy()
    encoder, heads, normalizer, payload = load_common_pretraining_checkpoint(
        checkpoint_path,
        device=device,
    )
    encoder.eval()
    heads.eval()
    totals = {name: [0.0, 0] for name in PUBLIC_SELECTION_WEIGHTS}
    with torch.inference_mode():
        for offset in range(0, len(ids), batch_size):
            current_ids = ids[offset : offset + batch_size]
            raw = select_observation_batch(batch, current_ids)
            normalized = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                current_ids,
                epoch=0,
                global_seed=seed,
                policy=policy,
            )
            augmented = apply_augmentation_realizations(
                normalized,
                plans,
                policy=policy,
            )
            targets = build_common_pretext_targets(normalized, augmented)
            lag_inputs = build_lag_discrimination_inputs(
                augmented.batch,
                augmented.augmentation_ids,
            )
            positive_batch = move_observation_batch(augmented.batch, device=device)
            negative_batch = move_observation_batch(
                lag_inputs.negative_batch,
                device=device,
            )
            targets = move_common_pretext_targets(targets, device=device)
            output = heads(
                encoder(positive_batch).sequence_embedding,
                encoder(negative_batch).sequence_embedding,
                targets,
                weights=CommonPretextWeights(),
            )
            for term in output.terms:
                if term.raw_loss is not None and term.count > 0:
                    totals[term.term_name][0] += float(term.raw_loss) * term.count
                    totals[term.term_name][1] += term.count
    losses = {}
    for name, (loss_sum, count) in totals.items():
        if count <= 0:
            raise RepresentationContractError(
                f"selected candidate confirmation loss unavailable: {name}"
            )
        losses[name] = loss_sum / count
    return {
        "method_name": payload["method_name"],
        "candidate_id": payload["candidate_config"]["candidate_id"],
        "sample_count": len(ids),
        "sample_ids": list(ids),
        "masked_reconstruction": losses["masked_reconstruction"],
        "short_horizon_prediction": losses["short_horizon_prediction"],
        "lag_discrimination": losses["lag_discrimination"],
        "public_confirmation_loss": sum(
            PUBLIC_SELECTION_WEIGHTS[name] * losses[name]
            for name in PUBLIC_SELECTION_WEIGHTS
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_training_device": payload["config"].get("device", "legacy_cpu"),
        "confirmation_device": device,
        "augmentation_device": "cpu",
        "task_labels_opened": False,
        "simulation_ground_truth_opened": False,
    }
