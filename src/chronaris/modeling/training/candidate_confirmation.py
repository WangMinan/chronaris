"""Public-pretext confirmation for already-selected encoder checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from chronaris.modeling.training.candidate_validation import _evaluate_public_losses, _public_selection_loss
from chronaris.modeling.training.common_pretraining import load_common_pretraining_checkpoint
from chronaris.representation import AugmentationPolicy, DualStreamObservationBatch
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
    if not set(ids) <= set(payload["fold"]["held_out_sample_ids"]):
        raise RepresentationContractError("pretext confirmation crossed the reserved role")
    losses = _evaluate_public_losses(encoder=encoder, heads=heads, batch=batch, batch_provider=None,
        sample_ids=ids, batch_size=batch_size, normalizer=normalizer, policy=policy, seed=seed, device=device)
    return {
        "method_name": payload["method_name"],
        "candidate_id": payload["candidate_config"]["candidate_id"],
        "sample_count": len(ids),
        "sample_ids": list(ids),
        "masked_reconstruction": losses.get("masked_reconstruction"),
        "short_horizon_prediction": losses.get("short_horizon_prediction"),
        "lag_discrimination": losses.get("lag_discrimination"),
        "public_confirmation_loss": _public_selection_loss(losses),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_training_device": payload["config"].get("device", "legacy_cpu"),
        "confirmation_device": device,
        "augmentation_device": "cpu",
        "task_labels_opened": False,
        "simulation_ground_truth_opened": False,
    }
