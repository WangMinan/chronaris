"""Public-pretext validation, independent of optimizer and checkpoint state."""
import torch
from typing import Mapping
from chronaris.modeling.training.candidate_step import public_pretext_forward
from chronaris.modeling.training.candidate_mechanisms import _load_batch, _batch_ids
from chronaris.representation import (build_batch_augmentation_realizations,
    apply_augmentation_realizations)
from chronaris.representation.contracts import RepresentationContractError

PUBLIC_SELECTION_WEIGHTS = {
    "masked_reconstruction": 0.50, "short_horizon_prediction": 0.25, "lag_discrimination": 0.25,
}

def _evaluate_public_losses(
    *, encoder, heads, batch, batch_provider, sample_ids, batch_size, normalizer, policy, seed, device
) -> Mapping[str, float]:
    encoder.eval()
    heads.eval()
    totals = _empty_loss_totals()
    with torch.inference_mode():
        for ids in _batch_ids(sample_ids, batch_size):
            raw = _load_batch(batch, batch_provider, ids)
            normalized = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                ids, epoch=0, global_seed=seed, policy=policy,
                context_duration_s=normalized.context_durations_s.tolist(),
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
            output, _positive, _negative = public_pretext_forward(
                encoder=encoder, heads=heads, normalized=normalized, augmented=augmented, device=device)
            _accumulate_loss_terms(totals, output.terms)
    return _finalize_loss_totals(totals, allow_unavailable=heads.modality_feature_counts is not None)


def _empty_loss_totals() -> dict[str, list[float]]:
    return {name: [0.0, 0.0] for name in PUBLIC_SELECTION_WEIGHTS}


def _accumulate_loss_terms(totals, terms) -> None:
    for term in terms:
        components = term.components or (term,)
        for component in components:
            key = f"{term.term_name}/{component.term_name}" if term.components else term.term_name
            entry = totals.setdefault(key, [0., 0.])
            if component.raw_loss is not None and component.count > 0:
                entry[0] += float(component.raw_loss.detach()) * component.count
                entry[1] += component.count


def _finalize_loss_totals(totals, *, allow_unavailable=False) -> Mapping[str, float]:
    losses = {}
    for name in PUBLIC_SELECTION_WEIGHTS:
        components = [value for key, value in totals.items() if key.startswith(name + "/")]
        means = [total / count for total, count in (components or [totals[name]]) if count > 0]
        if means:
            losses[name] = sum(means) / len(means)
        elif not allow_unavailable:
            raise RepresentationContractError(f"candidate screen loss unavailable: {name}")
    return losses


def _public_selection_loss(losses: Mapping[str, float]) -> float:
    if not losses:
        raise RepresentationContractError("candidate screen has no applicable validation objectives")
    # Only select updates within the same method/objectives; task metrics rank v4 methods.
    available_weight = sum(PUBLIC_SELECTION_WEIGHTS[name] for name in losses)
    return sum(PUBLIC_SELECTION_WEIGHTS[name] * value for name, value in losses.items()) / available_weight
