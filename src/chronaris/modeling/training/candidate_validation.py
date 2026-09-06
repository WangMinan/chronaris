"""Public-pretext validation, independent of optimizer and checkpoint state."""
import torch
from typing import Mapping
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.pretext import CommonPretextWeights
from chronaris.representation import (build_batch_augmentation_realizations,
    apply_augmentation_realizations, build_common_pretext_targets,
    build_lag_discrimination_inputs, move_common_pretext_targets, select_observation_batch)
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
            targets = build_common_pretext_targets(normalized, augmented)
            lag_inputs = build_lag_discrimination_inputs(
                augmented.batch, augmented.augmentation_ids
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
            _accumulate_loss_terms(totals, output.terms)
    return _finalize_loss_totals(totals)


def _empty_loss_totals() -> dict[str, list[float]]:
    return {name: [0.0, 0.0] for name in PUBLIC_SELECTION_WEIGHTS}


def _accumulate_loss_terms(totals, terms) -> None:
    for term in terms:
        if term.raw_loss is not None and term.count > 0:
            totals[term.term_name][0] += float(term.raw_loss.detach()) * term.count
            totals[term.term_name][1] += term.count


def _finalize_loss_totals(totals) -> Mapping[str, float]:
    losses = {}
    for name, (loss_sum, count) in totals.items():
        if count <= 0:
            raise RepresentationContractError(f"candidate screen loss unavailable: {name}")
        losses[name] = loss_sum / count
    return losses


def _public_selection_loss(losses: Mapping[str, float]) -> float:
    return sum(PUBLIC_SELECTION_WEIGHTS[name] * losses[name] for name in PUBLIC_SELECTION_WEIGHTS)


def _load_batch(batch, provider, sample_ids):
    loaded = provider(sample_ids) if provider is not None else select_observation_batch(batch, sample_ids)
    if tuple(loaded.sample_ids) != tuple(sample_ids):
        raise RepresentationContractError("candidate screen batch provider changed sample order")
    return loaded


def _batch_ids(sample_ids, batch_size):
    values = tuple(sample_ids)
    return tuple(values[index : index + batch_size] for index in range(0, len(values), batch_size))
