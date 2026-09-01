"""Mechanism objectives and validation for the canonical candidate trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.chronaris_auxiliary import (
    build_chronaris_auxiliary_losses,
    chronaris_auxiliary_losses_to_rows,
    lag_aware_alignment_loss,
)
from chronaris.modeling.training.pretext import (
    chronaris_auxiliary_weight_schedule,
    event_pair_contrastive_loss_term,
    explicit_time_shift_loss_term,
    pretext_loss_terms_to_rows,
)
from chronaris.representation import (
    apply_augmentation_realizations,
    build_batch_augmentation_realizations,
    build_explicit_time_shift_inputs,
    build_lag_discrimination_inputs,
    select_observation_batch,
)
from chronaris.representation.contracts import RepresentationContractError


@dataclass(frozen=True, slots=True)
class CandidateMechanismStep:
    additional_loss: torch.Tensor
    rows: tuple[Mapping[str, object], ...]
    metrics_by_term: Mapping[str, Mapping[str, object]]


def build_candidate_mechanism_step(
    *,
    encoder,
    shift_head,
    positive,
    negative,
    augmented,
    group_ids: Sequence[str],
    epoch: int,
    device: str,
    mechanism_enabled: bool,
    lag_aware_weight: float,
    explicit_shift_weight: float,
    event_pair_weight: float,
) -> CandidateMechanismStep:
    zero = positive.sequence_embedding.sum() * 0.0
    additional_loss = zero
    rows: list[Mapping[str, object]] = []
    metrics: dict[str, Mapping[str, object]] = {}
    warmup_fraction = min(epoch / 5.0, 1.0)
    if mechanism_enabled:
        weights = chronaris_auxiliary_weight_schedule(epoch)
        mechanism = build_chronaris_auxiliary_losses(
            positive,
            negative,
            weights=weights,
        )
        additional_loss = additional_loss + mechanism.total_loss
        rows.extend(chronaris_auxiliary_losses_to_rows(mechanism, weights=weights))
    if lag_aware_weight > 0:
        alignment = positive.auxiliary.get("alignment_output")
        if alignment is None:
            raise RepresentationContractError(
                "enabled lag-aware loss requires Chronaris alignment output"
            )
        result = lag_aware_alignment_loss(
            alignment,
            min_lag_s=0.0,
            max_lag_s=15.0,
        )
        weight = lag_aware_weight * warmup_fraction
        additional_loss = additional_loss + weight * result.loss
        rows.append(
            {
                "term_name": "lag_aware_alignment",
                "weight": weight,
                "status": "active" if result.count else "unavailable",
                "count": result.count,
                "raw_loss": float(result.loss.detach().cpu()),
                "weighted_loss": float((weight * result.loss).detach().cpu()),
                "reason": None if result.count else "no_valid_causal_lag",
            }
        )
    if shift_head is not None:
        inputs = build_explicit_time_shift_inputs(
            augmented.batch,
            augmented.augmentation_ids,
        )
        shifted = encoder(
            move_observation_batch(inputs.shifted_batch, device=device)
        )
        logits = shift_head(
            shifted.sequence_embedding,
            shifted.modality_available_mask,
        )
        term = explicit_time_shift_loss_term(
            logits,
            inputs.class_indices,
            weight=explicit_shift_weight * warmup_fraction,
        )
        additional_loss = additional_loss + term.weighted_loss
        rows.extend(pretext_loss_terms_to_rows((term,)))
        metrics[term.term_name] = {
            "accuracy": float(
                (
                    logits.argmax(dim=-1)
                    == inputs.class_indices.to(logits.device)
                )
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "shifts_s": [float(value) for value in inputs.shifts_s],
        }
    if event_pair_weight > 0:
        semantic_output = positive.auxiliary.get("semantic_event_output")
        if semantic_output is None:
            raise RepresentationContractError(
                "event-pair objective requires semantic event output"
            )
        term, pair_metrics = event_pair_contrastive_loss_term(
            semantic_output,
            group_ids,
            temperature=0.1,
            weight=event_pair_weight * warmup_fraction,
        )
        if term.weighted_loss is not None:
            additional_loss = additional_loss + term.weighted_loss
        rows.extend(pretext_loss_terms_to_rows((term,)))
        metrics[term.term_name] = pair_metrics
    return CandidateMechanismStep(
        additional_loss=additional_loss,
        rows=tuple(rows),
        metrics_by_term=metrics,
    )


def evaluate_candidate_mechanisms(
    *,
    encoder,
    shift_head,
    batch,
    batch_provider,
    sample_ids,
    batch_size,
    normalizer,
    policy,
    seed,
    device,
    mechanism_enabled,
    lag_aware_weight,
    explicit_shift_weight,
    event_pair_weight,
) -> Mapping[str, object]:
    if not any(
        (
            mechanism_enabled,
            lag_aware_weight > 0,
            shift_head is not None,
            event_pair_weight > 0,
        )
    ):
        return {"weighted_total": 0.0, "terms": []}
    encoder.eval()
    if shift_head is not None:
        shift_head.eval()
    batches = (
        interleaved_group_batch_ids(
            batch,
            batch_provider,
            sample_ids,
            batch_size,
        )
        if event_pair_weight > 0
        else _batch_ids(sample_ids, batch_size)
    )
    totals: dict[str, dict[str, object]] = {}
    shift_correct = 0
    shift_count = 0
    pair_sums = {
        "positive_similarity": 0.0,
        "negative_similarity": 0.0,
        "recall_at_1": 0.0,
        "valid_pair_count": 0,
    }
    with torch.inference_mode():
        for ids in batches:
            raw = _load_batch(batch, batch_provider, ids)
            normalized = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                ids,
                epoch=0,
                global_seed=seed,
                policy=policy,
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
            positive = encoder(
                move_observation_batch(augmented.batch, device=device),
                compute_chronaris_diagnostics=(mechanism_enabled or lag_aware_weight > 0),
            )
            negative = None
            if mechanism_enabled:
                lag_inputs = build_lag_discrimination_inputs(
                    augmented.batch,
                    augmented.augmentation_ids,
                )
                negative = encoder(
                    move_observation_batch(lag_inputs.negative_batch, device=device),
                    compute_chronaris_diagnostics=True,
                )
                weights = chronaris_auxiliary_weight_schedule(5)
                mechanism = build_chronaris_auxiliary_losses(
                    positive,
                    negative,
                    weights=weights,
                )
                for row in chronaris_auxiliary_losses_to_rows(
                    mechanism,
                    weights=weights,
                ):
                    _accumulate_row(totals, row)
            if lag_aware_weight > 0:
                result = lag_aware_alignment_loss(
                    positive.auxiliary["alignment_output"],
                    min_lag_s=0.0,
                    max_lag_s=15.0,
                )
                _accumulate_row(
                    totals,
                    {
                        "term_name": "lag_aware_alignment",
                        "weight": lag_aware_weight,
                        "count": result.count,
                        "raw_loss": float(result.loss.detach().cpu()),
                        "weighted_loss": float(
                            (lag_aware_weight * result.loss).detach().cpu()
                        ),
                        "reason": None if result.count else "no_valid_causal_lag",
                    },
                )
            if shift_head is not None:
                inputs = build_explicit_time_shift_inputs(
                    augmented.batch,
                    augmented.augmentation_ids,
                )
                shifted = encoder(
                    move_observation_batch(inputs.shifted_batch, device=device)
                )
                logits = shift_head(
                    shifted.sequence_embedding,
                    shifted.modality_available_mask,
                )
                term = explicit_time_shift_loss_term(
                    logits,
                    inputs.class_indices,
                    weight=explicit_shift_weight,
                )
                _accumulate_row(totals, pretext_loss_terms_to_rows((term,))[0])
                labels = inputs.class_indices.to(logits.device)
                shift_correct += int((logits.argmax(dim=-1) == labels).sum().item())
                shift_count += len(labels)
            if event_pair_weight > 0:
                term, metrics = event_pair_contrastive_loss_term(
                    positive.auxiliary["semantic_event_output"],
                    raw.group_ids,
                    temperature=0.1,
                    weight=event_pair_weight,
                )
                _accumulate_row(totals, pretext_loss_terms_to_rows((term,))[0])
                count = int(metrics["valid_pair_count"] or 0)
                if count:
                    for name in (
                        "positive_similarity",
                        "negative_similarity",
                        "recall_at_1",
                    ):
                        pair_sums[name] += float(metrics[name]) * count
                    pair_sums["valid_pair_count"] += count
    terms = _finalize_rows(totals)
    pair_count = int(pair_sums["valid_pair_count"])
    return {
        "weighted_total": sum(
            float(row["weighted_loss"])
            for row in terms
            if row["weighted_loss"] is not None
        ),
        "terms": terms,
        "explicit_time_shift_accuracy": shift_correct / shift_count if shift_count else None,
        "event_pair_positive_similarity": (
            pair_sums["positive_similarity"] / pair_count if pair_count else None
        ),
        "event_pair_negative_similarity": (
            pair_sums["negative_similarity"] / pair_count if pair_count else None
        ),
        "event_pair_recall_at_1": (
            pair_sums["recall_at_1"] / pair_count if pair_count else None
        ),
        "event_pair_count": pair_count,
    }


def interleaved_group_batch_ids(batch, provider, sample_ids, batch_size):
    groups_by_id: dict[str, str] = {}
    for ids in _batch_ids(sample_ids, batch_size):
        loaded = _load_batch(batch, provider, ids)
        groups_by_id.update(zip(loaded.sample_ids, loaded.group_ids, strict=True))
    buckets: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        buckets.setdefault(groups_by_id[sample_id], []).append(sample_id)
    interleaved: list[str] = []
    while any(buckets.values()):
        for group_id in sorted(buckets):
            if buckets[group_id]:
                interleaved.append(buckets[group_id].pop(0))
    return _batch_ids(interleaved, batch_size)


def parameter_gradient_norm(parameters) -> float:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter is not None and parameter.grad is not None
    ]
    return float(torch.stack(squared).sum().sqrt().cpu()) if squared else 0.0


def _load_batch(batch, provider, sample_ids):
    loaded = (
        provider(sample_ids)
        if provider is not None
        else select_observation_batch(batch, sample_ids)
    )
    if tuple(loaded.sample_ids) != tuple(sample_ids):
        raise RepresentationContractError("candidate batch provider changed sample order")
    return loaded


def _batch_ids(sample_ids, batch_size):
    values = tuple(sample_ids)
    return tuple(
        values[index : index + batch_size]
        for index in range(0, len(values), batch_size)
    )


def _accumulate_row(totals, row) -> None:
    term = totals.setdefault(
        row["term_name"],
        {
            "weight": row["weight"],
            "count": 0,
            "raw_sum": 0.0,
            "weighted_sum": 0.0,
            "reason": row.get("reason"),
        },
    )
    count = int(row["count"])
    if count and row.get("raw_loss") is not None:
        term["count"] += count
        term["raw_sum"] += float(row["raw_loss"]) * count
        term["weighted_sum"] += float(row["weighted_loss"]) * count


def _finalize_rows(totals) -> list[dict[str, object]]:
    rows = []
    for term_name, values in totals.items():
        count = int(values["count"])
        rows.append(
            {
                "term_name": term_name,
                "status": "active" if count else "unavailable",
                "count": count,
                "weight": float(values["weight"]),
                "raw_loss": float(values["raw_sum"]) / count if count else None,
                "weighted_loss": (
                    float(values["weighted_sum"]) / count if count else None
                ),
                "reason": None if count else values["reason"],
            }
        )
    return rows
