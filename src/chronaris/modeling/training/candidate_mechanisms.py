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
    independent_window_pair_loss_term,
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
    optimizer_updates: int | None = None,
    continuous_alignment_weight: float = 0.2,
    independent_pair_weight: float = 0.0,
) -> CandidateMechanismStep:
    zero = positive.sequence_embedding.sum() * 0.0
    additional_loss = zero
    rows: list[Mapping[str, object]] = []
    metrics: dict[str, Mapping[str, object]] = {}
    warmup_fraction = (
        min(epoch / 5.0, 1.0) if optimizer_updates is None
        else min(max((optimizer_updates - 50) / 150.0, 0.0), 1.0)
    )
    if mechanism_enabled:
        weights = chronaris_auxiliary_weight_schedule(epoch, optimizer_updates=optimizer_updates,
            continuous_alignment_weight=continuous_alignment_weight)
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
    if shift_head is not None and (optimizer_updates is None or warmup_fraction > 0):
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
    elif shift_head is not None:
        rows.append({"term_name": "explicit_time_shift", "weight": 0., "count": 0,
                     "status": "scheduled_zero", "raw_loss": None, "weighted_loss": None,
                     "reason": "mechanisms_disabled_first_50_updates"})
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
    if independent_pair_weight > 0:
        pairing = positive.auxiliary.get("independent_pairing")
        if pairing is None:
            raise RepresentationContractError("independent pair loss requires separate modality histories")
        if warmup_fraction > 0:
            term, pair_metrics = independent_window_pair_loss_term(pairing, group_ids, weight=independent_pair_weight * warmup_fraction)
            if term.weighted_loss is not None:
                additional_loss = additional_loss + term.weighted_loss
            rows.extend(pretext_loss_terms_to_rows((term,)))
            metrics[term.term_name] = pair_metrics
        else:
            rows.append({"term_name": "independent_window_pairing", "weight": 0., "count": 0,
                "status": "scheduled_zero", "raw_loss": None, "weighted_loss": None,
                "reason": "mechanisms_disabled_first_50_updates"})
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
    continuous_alignment_weight=0.2,
    independent_pair_weight=0.0,
    public_heads=None,
) -> Mapping[str, object]:
    from chronaris.modeling.training.candidate_step import public_pretext_forward
    from chronaris.modeling.training.candidate_validation import (
        _evaluate_public_losses, _empty_loss_totals, _accumulate_loss_terms, _finalize_loss_totals)
    if not any(
        (
            mechanism_enabled,
            lag_aware_weight > 0,
            shift_head is not None,
            event_pair_weight > 0,
            independent_pair_weight > 0,
        )
    ):
        result = {"weighted_total": 0.0, "terms": []}
        if public_heads is not None:
            result["public_losses"] = _evaluate_public_losses(encoder=encoder, heads=public_heads,
                batch=batch, batch_provider=batch_provider, sample_ids=sample_ids, batch_size=batch_size,
                normalizer=normalizer, policy=policy, seed=seed, device=device)
        return result
    encoder.eval()
    public_totals = _empty_loss_totals()
    if public_heads is not None:
        public_heads.eval()
    if shift_head is not None:
        shift_head.eval()
    batches = (
        interleaved_group_batch_ids(
            batch,
            batch_provider,
            sample_ids,
            batch_size,
        )
        if event_pair_weight > 0 or independent_pair_weight > 0
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
    independent_sums = dict(pair_sums, negative_pair_count=0)
    with torch.inference_mode():
        for ids in batches:
            raw = _load_batch(batch, batch_provider, ids)
            normalized = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                ids,
                epoch=0,
                global_seed=seed,
                context_duration_s=normalized.context_durations_s.tolist(),
                policy=policy,
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
            if public_heads is not None:
                public, positive, negative = public_pretext_forward(encoder=encoder, heads=public_heads,
                    normalized=normalized, augmented=augmented, device=device,
                    positive_diagnostics=mechanism_enabled or lag_aware_weight > 0, negative_diagnostics=mechanism_enabled)
                _accumulate_loss_terms(public_totals, public.terms)
            else:
                positive = encoder(move_observation_batch(augmented.batch, device=device),
                    compute_chronaris_diagnostics=mechanism_enabled or lag_aware_weight > 0)
                negative = None
                if mechanism_enabled:
                    lag_inputs = build_lag_discrimination_inputs(augmented.batch, augmented.augmentation_ids)
                    negative = encoder(move_observation_batch(lag_inputs.negative_batch, device=device),
                        compute_chronaris_diagnostics=True)
            if mechanism_enabled:
                weights = chronaris_auxiliary_weight_schedule(5, continuous_alignment_weight=continuous_alignment_weight)
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
            if independent_pair_weight > 0:
                term, metrics = independent_window_pair_loss_term(positive.auxiliary["independent_pairing"],
                    raw.group_ids, weight=independent_pair_weight)
                _accumulate_row(totals, pretext_loss_terms_to_rows((term,))[0])
                count = metrics["valid_pair_count"]
                if count:
                    for name in ("positive_similarity", "negative_similarity", "recall_at_1"):
                        independent_sums[name] += metrics[name] * count
                    independent_sums["valid_pair_count"] += count
                    independent_sums["negative_pair_count"] += metrics["negative_pair_count"]
    terms = _finalize_rows(totals)
    pair_count = int(pair_sums["valid_pair_count"])
    return {
        **({"public_losses": _finalize_loss_totals(public_totals,
            allow_unavailable=public_heads.modality_feature_counts is not None)} if public_heads is not None else {}),
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
        "independent_pairing": {"valid_pair_count": independent_sums["valid_pair_count"],
            "negative_pair_count": independent_sums["negative_pair_count"], "actual_batch_size": batch_size,
            "negative_pool": "actual_forward_batch", **{name: independent_sums[name] / independent_sums["valid_pair_count"]
                if independent_sums["valid_pair_count"] else None for name in ("positive_similarity", "negative_similarity", "recall_at_1")}},
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
