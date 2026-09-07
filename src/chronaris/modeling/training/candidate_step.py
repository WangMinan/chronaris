"""Shared microbatch preparation and forward objectives for candidate training."""
import time
import torch
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.candidate_mechanisms import build_candidate_mechanism_step, _load_batch
from chronaris.modeling.training.pretext import CommonPretextWeights
from chronaris.representation import (
    build_batch_augmentation_realizations, apply_augmentation_realizations,
    build_common_pretext_targets, build_lag_discrimination_inputs, move_common_pretext_targets,
)


def pretext_micro_step(
    *, encoder, heads, shift_head, batch, batch_provider, sample_ids, normalizer,
    resolved, policy, method_name, epoch, chronaris_lag_aware_weight,
    chronaris_mechanism_enabled, chronaris_explicit_shift_weight,
    chronaris_event_pair_weight, optimizer_updates=None,
):
    started = time.perf_counter()
    raw = _load_batch(batch, batch_provider, sample_ids)
    data_wait_s = time.perf_counter() - started
    normalized = normalizer.transform(raw)
    plans = build_batch_augmentation_realizations(
        sample_ids,
        epoch=epoch,
        global_seed=resolved.seed,
        context_duration_s=normalized.context_durations_s.tolist(),
        policy=policy,
    )
    augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
    output, positive, negative = public_pretext_forward(
        encoder=encoder, heads=heads, normalized=normalized, augmented=augmented, device=resolved.device,
        positive_diagnostics=method_name == "chronaris" and (chronaris_lag_aware_weight > 0 or chronaris_mechanism_enabled),
        negative_diagnostics=chronaris_mechanism_enabled,
    )
    mechanism_step = build_candidate_mechanism_step(
        encoder=encoder,
        shift_head=shift_head,
        positive=positive,
        negative=negative,
        augmented=augmented,
        group_ids=raw.group_ids,
        epoch=epoch,
        optimizer_updates=optimizer_updates,
        device=resolved.device,
        mechanism_enabled=chronaris_mechanism_enabled,
        lag_aware_weight=chronaris_lag_aware_weight,
        explicit_shift_weight=chronaris_explicit_shift_weight,
        event_pair_weight=chronaris_event_pair_weight,
        continuous_alignment_weight=resolved.continuous_alignment_weight,
        independent_pair_weight=resolved.independent_pair_weight,
    )
    return output, mechanism_step, augmented, data_wait_s


def public_pretext_forward(*, encoder, heads, normalized, augmented, device,
                           positive_diagnostics=False, negative_diagnostics=False):
    targets = move_common_pretext_targets(build_common_pretext_targets(normalized, augmented), device=device)
    positive_batch = move_observation_batch(augmented.batch, device=device)
    positive = encoder(positive_batch, compute_chronaris_diagnostics=positive_diagnostics)
    if heads.cross_stream_enabled:
        lag_inputs = build_lag_discrimination_inputs(augmented.batch, augmented.augmentation_ids)
        negative_batch = move_observation_batch(lag_inputs.negative_batch, device=device)
        negative = encoder(negative_batch, compute_chronaris_diagnostics=negative_diagnostics)
    else:
        negative_batch, negative = positive_batch, positive
    lag_valid = None
    if heads.modality_feature_counts is not None:
        lag_valid = torch.ones(len(positive_batch.sample_ids), dtype=torch.bool, device=device)
        for batch in (positive_batch, negative_batch):
            for stream in ("physiology", "vehicle"):
                history = getattr(batch, f"{stream}_timestamps_s") <= batch.query_timestamps_s[:, -1, None]
                lag_valid &= (getattr(batch, f"{stream}_point_mask") & history).any(dim=1)
    output = heads(positive.sequence_embedding, negative.sequence_embedding, targets,
        weights=CommonPretextWeights(), positive_valid_mask=positive.modality_available_mask,
        negative_valid_mask=negative.modality_available_mask, lag_valid_mask=lag_valid)
    return output, positive, negative
