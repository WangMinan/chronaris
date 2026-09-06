"""Shared microbatch preparation and forward objectives for candidate training."""
import time
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
    targets = build_common_pretext_targets(normalized, augmented)
    lag_inputs = build_lag_discrimination_inputs(
        augmented.batch,
        augmented.augmentation_ids,
    )
    positive_batch = move_observation_batch(
        augmented.batch,
        device=resolved.device,
    )
    negative_batch = move_observation_batch(
        lag_inputs.negative_batch,
        device=resolved.device,
    )
    targets = move_common_pretext_targets(
        targets,
        device=resolved.device,
    )
    diagnostics_required = method_name == "chronaris" and (
        chronaris_lag_aware_weight > 0 or chronaris_mechanism_enabled
    )
    positive = encoder(
        positive_batch,
        compute_chronaris_diagnostics=diagnostics_required,
    )
    negative = encoder(
        negative_batch,
        compute_chronaris_diagnostics=chronaris_mechanism_enabled,
    )
    output = heads(
        positive.sequence_embedding,
        negative.sequence_embedding,
        targets,
        weights=CommonPretextWeights(),
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
    )
    return output, mechanism_step, augmented, data_wait_s
