"""One retained Dingxin record with chronological, complete-support isolation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from chronaris.evaluation.dingxin.simple_downstream_protocol import (
    SimpleFoldTargetBundle, _fit_maneuver_fold, _fit_physiology_fold)
from chronaris.representation import FoldLineage

PROTOCOL = 'single_record_chronological_blocks_v1'


def deduplicated_dingxin_data(data):
    """Keep the lexically first record and all its views, independent of scores."""
    from chronaris.evaluation.application_tasks.v4_dingxin_data import _task_targets, audit_dingxin_vehicle_reuse
    audit = audit_dingxin_vehicle_reuse(data)
    signatures = {}
    for row in audit['context_rows']:
        signatures.setdefault(row['sortie_id'], {})[row['target_start_offset_ms']] = row['content_sha256']
    sorties = sorted(signatures)
    if len(sorties) != 2 or signatures[sorties[0]] != signatures[sorties[1]]:
        raise ValueError('deduplication requires the verified two-record vehicle duplication')
    retained = sorties[0]
    contexts = data.raw_targets.contexts[data.raw_targets.contexts.sortie_id == retained].copy()
    ids = set(contexts.context_id)
    vehicle_ids = set(contexts.vehicle_context_id)
    raw = replace(data.raw_targets, contexts=contexts,
        maneuver_statistics=data.raw_targets.maneuver_statistics[
            data.raw_targets.maneuver_statistics.vehicle_context_id.isin(vehicle_ids)].copy(),
        physiology_statistics=data.raw_targets.physiology_statistics[
            data.raw_targets.physiology_statistics.context_id.isin(ids)].copy())
    blocks = contexts.drop_duplicates('vehicle_context_id').sort_values('target_start_offset_ms')
    if len(blocks) != 30:
        raise ValueError('retained Dingxin record must have the frozen 30 vehicle contexts')
    held = blocks.iloc[-3:]
    outer_train = blocks[blocks.target_end_exclusive_ms <= held.input_start_offset_ms.min()]
    validation = outer_train.iloc[-3:]
    train = outer_train[outer_train.target_end_exclusive_ms <= validation.input_start_offset_ms.min()]
    if (len(train), len(validation), len(held), len(outer_train)) != (12, 3, 3, 21):
        raise ValueError('deduplicated complete-support boundaries differ from the frozen split')
    def samples(frame):
        return tuple(contexts.loc[contexts.vehicle_context_id.isin(frame.vehicle_context_id), 'context_id'].astype(str))
    fold = FoldLineage(PROTOCOL, samples(train), samples(validation), samples(held))
    outer = FoldLineage(PROTOCOL, samples(outer_train), (), fold.held_out_sample_ids)
    embargo = tuple(sample for sample in contexts.context_id
                    if sample not in set(fold.train_sample_ids + fold.validation_sample_ids + fold.held_out_sample_ids))
    fitted = fit_deduplicated_targets(raw, fold, fit_ids=fold.train_sample_ids)
    targets, definitions, sampling = _task_targets(fitted, fold, fold.train_sample_ids + fold.validation_sample_ids, 'inner_training')
    targets.manifest.update(evaluation_protocol=PROTOCOL, retained_sortie=retained)
    manifest = dict(parent_data_manifest_sha256=data.data_manifest_sha256, evaluation_protocol=PROTOCOL,
        retained_sortie=retained, excluded_sorties=sorties[1:], selection_rule='lexical_first_sortie_without_outcomes',
        encoder_fold=fold.to_dict(), consumer_fold=outer.to_dict(), embargo=embargo)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    result = replace(data, raw_targets=raw, folds=(fold,), outer_folds=(outer,), embargo={fold.fold_id: embargo},
        fitted_targets=fitted, targets_by_fold={fold.fold_id: targets}, definitions_by_fold={fold.fold_id: definitions},
        sampling_by_fold={fold.fold_id: sampling}, data_manifest_sha256=digest, evaluation_protocol=PROTOCOL)
    return result, manifest


def fit_deduplicated_targets(raw, fold, *, fit_ids):
    """Reuse original target formulas; all learned target parameters use fit_ids only."""
    fit_ids = tuple(fit_ids)
    contexts = raw.contexts
    if not fit_ids or len(set(fit_ids)) != len(fit_ids) or not set(fit_ids) <= set(fold.train_sample_ids):
        raise ValueError('deduplicated target fitting crossed the allowed training role')
    groups = contexts.loc[contexts.context_id.isin(fit_ids), 'vehicle_context_id']
    if set(contexts.loc[contexts.vehicle_context_id.isin(groups), 'context_id']) != set(fit_ids):
        raise ValueError('deduplicated target fitting must retain all shared vehicle views')
    roles = {sample: role for role in ('train', 'validation', 'held_out')
             for sample in getattr(fold, role + '_sample_ids')}
    if len(roles) != sum(len(getattr(fold, role + '_sample_ids')) for role in ('train', 'validation', 'held_out')):
        raise ValueError('deduplicated target roles overlap')
    if not set(roles) <= set(contexts.context_id):
        raise ValueError('deduplicated target roles contain unknown samples')
    for left, right in (('train', 'validation'), ('train', 'held_out'), ('validation', 'held_out')):
        earlier = contexts[contexts.context_id.isin(getattr(fold, left + '_sample_ids'))]
        later = contexts[contexts.context_id.isin(getattr(fold, right + '_sample_ids'))]
        if len(earlier) and len(later) and earlier.target_end_exclusive_ms.max() > later.input_start_offset_ms.min():
            raise ValueError('deduplicated roles overlap in complete input plus target support')
    roles = {str(sample): roles.get(sample, 'embargo') for sample in contexts.context_id}
    maneuver, thresholds = _fit_maneuver_fold(fold_id=fold.fold_id, contexts=contexts,
        statistics=raw.maneuver_statistics, role_by_context=roles, fit_context_ids=fit_ids,
        minimum_semantic_count=4, eps=1e-6)
    physiology, field_thresholds = _fit_physiology_fold(fold_id=fold.fold_id,
        statistics=raw.physiology_statistics, role_by_context=roles, fit_context_ids=fit_ids,
        train_valid_ratio=.80, eps=1e-6)
    return SimpleFoldTargetBundle(pd.DataFrame([dict(fold_id=fold.fold_id, split_strategy=PROTOCOL,
        fit_context_count=len(fit_ids), fit_sample_ids=fit_ids)]), pd.DataFrame(maneuver),
        pd.DataFrame(physiology), pd.DataFrame(thresholds + field_thresholds))


def deduplicated_outer_consumer_inputs(data, fold_id):
    from chronaris.evaluation.application_tasks.v4_dingxin_data import _task_targets
    from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context
    inner = next(fold for fold in data.folds if fold.fold_id == fold_id)
    outer = next(fold for fold in data.outer_folds if fold.fold_id == fold_id)
    fitted = fit_deduplicated_targets(data.raw_targets, outer, fit_ids=outer.train_sample_ids)
    targets, definitions, _ = _task_targets(fitted, outer, outer.train_sample_ids + outer.held_out_sample_ids,
                                           'outer_training_after_encoder_freeze')
    targets.manifest['evaluation_protocol'] = PROTOCOL
    context_data = replace(data, fitted_targets=fitted, targets_by_fold={fold_id: targets})
    return dict(fold=outer, targets=targets, definitions=definitions, encoder_fold=inner,
        context=native_consumer_context('dingxin', context_data, outer, include_held_out=True),
        data_manifest_sha256=data.data_manifest_sha256)


def prepare_deduplicated_dingxin(output_root):
    """Write the actual retained-record audit without computing model scores."""
    from chronaris.evaluation.application_tasks.v4_dingxin_data import (
        load_v4_dingxin_development, audit_dingxin_vehicle_reuse)
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    snapshot = Path('artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot')
    original = json.loads((snapshot / 'snapshot_manifest.json').read_text())
    for item in original['files']:
        if sha256_file(snapshot / item['relative_path']) != item['sha256']:
            raise ValueError('original Dingxin snapshot file changed')
    data, selection = deduplicated_dingxin_data(load_v4_dingxin_development())
    audit = audit_dingxin_vehicle_reuse(data)
    outer = deduplicated_outer_consumer_inputs(data, data.folds[0].fold_id)
    result = dict(status='completed', selection=selection, content_audit=audit,
        encoder_context_counts={role: len(getattr(data.folds[0], role + '_sample_ids'))
                                for role in ('train', 'validation', 'held_out')},
        consumer_context_counts={role: len(getattr(outer['fold'], role + '_sample_ids')) for role in ('train', 'held_out')},
        inner_fields=data.targets_by_fold[data.folds[0].fold_id].manifest['physiology_field_order'],
        outer_fields=outer['targets'].manifest['physiology_field_order'],
        model_scores_generated=False, source_files_unchanged=True)
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
    (root / 'data_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result
