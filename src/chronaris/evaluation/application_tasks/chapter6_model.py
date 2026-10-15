"""Bounded application-model training using the existing observed-data contract."""
from dataclasses import asdict
from pathlib import Path

from chronaris.evaluation.application_tasks.dingxin_context_data import build_dingxin_lazy_context_index
from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe
from chronaris.modeling.training import load_common_pretraining_checkpoint, train_pretext_candidate
from chronaris.models.alignment.calibrated_physics import fit_physics_calibration
from chronaris.representation import (DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY, FoldLineage,
                                     TrainOnlyRobustNormalizer, select_observation_batch)
from chronaris.serving.fusion_workbench.cases import write_json
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def train_application_model(*, source, snapshot, audit, output):
    """Reuse 12 chronological training blocks; never request held-out observations."""
    source, audit, output = Path(source).resolve(), Path(audit), Path(output)
    source_hash = sha256_file(source)
    _, _, saved_normalizer, payload = load_common_pretraining_checkpoint(source, device='cpu')
    if payload['method_name'] != 'chronaris' or payload['seed'] != 17:
        raise ValueError('application model requires the fixed seed-17 source')
    fold = FoldLineage(payload['fold']['fold_id'], **{
        role + '_sample_ids': tuple(payload['fold'][role + '_sample_ids'])
        for role in ('train', 'validation', 'held_out')})
    if len(fold.train_sample_ids) != 24 or len(fold.validation_sample_ids) != 6:
        raise ValueError('application model requires the frozen 12-block, two-view split')
    index = build_dingxin_lazy_context_index(snapshot_root=snapshot,
        field_role_manifest_path=audit / 'field_role_manifest.csv',
        context_manifest_path=audit / 'context_sample_manifest.jsonl',
        maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY)
    for stream in ('physiology', 'vehicle'):
        if tuple(payload[stream + '_feature_names']) != getattr(index.plan.schema, stream + '_feature_names'):
            raise ValueError('training observations differ from the source schema')
    raw = index.load_batch(fold.train_sample_ids + fold.validation_sample_ids)
    def provider(ids):
        if not set(ids) <= set(raw.sample_ids):
            raise ValueError('application training requested an unauthorized sample')
        return select_observation_batch(raw, ids)
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids, batch_size=4)
    if normalizer.to_manifest() != saved_normalizer.to_manifest():
        raise ValueError('training-only normalization differs from the frozen source')
    # Field units and derivative identities are not calibrated for this snapshot.
    calibration = fit_physics_calibration(normalizer, provider, train_sample_ids=fold.train_sample_ids,
        vehicle_feature_names=index.plan.schema.vehicle_feature_names, relations=())
    candidate, config, _, mechanisms, record = training_recipe('thesis_reference', method='chronaris',
        full=True, seed=17, digest=payload['config']['data_manifest_sha256'],
        train_count=len(fold.train_sample_ids), calibration=calibration, device='cpu')
    provenance = dict(recipe=record, source_checkpoint=str(source), source_sha256=source_hash,
        fold=fold.to_dict(), schema=index.plan.schema.to_dict(), snapshot=str(Path(snapshot).resolve()),
        scope='chapter6_application_only_no_task_labels_no_held_out',
        physics_relations='none_verified_for_observed_field_units_and_derivatives')
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / 'application-training.json'
    if record_path.exists():
        import json
        if json.loads(record_path.read_text()) != json.loads(json.dumps(provenance)):
            raise ValueError('application training provenance changed')
    write_json(record_path, provenance)
    result = train_pretext_candidate('chronaris', candidate=candidate, batch=None, batch_provider=provider,
        fold=fold, physiology_feature_names=index.plan.schema.physiology_feature_names,
        vehicle_feature_names=index.plan.schema.vehicle_feature_names,
        vehicle_field_labels=tuple(tuple(v) for v in payload['vehicle_field_labels']),
        normalizer=normalizer, output_root=output, config=config,
        initialization_checkpoint=source, **mechanisms)
    _, _, _, completed = load_common_pretraining_checkpoint(result.best_checkpoint_path, device='cpu')
    assert completed['config']['semantic_event_enabled'] and completed['config']['learnable_semantic_queries']
    assert completed['total_optimizer_updates'] == config.max_updates == 300
    assert sha256_file(source) == source_hash
    write_json(output / 'training-result.json', asdict(result))
    return result
