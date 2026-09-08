"""Separate frozen-representation probes for clock magnitude, sign, and response delay."""
from pathlib import Path
import json

import joblib
import numpy as np
import torch

from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder, export_loaded_application_encoder
from chronaris.evaluation.application_tasks.simulation_mechanism_targets import build_simulation_mechanism_targets, CLOCK_OFFSET_TARGET, RESPONSE_LAG_TARGET
from chronaris.evaluation.application_tasks.v4_mechanism_data import timing_scenarios, timing_batch
from chronaris.evaluation.application_tasks.v4_naive_baseline import load_v4_naive_encoder
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY, simulation_unit_root
from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import read_simulation_model_freeze, load_simulation_confirmation
from chronaris.evaluation.application_tasks.v4_simulation_data import load_v4_simulation_development
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.representation import select_observation_batch, load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

SIGNED_CLOCK_TARGET = 'signed_relative_clock_offset_s'
TARGETS = (CLOCK_OFFSET_TARGET, SIGNED_CLOCK_TARGET, RESPONSE_LAG_TARGET)
TARGET_LABELS = ('时钟偏移幅值（秒）', '有符号时钟偏移（秒）', '首个生理字段实际响应时延（秒）')


def mechanism_values(rows, scenario):
    """Open only registered timing targets, after their representations exist."""
    rows = [dict(row, sample_id=row['sample_id']+'::'+scenario.scenario_id, scenario_id=scenario.scenario_id) for row in rows]
    target = build_simulation_mechanism_targets(rows, scenarios=(scenario,), representation_evidence_completed=True)
    ids = [row['sample_id'] for row in rows]
    signed = np.full(len(ids), scenario.physiology_clock_offset_s-scenario.vehicle_clock_offset_s)
    values = np.column_stack((target.values(ids, CLOCK_OFFSET_TARGET), signed, target.values(ids, RESPONSE_LAG_TARGET)))
    if not np.isfinite(values).all():
        raise ValueError('nonfinite timing target')
    return values, target.manifest


def timing_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 3 or len(truth)==0 or not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError('timing predictions require three separate aligned targets')
    rows = []
    for index, (name, label) in enumerate(zip(TARGETS, TARGET_LABELS, strict=True)):
        error = np.abs(truth[:, index]-predicted[:, index]); squared = error**2
        total = squared.sum()
        rows.append(dict(target=name, label=label, rmse=float(np.sqrt(squared.mean())),
            p95_absolute_error=float(np.quantile(error, .95)),
            largest_five_squared_error_fraction=float(np.sort(squared)[-5:].sum()/total) if total else 0.,
            sample_count=len(error), all_windows_retained=True))
    return rows


def run_final_mechanisms(*, formal_root, timing_root, confirmation_root, output_root):
    formal, root = Path(formal_root), Path(output_root)
    frozen_path, model_path = formal/'frozen_configuration.json', formal/'simulation_frozen_models.json'
    models = read_simulation_model_freeze(model_path, sha256_file(model_path),
        freeze_path=frozen_path, freeze_sha256=sha256_file(frozen_path))
    development, fold = load_v4_simulation_development(simulation_root=models['simulation_root'], registry_path=SIMULATION_REGISTRY)
    source = dict(model_freeze_sha256=sha256_file(model_path), timing_audit_sha256=sha256_file(Path(timing_root)/'timing_audit.json'),
        confirmation_audit_sha256=sha256_file(Path(confirmation_root)/'confirmation_generation_audit.json'),
        targets=list(TARGETS), ridge_alpha=1., consumer_fit_role='train', encoder_updates=0)
    path = root/'mechanism_summary.json'
    state = json.loads(path.read_text()) if path.exists() else dict(source=source, records=[], completed_units=[], status='running')
    if state['source'] != source:
        raise ValueError('frozen mechanism inputs changed')
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        with _periodic_training_heartbeat('final_time_mechanisms', 30, root=root) as progress:
            for unit in models['records']:
                method, seed = unit['method'], unit['seed']
                for route in (('self_supervised',) if method == 'naive_time_sync' else unit['routes']):
                    key = f'{method}/{unit["candidate_name"]}/seed{seed}/{route}'
                    if key in state['completed_units']:
                        for record in state['records']:
                            if record['unit'] == key:
                                for filename, digest in record['files'].items():
                                    if sha256_file(filename) != digest:
                                        raise ValueError('saved timing evidence changed')
                        continue
                    progress.update(current_unit=key, phase='export_train_validation')
                    checkpoint = unit['checkpoints'][route]
                    if method == 'naive_time_sync':
                        encoder, normalizer, _ = load_v4_naive_encoder(checkpoint, fold=fold, data_manifest_sha256=unit['data_manifest_sha256'])
                    else:
                        encoder, normalizer, _ = load_frozen_application_encoder(checkpoint, route=route, fold=fold, device='cuda')
                    location = root/key; files = {str(model_path): sha256_file(model_path)}
                    chunks = {role: [] for role in ('train', 'validation', 'held_out')}
                    for scenario in timing_scenarios():
                        batch, manifest = timing_batch(timing_root, scenario.scenario_id, development.sample_manifest_rows)
                        destination = location/'representations'/scenario.scenario_id
                        outputs = export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
                            provider=lambda ids: select_observation_batch(batch, ids), fold=fold, root=destination,
                            export_roles=('train', 'validation'), export_prefix='frozen_timing_probe',
                            label_used_for_encoder_training=route == 'task_guided')
                        held = load_simulation_confirmation(confirmation_root, model_freeze_sha256=sha256_file(model_path), condition=scenario.scenario_id)
                        held_path = simulation_unit_root(formal, unit)/'evaluation'/route/'pressure'/scenario.scenario_id/'representation/held_out'
                        outputs['held_out'] = load_fusion_stream_batch(held_path)
                        rows_by_id = {row['sample_id']: row for row in [*manifest, *held.sample_manifest_rows]}
                        for role, output in outputs.items():
                            if output.sample_ids != getattr(fold, role+'_sample_ids') or output.checkpoint_sha256 != sha256_file(checkpoint):
                                raise ValueError('timing representation differs from its frozen role/checkpoint')
                            directory = held_path if role == 'held_out' else destination/role
                            for filename in ('representation_manifest.json', 'fusion_stream.npz'):
                                artifact = directory/filename; files[str(artifact)] = sha256_file(artifact)
                            rows = [rows_by_id[sample] for sample in output.sample_ids]
                            values, oracle = mechanism_values(rows, scenario)
                            files.update({row['oracle_path']: row['oracle_sha256'] for row in oracle['oracle_files']})
                            chunks[role].append(dict(x=output.pooled_embedding.cpu().numpy(), y=values,
                                ids=[sample+'::'+scenario.scenario_id for sample in output.sample_ids],
                                profiles=[row['profile_id'] for row in rows], condition=[scenario.scenario_id]*len(rows)))
                    arrays = {role: {key: np.concatenate([chunk[key] for chunk in chunks[role]])
                                     for key in ('x', 'y', 'ids', 'profiles', 'condition')} for role in chunks}
                    progress['phase'] = 'fixed_probe_consumers'
                    predictions, probe_files = [], []
                    for index, target in enumerate(TARGETS):
                        model, _ = fit_regressor(arrays['train']['x'], arrays['train']['y'][:, index],
                            arrays['validation']['x'], arrays['validation']['y'][:, index], alpha_values=(1.,), scaler_with_mean=True)
                        predicted = model.predict(arrays['held_out']['x']); predictions.append(predicted)
                        model_file = location/(target+'.joblib'); joblib.dump(model, model_file); probe_files.append(model_file)
                    predicted = np.column_stack(predictions)
                    prediction = location/'predictions.npz'
                    np.savez_compressed(prediction, truth=arrays['held_out']['y'], predicted=predicted,
                        sample_ids=arrays['held_out']['ids'], profile_ids=arrays['held_out']['profiles'], conditions=arrays['held_out']['condition'])
                    for artifact in [prediction, *probe_files]: files[str(artifact)] = sha256_file(artifact)
                    record = dict(unit=key, method=method, seed=seed, route=route, prediction_path=str(prediction), files=files,
                        metrics=timing_metrics(arrays['held_out']['y'], predicted), encoder_labels_used=route == 'task_guided',
                        condition_metrics={scenario.scenario_id: timing_metrics(
                            arrays['held_out']['y'][arrays['held_out']['condition']==scenario.scenario_id],
                            predicted[arrays['held_out']['condition']==scenario.scenario_id]) for scenario in timing_scenarios()},
                        consumer_labels_used=True, interpretation='linear_decodability_of_frozen_representations_not_direct_delay_identification',
                        training_profiles=len(set(arrays['train']['profiles'])), held_out_profiles=len(set(arrays['held_out']['profiles'])))
                    state['records'].append(record)
                    if method == 'naive_time_sync':
                        state['records'].append(record | dict(route='task_guided', shared_with='self_supervised'))
                    state['completed_units'].append(key); write_result(path, state)
                    del encoder, arrays, chunks
                    torch.cuda.empty_cache()
    if len(state['records']) != 36:
        raise ValueError('time mechanisms require six methods, three seeds and both routes')
    state['status'] = 'completed'
    return write_result(path, state)
