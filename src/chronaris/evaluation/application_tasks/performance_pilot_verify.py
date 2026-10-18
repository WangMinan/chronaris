"""Independent result arithmetic, role, training-receipt and source verification."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json


def verify_performance_pilot(report):
    report = Path(report)
    protected = json.loads((report/'protected_sources.json').read_text())
    changed = [p for p, expected in protected.items() if file_digest(p) != expected]
    if changed:
        raise AssertionError(f'protected sources changed: {changed[:3]}')
    qualification = json.loads((report/'qualification.json').read_text())
    assert file_digest(qualification['snapshot_path']) == qualification['snapshot_sha256']
    run = json.loads((report/'run_status.json').read_text())
    assert run['status'] == 'completed'
    lock = json.loads((report/'locked_models.json').read_text())
    assert lock['binding'] == run['binding']
    assert set(lock['receipts']) == {'vehicle_only', 'physiology_only', 'contiformer', 'mult', 'same_backbone'}
    readout_path = Path(qualification['snapshot_path']).parent/'readouts.joblib'
    assert file_digest(readout_path) == lock['readout_sha256']
    assert file_digest(report/'qualification.json') == lock['binding']['qualification_sha256']
    assert file_digest(report/'PLAN.md') == lock['binding']['plan_sha256']
    roles = lock['binding']['role_subjects']
    assert not set(roles['fit']) & set(roles['select'])
    assert not (set(roles['fit']) | set(roles['select'])) & set(roles['score'])
    normalizer = json.loads((report/'normalization.json').read_text())
    assert set(normalizer['fit_subjects']) == set(roles['fit'])
    manifest = list(csv.DictReader((report/'sample_manifest.csv').open()))
    accepted = {r['sample_id']: r for r in manifest if r['accepted'] == 'True'}
    assert all(accepted[s]['subject'] in roles['fit'] for s in normalizer['fit_sample_ids'])
    source_files = json.loads((report/'sources.json').read_text())
    assert len(source_files) == 420
    samples = {s for s, r in accepted.items() if r['subject'] in roles['score']}
    assert len(samples) == run['score_samples']
    with gzip.open(report/'predictions.csv.gz', 'rt') as handle:
        predictions = list(csv.DictReader(handle))
    scores = list(csv.DictReader((report/'subject_metrics.csv').open()))
    summary = list(csv.DictReader((report/'summary.csv').open()))
    actual = {}; max_delta = 0.
    for method in sorted({r['method'] for r in predictions}):
        rows = [r for r in predictions if r['method'] == method]
        assert len(rows) == len(samples) and {r['sample_id'] for r in rows} == samples
        for row in rows:
            assert row['subject'] == accepted[row['sample_id']]['subject']
            assert float(row['truth']) == float(accepted[row['sample_id']]['target'])
        per_person = []
        for subject in roles['score']:
            group = [r for r in rows if r['subject'] == subject]
            errors = np.array([float(r['prediction'])-float(r['truth']) for r in group])
            rmse, mae = float(np.linalg.norm(errors)/np.sqrt(len(errors))), float(np.abs(errors).sum()/len(errors))
            recorded = next(r for r in scores if r['method'] == method and r['subject'] == subject)
            max_delta = max(max_delta, abs(rmse-float(recorded['rmse'])), abs(mae-float(recorded['mae'])))
            actual[method, subject] = rmse; per_person.append((rmse, mae))
        recorded = next(r for r in summary if r['method'] == method)
        average = np.mean(per_person, axis=0)
        max_delta = max(max_delta, abs(average[0]-float(recorded['rmse'])), abs(average[1]-float(recorded['mae'])))
        actual[method, 'mean'] = float(average[0])
    assert max_delta < 1e-10
    controls = [r['method'] for r in summary if r['scope'] == 'primary' and r['method'] != 'same_backbone']
    best = min(controls, key=lambda m: actual[m, 'mean'])
    improvement = 1-actual['same_backbone', 'mean']/actual[best, 'mean']
    wins = sum(actual['same_backbone', s] < actual[best, s] for s in roles['score'])
    assert run['success'] == (improvement >= .02 and wins >= 2)
    updates = 0
    for method, receipt in lock['receipts'].items():
        assert receipt['complete'] and receipt['reload_passed'] and receipt['epochs'] >= 20
        assert file_digest(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        rows = list(csv.DictReader((report/f'{method}_history.csv').open()))
        assert len(rows) == receipt['epochs'] and int(rows[-1]['updates']) == receipt['updates']
        assert receipt['best_epoch'] <= receipt['epochs'] <= 100
        if receipt['status'] == 'early_stopped':
            assert receipt['epochs']-receipt['best_epoch'] >= 15
        assert all(float(r['selection_normalized_rmse']) >= 0 for r in rows)
        updates += receipt['updates']
    assert updates == run['encoder_updates']
    result = dict(status='passed', protected_files=len(protected), changed_sources=changed,
                  source_files_bound=len(source_files), prediction_rows=len(predictions),
                  subject_metrics=len(scores), max_metric_recompute_delta=max_delta,
                  complete_models=len(lock['receipts']), encoder_updates=updates,
                  candidate_passed=run['success'], best_control=best,
                  candidate_relative_improvement=float(improvement), score_subject_wins=wins)
    save_json(report/'verification.json', result)
    return result


def replay_performance_pilot(report):
    """Reload all five locked checkpoints and reproduce scored predictions."""
    import joblib
    import torch
    from chronaris.evaluation.application_tasks.performance_pilot_data import PerformancePilotData
    from chronaris.evaluation.application_tasks.performance_pilot_training import build_pilot_model, export_features
    from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock

    report = Path(report)
    lock = json.loads((report/'locked_models.json').read_text())
    qualification = json.loads((report/'qualification.json').read_text())
    snapshot = Path(qualification['snapshot_path'])
    torch.set_num_threads(1)
    with development_gpu_lock() as acquired:
        if not acquired:
            raise BlockingIOError('GPU replay must respect the experiment lock')
        data = PerformancePilotData(snapshot, report/'qualification.json', 'cuda')
        assert json.loads(json.dumps(data.normalizer)) == json.loads((report/'normalization.json').read_text())
        data.score_open = True; score = data.indices('score')
        path = snapshot.parent/'readouts.joblib'
        assert file_digest(path) == lock['readout_sha256']
        readouts = joblib.load(path)
        with gzip.open(report/'predictions.csv.gz', 'rt') as handle:
            predictions = list(csv.DictReader(handle))
        deltas = {}
        for method, receipt in lock['receipts'].items():
            assert file_digest(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
            state = torch.load(receipt['checkpoint_path'], map_location='cuda', weights_only=True)
            assert state['binding'] == lock['binding']
            model = build_pilot_model(method, 'cuda'); model.load_state_dict(state['encoder'])
            head = torch.nn.Linear(64, 1).cuda(); head.load_state_dict(state['head'])
            features, head_prediction = export_features(model, head, method, data, score)
            for name, actual in [(method, readouts[method].predict(features)),
                                 (method+'_task_head', head_prediction*data.target_scale+data.target_center)]:
                rows = [r for r in predictions if r['method'] == name]
                assert [r['sample_id'] for r in rows] == [data.rows[i]['sample_id'] for i in score]
                expected = np.array([float(r['prediction']) for r in rows])
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                deltas[name] = float(np.max(np.abs(actual-expected)))
            del model, head
    result = dict(status='passed', models=len(lock['receipts']), samples_per_model=len(score),
                  max_prediction_deltas=deltas, optimizer_updates=0, atol=1e-5, rtol=1e-5)
    save_json(report/'fresh_process_replay.json', result)
    return result
