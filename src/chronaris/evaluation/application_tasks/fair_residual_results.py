"""Score both locked folds and apply the fixed cross-person decision rule."""
import csv
import gzip
import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_training import load_fair_model, predict_fair
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_rows, save_json


def score_fair_folds(folds, root, report, assets, receipts):
    summary, people, predictions, gates = [], [], [], []
    for name, item in folds.items():
        data = item['data']; data.score_open = True
        ids = data.indices('score'); residual = item['residual_data']; residual.score_open = True
        model = load_fair_model(root/(name+'_aircraft')/'best.pt', data)
        base_features, base_pred = predict_fair(model, data, ids)
        del model
        # Only the scoring part changes; the fitted residual scale must stay fixed.
        scale = residual.target_scale
        residual.anchor[ids] = torch.tensor(base_pred, device=data.device)
        assert residual.target_scale == scale
        result = dict(aircraft_head=base_pred[:, -1], aircraft_ridge=item['readouts']['aircraft'].predict(base_features))
        for method in ('contiformer', 'mult', 'control', 'candidate'):
            current = residual if method in ('control', 'candidate') else data
            checkpoint = root/(name+'_'+method)/'best.pt'
            assert file_digest(checkpoint) == receipts[name+'_'+method]['checkpoint_sha256']
            model = load_fair_model(checkpoint, current)
            features, pred = predict_fair(model, current, ids)
            if method in ('control', 'candidate'):
                features = np.c_[base_features, features]
            result[method+'_ridge'] = item['readouts'][method].predict(features)
            result[method+'_head'] = pred[:, -1]
            del model
        for method, phys in [('vehicle_statistics', False), ('dual_statistics', True)]:
            result[method] = item['readouts'][method].predict(data.statistics(ids, phys))
        fit = data.indices('fit'); y = data.y[ids]
        result['train_mean'] = np.full(len(ids), np.average(data.y[fit], weights=data.weights[fit].cpu().numpy()))
        result['train_median'] = np.full(len(ids), np.median(data.y[fit]))
        result['persistence'] = np.array([data.rows[i]['persistence'] for i in ids])
        for method, pred in result.items():
            if not np.isfinite(pred).all():
                raise ValueError('nonfinite final prediction')
            per_person = []
            for person in sorted(set(data.subjects[ids])):
                mask = data.subjects[ids] == person
                delta = pred[mask]-y[mask]
                row = dict(fold=name, method=method, subject=str(person), samples=int(mask.sum()),
                           rmse=float(np.sqrt(np.mean(delta**2))), mae=float(np.mean(np.abs(delta))))
                people.append(row); per_person.append(row)
            summary.append(dict(fold=name, method=method, rmse=float(np.mean([r['rmse'] for r in per_person])),
                                mae=float(np.mean([r['mae'] for r in per_person]))))
            predictions.extend(dict(fold=name, method=method, sample_id=data.rows[i]['sample_id'], subject=str(data.subjects[i]),
                                    truth=float(data.y[i]), prediction=float(v)) for i,v in zip(ids,pred))
        controls = [r for r in summary if r['fold'] == name and not r['method'].startswith('candidate_')]
        strongest = min(controls, key=lambda r:r['rmse'])
        for candidate in ('candidate_ridge', 'candidate_head'):
            current = next(r for r in summary if r['fold'] == name and r['method'] == candidate)
            wins = sum(next(r['rmse'] for r in people if r['fold']==name and r['method']==candidate and r['subject']==s) <
                       next(r['rmse'] for r in people if r['fold']==name and r['method']==strongest['method'] and r['subject']==s)
                       for s in set(data.subjects[ids]))
            improvement = min(1-current['rmse']/r['rmse'] for r in controls)
            mae_ok = all(current['mae'] <= 1.02*r['mae'] for r in controls)
            gates.append(dict(fold=name, candidate=candidate, strongest=strongest['method'], improvement=improvement,
                              subjects_improved=wins, subjects=len(set(data.subjects[ids])), mae_passed=mae_ok,
                              passed=improvement >= .02 and wins >= 2 and mae_ok))
    save_rows(report/'summary.csv', summary); save_rows(report/'subject_metrics.csv', people)
    with gzip.open(report/'predictions.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    save_json(report/'performance_gates.json', gates)
    return dict(status='completed', representation_success=all(r['passed'] for r in gates if r['candidate']=='candidate_ridge'),
                task_head_success=all(r['passed'] for r in gates if r['candidate']=='candidate_head'),
                stable_success=all(r['passed'] for r in gates), gates=gates, prediction_rows=len(predictions),
                scoring_scope='two_previously_exposed_development_folds_seed17')
