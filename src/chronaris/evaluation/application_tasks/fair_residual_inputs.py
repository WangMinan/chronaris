"""Qualification and fold-local tensors for the fair residual experiment."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.cogpilot_history_features import QUERY_SECONDS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows
from chronaris.modeling.fusion_encoders.causal_query import CausalQueryStream


FOLDS = (
    dict(name='fold01', fit=('sub-cp004', 'sub-cp008', 'sub-cp019'), select=('sub-cp025',),
         score=('sub-cp003', 'sub-cp011', 'sub-cp022')),
    dict(name='fold02', fit=('sub-cp003', 'sub-cp004', 'sub-cp011', 'sub-cp019'), select=('sub-cp022',),
         score=('sub-cp008', 'sub-cp025')),
)


def qualify_fair_inputs(snapshot, output):
    output = Path(output)
    if (output/'qualification.json').exists():
        raise FileExistsError('qualified input record already retained')
    with np.load(snapshot, allow_pickle=False) as raw:
        meta = json.loads(str(raw['metadata']))
        accepted = [r for r in meta['windows'] if r['accepted']]
        values = {k: raw[k] for k in raw.files if k != 'metadata'}
    old_sources = json.loads(Path('docs/review/stage/performance-pilot-20261018/sources.json').read_text())
    assert meta['sources'] == old_sources
    assert len(meta['windows']) == 5951
    assert len(accepted) == len(values['anchor'])
    assert len({r['sample_id'] for r in meta['windows']}) == 5951
    assert all(np.isfinite(v).all() for v in values.values())
    for stream in ('physiology', 'vehicle'):
        assert values[stream].shape[:2] == (len(accepted), 96)
        assert values[stream+'_mask'].mean(axis=1).min() >= .9
        m = values[stream+'_mask']
        assert (values[stream+'_age'][m] > 0).all() and (values[stream+'_age'][m] <= 2).all()
    assert np.max(np.abs(values['anchor'][:, -1]-[r['persistence'] for r in accepted])) <= 1e-10
    groups = []
    for subject in sorted({r['subject'] for r in meta['windows']}):
        rows = [r for r in accepted if r['subject'] == subject]
        groups.append(dict(subject=subject, windows=len(rows), records=len({r['record_id'] for r in rows})))
    errors = np.array([r['component_max_errors'] for r in meta['reconstruction']])
    gates = dict(sources=len(meta['sources']) == 420, records=len(meta['records']) == 84,
                 coverage=len(accepted)/5951 >= .8,
                 per_person=all(r['windows'] >= 100 and r['records'] >= 8 for r in groups),
                 formula=bool((errors[:, :3] <= .0002+1e-10).all() and (errors[:, 3] <= .0006+1e-10).all()),
                 historical_anchor=True, finite=True)
    save_json(output/'source_hashes.json', meta['sources'])
    save_json(output/'reconstruction.json', meta['reconstruction'])
    save_rows(output/'subject_coverage.csv', groups)
    simple = [{k: v for k, v in r.items() if k != 'physiology_quality'} for r in meta['windows']]
    save_rows(output/'sample_manifest.csv', simple)
    quality = [dict(sample_id=r['sample_id'], channel=i, **q)
               for r in meta['windows'] for i, q in enumerate(r.get('physiology_quality', []))]
    save_rows(output/'physiology_quality.csv', quality)
    receipt = dict(status='passed' if all(gates.values()) else 'failed', gates=gates,
                   snapshot_sha256=file_digest(snapshot), snapshot_path=str(Path(snapshot).resolve()),
                   windows=len(accepted), rejected=dict(Counter(r['reason'] for r in meta['windows'] if not r['accepted'])),
                   maximum_reconstruction_errors=errors.max(0).tolist(), encoder_updates=0)
    save_json(output/'qualification.json', receipt)
    return receipt


class FairResidualData:
    def __init__(self, snapshot, receipt, fold, device='cuda'):
        receipt = json.loads(Path(receipt).read_text())
        if receipt['status'] != 'passed' or not all(receipt['gates'].values()):
            raise ValueError('new input qualification did not pass')
        if file_digest(snapshot) != receipt['snapshot_sha256']:
            raise ValueError('new input snapshot changed')
        with np.load(snapshot, allow_pickle=False) as raw:
            self.rows = [r for r in json.loads(str(raw['metadata']))['windows'] if r['accepted']]
            raw_values = {s: raw[s] for s in ('physiology', 'vehicle')}
            masks = {s: raw[s+'_mask'] for s in raw_values}
            ages = {s: raw[s+'_age'] for s in raw_values}
            self.anchor = torch.tensor(raw['anchor'], dtype=torch.float32, device=device)
        self.subjects = np.array([r['subject'] for r in self.rows])
        self.y = np.array([r['target'] for r in self.rows])
        self.fold = fold
        self.roles = {role: np.flatnonzero(np.isin(self.subjects, fold[role])) for role in ('fit', 'select', 'score')}
        assert not set(fold['fit']) & (set(fold['select']) | set(fold['score']))
        assert not set(fold['select']) & set(fold['score']) and all(map(len, self.roles.values()))
        self.score_open = False; self.device = device
        fit = self.roles['fit']
        w = np.array([len(fit)/(len(fold['fit'])*np.sum(self.subjects[fit] == s)) for s in self.subjects[fit]])
        self.weights = torch.zeros(len(self.rows), device=device)
        self.weights[fit] = torch.tensor(w, dtype=torch.float32, device=device)
        self.normalizer = dict(fit_subjects=list(fold['fit']), fit_sample_ids=[self.rows[i]['sample_id'] for i in fit])
        self.streams = {}
        for stream, x in raw_values.items():
            center, scale = [], []
            for field in range(x.shape[-1]):
                vals = x[fit, :, field][masks[stream][fit, :, field]]
                center.append(float(np.median(vals)))
                scale.append(max(1e-6, float(np.quantile(vals, .75)-np.quantile(vals, .25))))
            self.normalizer[stream] = dict(center=center, scale=scale)
            values = np.where(masks[stream], (x-np.array(center))/np.array(scale), 0).astype(np.float32)
            self.streams[stream] = (torch.tensor(values, device=device), torch.tensor(masks[stream], device=device),
                                    torch.tensor(ages[stream], device=device))
        self.queries_s = torch.tensor(QUERY_SECONDS, dtype=torch.float32, device=device)[None, :].expand(len(self.rows), -1)
        self.set_anchor(self.anchor)

    def set_anchor(self, anchor):
        self.anchor = anchor.detach().clone()
        residual = self.y-self.anchor[:, -1].cpu().numpy()
        fit = self.roles['fit']; weights = self.weights[fit].cpu().numpy()
        mean = np.average(residual[fit], weights=weights)
        self.target_scale = max(1e-6, float(np.sqrt(np.average((residual[fit]-mean)**2, weights=weights))))
        self.targets = torch.tensor(residual/self.target_scale, dtype=torch.float32, device=self.device)

    def indices(self, role):
        if role == 'score' and not self.score_open:
            raise PermissionError('outer score data is sealed')
        return self.roles[role]

    def queries(self, ids):
        permitted = np.r_[self.roles['fit'], self.roles['select'], self.roles['score'] if self.score_open else []]
        if not np.isin(ids, permitted).all():
            raise PermissionError('unassigned or sealed subjects requested')
        outputs = []
        for stream in ('physiology', 'vehicle'):
            x, m, age = [a[ids] for a in self.streams[stream]]
            indices = torch.arange(96, device=m.device)[None, :, None].expand_as(m)
            outputs.append(CausalQueryStream(x, m, m.any(-1), torch.where(m, age, torch.inf), self.queries_s[ids],
                                             torch.where(m, indices, -1)))
        return tuple(outputs)

    def statistics(self, ids, include_physiology):
        values = []
        for stream in self.queries(ids)[0 if include_physiology else 1:]:
            x, m = stream.values.cpu().numpy(), stream.feature_mask.cpu().numpy()
            count = np.maximum(m.sum(1), 1)
            mean = (x*m).sum(1)/count
            std = np.sqrt((((x-mean[:, None])*m)**2).sum(1)/count)
            values.append(np.c_[mean, std, x[:, -1], m.mean(1)])
        return np.concatenate(values, axis=1)
