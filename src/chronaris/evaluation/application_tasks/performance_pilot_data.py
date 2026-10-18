"""Verified historical query tensors with fit-only normalization and role access."""
from dataclasses import replace
import json

import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import FIT_SUBJECTS, SELECT_SUBJECTS, SCORE_SUBJECTS
from chronaris.modeling.fusion_encoders.causal_query import CausalQueryStream


class PerformancePilotData:
    def __init__(self, snapshot, receipt, device):
        qualification = json.loads(receipt.read_text())
        if qualification['status'] != 'passed' or not all(qualification['gates'].values()):
            raise ValueError('task qualification did not pass')
        if file_digest(snapshot) != qualification['snapshot_sha256']:
            raise ValueError('qualified snapshot changed')
        with np.load(snapshot, allow_pickle=False) as raw:
            meta = json.loads(str(raw['metadata']))
            self.rows = [r for r in meta['windows'] if r['accepted']]
            x, t, mask = (raw[k] for k in ('values', 'observation_times', 'masks'))
        self.subjects = np.array([r['subject'] for r in self.rows])
        self.roles = {name: np.flatnonzero(np.isin(self.subjects, group))
                      for name, group in [('fit', FIT_SUBJECTS), ('select', SELECT_SUBJECTS), ('score', SCORE_SUBJECTS)]}
        if sum(map(len, self.roles.values())) != len(self.rows) or not all(map(len, self.roles.values())):
            raise ValueError('unassigned or empty subject role')
        self.score_open = False
        self.y = np.array([r['target'] for r in self.rows])
        fit = self.roles['fit']
        weights = np.array([len(fit)/(len(FIT_SUBJECTS)*np.sum(self.subjects[fit] == s)) for s in self.subjects[fit]])
        self.target_center = float(np.average(self.y[fit], weights=weights))
        self.target_scale = float(np.sqrt(np.average((self.y[fit]-self.target_center)**2, weights=weights)))
        center, scale = [], []
        for f in range(x.shape[-1]):
            valid = x[fit, :, f][mask[fit, :, f]]
            center.append(float(np.median(valid)))
            scale.append(max(1e-6, float(np.quantile(valid, .75)-np.quantile(valid, .25))))
        self.normalizer = dict(center=center, scale=scale, fit_subjects=FIT_SUBJECTS,
                               fit_sample_ids=[self.rows[i]['sample_id'] for i in fit],
                               weighting='finite causal query values', target_center=self.target_center,
                               target_scale=self.target_scale)
        x = np.where(mask, (x-np.array(center))/np.array(scale), 0).astype(np.float32)
        q = np.broadcast_to(np.linspace(0, 30, 96, endpoint=False)[None, :], (len(x), 96)).copy()
        age = np.where(mask, q[:, :, None]-t, np.inf).astype(np.float32)
        self.values = torch.tensor(x, device=device)
        self.mask = torch.tensor(mask, device=device)
        self.age = torch.tensor(age, device=device)
        self.times = torch.tensor(q, dtype=torch.float32, device=device)
        self.weights = torch.zeros(len(x), device=device)
        self.weights[fit] = torch.tensor(weights, dtype=torch.float32, device=device)
        self.targets = torch.tensor((self.y-self.target_center)/self.target_scale, dtype=torch.float32, device=device)
        self.snapshot_sha256 = qualification['snapshot_sha256']

    def indices(self, role):
        if role == 'score' and not self.score_open:
            raise PermissionError('score role remains sealed until every model and readout is fixed')
        return self.roles[role]

    def queries(self, indices):
        if not self.score_open and np.isin(indices, self.roles['score']).any():
            raise PermissionError('score query access is sealed')
        values, mask, age, time = (a[indices] for a in (self.values, self.mask, self.age, self.times))
        streams = []
        for section in (slice(0, 4), slice(4, 12)):
            m = mask[:, :, section]
            snapshot_rows = torch.arange(96, device=m.device)[None, :, None].expand_as(m)
            streams.append(CausalQueryStream(values[:, :, section], m, m.any(-1), age[:, :, section], time,
                                            torch.where(m, snapshot_rows, -1)))
        return tuple(streams)

    def historical_statistics(self, indices):
        p, v = self.queries(indices)
        result = []
        for stream in (p, v):
            x, m = stream.values.cpu().numpy(), stream.feature_mask.cpu().numpy()
            count = m.sum(1).clip(1)
            mean = (x*m).sum(1)/count
            std = np.sqrt((((x-mean[:, None])*m)**2).sum(1)/count)
            result.append(np.concatenate((mean, std, x[:, -1], m.mean(1)), axis=1))
        return result


def future_query_variants(stream, cutoff=15.):
    future = stream.timestamps_s > cutoff
    changed = replace(stream, values=torch.where(future[:, :, None], torch.full_like(stream.values, 12345), stream.values))
    mask = stream.feature_mask & ~future[:, :, None]
    deleted = replace(stream, values=stream.values.masked_fill(~mask, 0), feature_mask=mask,
                      modality_mask=mask.any(-1), observation_age_s=stream.observation_age_s.masked_fill(~mask, torch.inf),
                      source_indices=stream.source_indices.masked_fill(~mask, -1))
    return changed, deleted
