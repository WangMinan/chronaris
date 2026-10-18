"""Explicit, versioned public-task views; historical preparation remains immutable."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from chronaris.dataset.lazy_observed import LazyObservedDataset
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

FORMAT = 'chronaris.public_readout_revision.v1'
STANDARD_TASK = 'workload_standard_classification'
ENDPOINT_VERSION = 'exclusive_event_endpoint_v1'


def event_endpoint_s(row):
    if row['native_time_scale_s'] != 86400.:
        raise ValueError('event endpoint requires the recorded MATLAB date coordinate')
    start = row['record_time_origin_native'] + row['context_start_s'] / 86400.
    return ((start + 12. / 86400.) - start) * 86400.


def revised_observation_hash(row):
    if not row['task_valid_mask'].get('event_response'):
        return row['source_sample_hash']
    return hashlib.sha256((row['source_sample_hash'] + ':' + ENDPOINT_VERSION).encode()).hexdigest()


def revise_public_observation(sample, row):
    if sample.sample_id != row['sample_id'] or sample.source_sample_hash != row['source_sample_hash']:
        raise ValueError('revision source observation identity differs from manifest')
    if not row['task_valid_mask'].get('event_response'):
        return sample
    if sample.context_duration_s != 12.:
        raise ValueError('event revision requires the frozen 12-second window')
    cutoff = event_endpoint_s(row)
    updates = {'source_sample_hash': revised_observation_hash(row)}
    for stream in ('physiology', 'vehicle'):
        keep = getattr(sample, stream + '_timestamps_s') < cutoff
        for field in ('timestamps_s', 'values', 'feature_mask'):
            name = stream + '_' + field
            updates[name] = getattr(sample, name)[keep].copy()
    return replace(sample, **updates)


def revised_public_data(domain, *, source_root, registry_path):
    source = Path(source_root) / domain
    if not (source / 'prepared_cache_index.json').is_file():
        raise ValueError('source preparation must already be indexed; revision never writes sources')
    data = load_prepared_public_development(domain, output_root=source_root, registry_path=registry_path)
    values = {k: v.clone() for k, v in data.targets.values.items()}
    masks = {k: v.clone() for k, v in data.targets.valid_masks.items()}
    definitions = data.task_definitions
    if domain == 'clare':
        scores = values['workload_regression']
        valid = masks['workload_regression']
        if not bool(((scores[valid] >= 1) & (scores[valid] <= 9)).all()):
            raise ValueError('CLARE scores exceed the documented 1-9 scale')
        values[STANDARD_TASK] = (scores >= 5).long()
        masks[STANDARD_TASK] = valid.clone()
        definitions += (ApplicationTaskDefinition(STANDARD_TASK, 'classification', 2),)
    elif domain != 'cogpilot':
        raise ValueError('unsupported public task revision')
    original_rows = {r['sample_id']: r for r in data.sample_manifest}
    rows = tuple(r | {'source_sample_hash': revised_observation_hash(r),
                     'task_valid_mask': r['task_valid_mask'] | ({STANDARD_TASK: r['task_valid_mask']['workload_regression']} if domain == 'clare' else {})}
                 for r in data.sample_manifest)
    records = tuple(replace(r, source_sample_hash=revised_observation_hash(original_rows[r.sample_id])) for r in data.dataset.records)
    source_dataset = data.dataset
    def loader(record):
        return revise_public_observation(source_dataset.load_sample(record.sample_id), original_rows[record.sample_id])
    dataset = LazyObservedDataset(records, schema=data.dataset.schema, loader=loader)
    targets = replace(data.targets, values=values, valid_masks=masks,
                      manifest=dict(data.targets.manifest) | {'revision_format': FORMAT})
    return replace(data, dataset=dataset, targets=targets, task_definitions=definitions, sample_manifest=rows)


def revision_sources(domain, source_root, registry_path):
    root = Path(source_root) / domain
    paths = [root / n for n in ('summary.json', 'sample_manifest.jsonl', 'task_targets.pt', 'prepared_cache_index.json')]
    paths += [Path(registry_path), Path(__file__)]
    return {str(p.resolve()): sha256_file(p) for p in paths}


def prepare_readout_revision(domain, *, source_root, registry_path, output_root):
    root = Path(output_root) / domain
    if root.resolve().is_relative_to(Path(source_root).resolve()):
        raise ValueError('revision output must be outside frozen source preparation')
    if root.exists():
        raise FileExistsError('revision output exists; verify and load it rather than overwriting')
    data = revised_public_data(domain, source_root=source_root, registry_path=registry_path)
    sources = revision_sources(domain, source_root, registry_path)
    root.mkdir(parents=True)
    torch.save(dict(sample_ids=data.targets.sample_ids, values=data.targets.values,
                    valid_masks=data.targets.valid_masks, manifest=data.targets.manifest), root / 'task_targets.pt')
    (root / 'sample_manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data.sample_manifest))
    payload = dict(format=FORMAT, domain=domain, source_root=str(Path(source_root).resolve()),
        registry_path=str(Path(registry_path).resolve()), sources=sources,
        files={n: sha256_file(root / n) for n in ('task_targets.pt', 'sample_manifest.jsonl')},
        observations='lazy immutable source view; no source cache writes',
        task_definitions=[dict(name=d.name, kind=d.kind, output_dim=d.output_dim) for d in data.task_definitions],
        standard_threshold=5 if domain == 'clare' else None, historical_threshold=7 if domain == 'clare' else None,
        endpoint_version=ENDPOINT_VERSION if domain == 'cogpilot' else None,
        encoder_updates=0)
    (root / 'revision_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def load_readout_revision(root):
    root = Path(root)
    payload = json.loads((root / 'revision_manifest.json').read_text())
    if payload['format'] != FORMAT:
        raise ValueError('unsupported public revision format')
    for name, expected in payload['sources'].items():
        if sha256_file(name) != expected:
            raise ValueError('revision source changed: ' + name)
    for name, expected in payload['files'].items():
        if sha256_file(root / name) != expected:
            raise ValueError('revision artifact changed: ' + name)
    data = revised_public_data(payload['domain'], source_root=payload['source_root'], registry_path=payload['registry_path'])
    saved = torch.load(root / 'task_targets.pt', weights_only=True)
    if data.targets.sample_ids != saved['sample_ids'] or any(not torch.equal(v, saved['values'][k]) for k, v in data.targets.values.items()):
        raise ValueError('revision target reconstruction differs')
    return replace(data, prepared_manifest_sha256=sha256_file(root / 'revision_manifest.json'))
