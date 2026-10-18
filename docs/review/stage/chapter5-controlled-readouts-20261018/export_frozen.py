"""Export fixed first-fold models onto disjoint registered development subjects."""
import csv
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from common import HERE, PREP, REGISTRY, ASSETS, INVENTORY, sha, save_json
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.v4_naive_baseline import load_v4_naive_encoder
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.training import TrainedFusionAdapter
from chronaris.modeling.fusion_encoders import NaiveTimeSyncFusionAdapter
from chronaris.representation import FoldLineage
from chronaris.representation import collate_observation_samples


def fold_from_dict(row):
    return FoldLineage(**{key: tuple(row[key]) if key.endswith('_ids') else row[key] for key in
        ('fold_id', 'train_sample_ids', 'validation_sample_ids', 'held_out_sample_ids', 'development_only')})


def exclusive_sample(sample, row):
    if not row['task_valid_mask'].get('event_response'):
        return sample
    start=row['record_time_origin_native']+row['context_start_s']/86400
    cutoff=((start+12/86400)-start)*86400
    updates={}
    for stream in ('physiology','vehicle'):
        keep=getattr(sample,stream+'_timestamps_s') < cutoff
        for field in ('timestamps_s','values','feature_mask'):
            name=stream+'_'+field
            updates[name]=getattr(sample,name)[keep].copy()
    # This is an explicitly derived view; never reuse the original observation identity.
    import hashlib
    updates['source_sample_hash']=hashlib.sha256((sample.source_sample_hash+':exclusive_event_endpoint_v1').encode()).hexdigest()
    return replace(sample,**updates)


def main(exclusive=False):
    assert json.loads((HERE/'raw_verification.json').read_text())['status'] == 'passed'
    torch.set_num_threads(1)
    began = time.monotonic()
    with INVENTORY.open() as f:
        units = [r for r in csv.DictReader(f) if r['domain'] in ('clare', 'cogpilot') and r['fold_index']=='0'
                 and r['seed']=='17' and ((r['method']=='chronaris' and r['variant']=='thesis_reference')
                 or r['method'] in ('physiology_only', 'vehicle_only', 'naive_time_sync', 'mult', 'contiformer'))]
    assert len(units) == 12
    units.sort(key=lambda r: (r['domain'], r['method'] != 'chronaris', r['method']))
    records = []
    asset_root=ASSETS/'exclusive_endpoint' if exclusive else ASSETS
    asset_root.mkdir(parents=True,exist_ok=True)
    with development_gpu_lock() as acquired:
        if not acquired:
            raise RuntimeError('GPU locked: no preemption or fallback')
        for domain in ('clare', 'cogpilot'):
            if exclusive and domain!='cogpilot':continue
            assert (PREP/domain/'prepared_cache_index.json').exists()
            data = load_prepared_public_development(domain, output_root=PREP, registry_path=REGISTRY)
            ids = data.dataset.sample_ids
            groups = set(data.dataset.group_ids)
            by_id={r['sample_id']:r for r in data.sample_manifest}
            def provider(batch_ids):
                if not exclusive:return data.dataset.batch_provider(batch_ids)
                return collate_observation_samples([exclusive_sample(data.dataset.load_sample(s),by_id[s]) for s in batch_ids])
            for unit in [r for r in units if r['domain']==domain]:
                receipt = json.loads(Path(unit['receipt']).read_text())
                for result in receipt['results']:
                    method, route = unit['method'], result['route']
                    out = asset_root/f'{domain}__{method}__{route}.npz'
                    proof_path = out.with_suffix('.json')
                    training = result['training']
                    if method == 'naive_time_sync':
                        checkpoint = Path(result['consumers']['components']['linear']['model_path']).parents[3]/'checkpoint/best.pt'
                        fold = fold_from_dict(training['fold'])
                    else:
                        checkpoint = Path(training['best_checkpoint_path'])
                        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                        parent = payload if route == 'self_supervised' else torch.load(payload['source_checkpoint_path'], map_location='cpu', weights_only=True)
                        fold = fold_from_dict(parent['fold'])
                    old_ids = fold.train_sample_ids+fold.validation_sample_ids+fold.held_out_sample_ids
                    old_groups = {s.split('__')[0] if domain=='clare' else s.split('::')[0] for s in old_ids}
                    assert not groups & old_groups and not set(ids) & set(old_ids)
                    checkpoint_hash = sha(checkpoint)
                    if proof_path.exists():
                        proof = json.loads(proof_path.read_text())
                        assert proof['checkpoint_sha256']==checkpoint_hash and sha(out)==proof['output_sha256']
                        records.append(proof)
                        continue
                    assert not out.exists(), 'Uncommitted output retained; inspect before rerunning'
                    if time.monotonic()-began > 7200:
                        raise TimeoutError('Frozen export budget exceeded; completed results retained')
                    if method=='naive_time_sync':
                        encoder, normalizer, _ = load_v4_naive_encoder(checkpoint, fold=fold,
                            data_manifest_sha256=training['config']['data_manifest_sha256'],
                            expected_training_source_sha256=training['source_code_sha256'])
                        adapter = NaiveTimeSyncFusionAdapter(encoder=encoder, fold_id=fold.fold_id, checkpoint_sha256=checkpoint_hash)
                    else:
                        encoder, normalizer, _ = load_frozen_application_encoder(checkpoint, route=route, fold=fold, device='cuda')
                        encoder.requires_grad_(False)
                        adapter = TrainedFusionAdapter(encoder=encoder, normalizer=normalizer, fold_id=fold.fold_id, checkpoint_sha256=checkpoint_hash)
                    start = time.monotonic()
                    chunks = []
                    print('start', domain, method, route, len(ids), flush=True)
                    for i in range(0, len(ids), 4):
                        if time.monotonic()-began > 7200:
                            raise TimeoutError('Frozen export budget exceeded')
                        output = adapter(provider(ids[i:i+4]))
                        assert output.sample_ids == ids[i:i+4]
                        chunks.append(output.pooled_embedding.detach().cpu().numpy())
                        if i % 128 == 0:
                            print('progress', domain, method, route, i, round(time.monotonic()-start, 1), flush=True)
                    pooled = np.concatenate(chunks)
                    assert pooled.shape == (len(ids), 64) and np.isfinite(pooled).all()
                    assert sha(checkpoint) == checkpoint_hash
                    np.savez_compressed(out, sample_ids=np.array(ids), pooled=pooled)
                    proof = dict(domain=domain, method=method, route=route, checkpoint=str(checkpoint),
                        checkpoint_sha256=checkpoint_hash, output=str(out), output_sha256=sha(out),
                        receipt=unit['receipt'], receipt_sha256=sha(unit['receipt']), original_fold_id=fold.fold_id,
                        original_subjects=sorted(old_groups), development_subjects=sorted(groups), disjoint_subjects=True,
                        encoder_updates=0, samples=len(ids), dimensions=64, seconds=time.monotonic()-start,
                        input_view='exclusive_event_endpoint_v1' if exclusive else 'original_frozen_input',
                        normalizer_refitted=False, prepared_manifest_sha256=data.prepared_manifest_sha256)
                    save_json(proof_path, proof)
                    records.append(proof)
                    save_json(HERE/('export_boundary_progress.json' if exclusive else 'export_progress.json'), records)
                    del adapter, encoder, output, chunks
                    torch.cuda.empty_cache()
            del data
    save_json(HERE/('export_boundary_verification.json' if exclusive else 'export_verification.json'), dict(status='completed', records=records, encoder_updates=0,
        registry_sha256=sha(REGISTRY), export_script_sha256=sha(HERE/'export_frozen.py')))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--exclusive-boundary',action='store_true')
    main(p.parse_args().exclusive_boundary)
