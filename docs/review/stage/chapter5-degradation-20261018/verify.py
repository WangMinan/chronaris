"""Verify preserved source files and diagnostic claims; never run training."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    sources={}
    for name in ('statistics','training','readouts'):
        data=json.loads((HERE/name/'verification.json').read_text())
        for key,value in data['sources'].items():
            path=Path(key) if Path(key).is_absolute() else REPO/key
            assert str(path) not in sources or sources[str(path)]==value
            sources[str(path)]=value
    constants=json.loads((HERE/'statistics/constant_verification.json').read_text())
    for key,value in constants['sources'].items():
        assert key not in sources or sources[key]==value
        sources[key]=value
    revision=json.loads((HERE/'revision/verification.json').read_text())
    assert revision['model_fits']==0 and revision['encoder_updates']==0
    for key,value in revision['sources'].items():
        assert key not in sources or sources[key]==value
        sources[key]=value
    with (HERE/'revision/validation_summary.csv').open() as f:
        corrected=next(r for r in csv.DictReader(f) if r['domain']=='cogpilot')
    assert float(corrected['train_mean_rmse'])<float(corrected['strong_ridge_rmse'])
    assert int(corrected['train_mean_better_than_strong_folds'])==4
    forward=json.loads((HERE/'forward/summary.json').read_text())
    sources.update(forward['checkpoint_hashes'])
    assert forward['optimizer_updates']==0 and forward['consumer_fits']==0
    assert len(forward['records'])==10
    assert sum(len(x['result']['batches']) for x in forward['gradients'])==12
    assert all(x['result']['optimizer_steps']==0 for x in forward['gradients'])
    for key,expected in sources.items():assert digest(Path(key))==expected,key
    stats=json.loads((HERE/'statistics/verification.json').read_text())
    assert stats['public_prediction_rows']==740658
    assert stats['recomputed_group_metrics']==2644 and stats['max_metric_recomputation_error']==0
    train=json.loads((HERE/'training/verification.json').read_text())
    assert train['audited_training_routes']==130 and train['within_fold_route_budgets_and_counts_equal']
    readouts=json.loads((HERE/'readouts/verification.json').read_text())
    assert len(readouts['full_feature_refit_checks'])==40
    assert max(x['delta'] for x in readouts['full_feature_refit_checks'])<1e-7
    # A constant positive gate is canceled by per-column train standardization.
    x=np.arange(30,dtype=float).reshape(10,3)**1.3
    a=(x-x.mean(0))/x.std(0);g=.018
    b=(g*x-(g*x).mean(0))/(g*x).std(0)
    assert np.allclose(a,b,atol=1e-12)
    result=dict(status='passed',original_files_rehashed=len(sources),frozen_files_unchanged=True,
        public_metric_recomputation='exact',full_feature_refits=40,encoder_optimizer_steps=0,
        bounded_linear_probe_fits=readouts['independent_selected_probe_heads'],probe_models_saved=False,
        revision_no_fits=True,revision_strong_ridge_still_worse_than_train_mean=True,
        constant_gate_standardization_identity_max_error=float(np.max(np.abs(a-b))),
        cosine_attention_max_weight_ratio=float(np.exp(2/np.sqrt(32))))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
