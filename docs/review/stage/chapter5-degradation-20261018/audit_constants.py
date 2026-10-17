"""Post-hoc train-mean/majority controls; no test-fitted values or model changes."""
import csv
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from audit_statistics import DATA,HERE,REPO,digest
from audit_training import grouped


def main():
    sources={};rows=[]
    def track(path):
        path=Path(path)
        if str(path) not in sources:sources[str(path)]=digest(path)
        return path
    with track(DATA/'inventory.csv').open() as f:units=list(csv.DictReader(f))
    for domain in ('clare','cogpilot'):
        target=torch.load(track(REPO/'artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared'/domain/'task_targets.pt'),map_location='cpu',weights_only=True)
        index={s:i for i,s in enumerate(target['sample_ids'])}
        for unit in units:
            if unit['domain']!=domain or unit['variant']!='thesis_reference':continue
            receipt=json.loads(track(unit['receipt']).read_text())
            route=next(x for x in receipt['results'] if x['route']=='task_guided')
            consumer=joblib.load(track(route['consumers']['components']['linear']['model_path']))['consumer']
            positions=[index[s] for s in consumer['train_sample_ids']]
            with gzip.open(track(DATA/'native_predictions'/f"{unit['unit_key']}_task_guided_linear.csv.gz"),'rt') as f:predictions=list(csv.DictReader(f))
            assert not set(consumer['train_sample_ids'])&{r['sample_id'] for r in predictions}
            for task in consumer['definitions']:
                valid=target['valid_masks'][task.name][positions].numpy().reshape(-1)
                ytrain=target['values'][task.name][positions].numpy().reshape(-1)[valid]
                selected=[r for r in predictions if r['task']==task.name and r['target_valid']=='True']
                truth=np.array([float(r['target']) for r in selected]);groups=np.array([r['group_id'] for r in selected])
                classification=task.kind=='classification'
                constant=float(np.bincount(ytrain.astype(int),minlength=task.output_dim).argmax()) if classification else float(ytrain.mean())
                prediction=np.full(len(truth),constant)
                classes=list(range(task.output_dim)) if classification else None
                full=np.array([float(r['prediction']) for r in selected])
                rows.append(dict(domain=domain,fold=unit['fold_index'],task=task.name,
                    baseline='train_majority' if classification else 'train_mean',train_constant=constant,
                    held_out_value=grouped(truth,prediction,groups,classification,classes),
                    full_task_guided_linear_value=grouped(truth,full,groups,classification,classes),
                    train_valid_samples=len(ytrain),held_out_subjects=len(set(groups))))
    with (HERE/'statistics/constant_baselines.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for path,expected in sources.items():assert digest(Path(path))==expected
    (HERE/'statistics/constant_verification.json').write_text(json.dumps(dict(sources=sources,
        scope='post_hoc_fixed_train_only_constants_no_model_selection',original_files_unchanged=True),indent=2)+'\n')
    print('constant baseline checks',len(rows))


if __name__=='__main__':main()
