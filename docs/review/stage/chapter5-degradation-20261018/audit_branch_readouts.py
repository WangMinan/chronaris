"""Independent small linear probes on frozen features, train/validation only."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from audit_statistics import DATA,HERE,digest
from audit_training import grouped
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_classifier,fit_regressor

OUT=HERE/'readouts'
ROOT=DATA.parents[4]/'artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared'


def main():
    OUT.mkdir(exist_ok=True);sources={};rows=[];checks=[]
    def track(path):
        path=Path(path)
        if str(path) not in sources:sources[str(path)]=digest(path)
        return path
    with track(DATA/'inventory.csv').open() as f:units=list(csv.DictReader(f))
    for domain in ('clare','cogpilot'):
        target_path=track(ROOT/domain/'task_targets.pt')
        target=torch.load(target_path,map_location='cpu',weights_only=True);index={s:i for i,s in enumerate(target['sample_ids'])}
        for unit in units:
            if unit['domain']!=domain or unit['variant']!='thesis_reference':continue
            receipt=json.loads(track(unit['receipt']).read_text())
            for r in receipt['results']:
                component=r['consumers']['components']['linear'];path=track(component['model_path'])
                frozen=joblib.load(path)['consumer'];features={};ids={};groups={}
                for role in ('train','validation'):
                    with np.load(track(path.parents[2]/'representations'/role/'fusion_stream.npz')) as a:
                        features[role]=a['pooled_embedding'];ids[role]=list(a['sample_ids'])
                    groups[role]=np.array([frozen['context']['groups'][s] for s in ids[role]])
                assert tuple(ids['train'])==tuple(frozen['train_sample_ids'])
                assert not set(groups['train'])&set(groups['validation'])
                for task in frozen['definitions']:
                    is_class=task.kind=='classification';classes=list(range(task.output_dim)) if is_class else None
                    values={};mask={}
                    for role in ids:
                        positions=[index[s] for s in ids[role]]
                        mask[role]=target['valid_masks'][task.name][positions].numpy().reshape(-1)
                        values[role]=target['values'][task.name][positions].numpy().reshape(-1)[mask[role]]
                    probes=[('full',slice(None)),('physiology',slice(0,24)),('vehicle',slice(24,48)),
                            ('cross',slice(48,64)),('branches',slice(0,48))]
                    if domain=='cogpilot' and not is_class:
                        probes += [('full_alpha1000',slice(None)),('full_alpha10000',slice(None))]
                    for name,columns in probes:
                        x={role:features[role][mask[role],columns] for role in ids}
                        kwargs=dict(scaler_with_mean=True,validation_groups=groups['validation'][mask['validation']])
                        if is_class:
                            model,param=fit_classifier(x['train'],values['train'],x['validation'],values['validation'],
                                c_values=frozen['config']['classification_c_grid'],random_state=17,classification_labels=classes,**kwargs)
                        else:
                            alphas=(float(name.removeprefix('full_alpha')),) if name.startswith('full_alpha') else frozen['config']['regression_alpha_grid']
                            model,param=fit_regressor(x['train'],values['train'],x['validation'],values['validation'],
                                alpha_values=alphas,**kwargs)
                        pred=model.predict(x['validation'])
                        if name=='full':
                            old=frozen['models'][(task.name,0)].predict(x['validation'])
                            delta=float(np.max(np.abs(pred-old)));assert delta<1e-7
                            checks.append(dict(domain=domain,fold=unit['fold_index'],route=r['route'],task=task.name,delta=delta))
                        for role in ('train','validation'):
                            value=grouped(values[role],model.predict(x[role]),groups[role][mask[role]],is_class,classes)
                            rows.append(dict(domain=domain,fold=unit['fold_index'],route=r['route'],task=task.name,
                                block=name,role=role,value=value,selected_parameter=param,dimensions=x[role].shape[1],
                                samples=len(x[role]),subjects=len(set(groups[role][mask[role]]))))
                        del model
                print('probed',domain,unit['fold_index'],r['route'],flush=True)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for path,expected in sources.items():assert digest(Path(path))==expected
    (OUT/'verification.json').write_text(json.dumps(dict(sources=sources,encoder_updates=0,
        original_consumers_unchanged=True,independent_probe_models_persisted=False,roles=['train','validation'],
        independent_selected_probe_heads=len(rows)//2,
        full_feature_refit_checks=checks,
        scope='post_hoc_existing_grid_branch_probes_plus_two_fixed_CogPilot_ridge_controls_not_new_main_result'),indent=2)+'\n')


if __name__=='__main__':main()
