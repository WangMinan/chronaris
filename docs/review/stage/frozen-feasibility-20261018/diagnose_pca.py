"""Explain the registered PCA arm's scaling using only its original training folds."""
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from audit_io import HERE,REVISED,REGISTRY,feature_path,save_csv
from chronaris.evaluation.application_tasks.public_readout_revision import load_readout_revision


def main():
    rows=[]
    for domain in ('clare','cogpilot'):
        data=load_readout_revision(REVISED/domain);groups=np.array(data.dataset.group_ids)
        task='workload_regression' if domain=='clare' else 'event_response'
        valid=data.targets.valid_masks[task].numpy().reshape(-1)
        folds=json.loads(REGISTRY.read_text())['domains'][domain]['folds']['development']
        for route in ('self_supervised','task_guided'):
            blocks=[]
            for method in ('physiology_only','vehicle_only'):
                with np.load(feature_path(domain,method,route)) as z:blocks.append(z['pooled'])
            x=np.concatenate(blocks,axis=1).astype(float)
            for i,fold in enumerate(folds):
                train=valid&np.isin(groups,fold['train_subjects']);test=valid&np.isin(groups,fold['validation_subjects'])
                scale=StandardScaler().fit(x[train]);a,b=scale.transform(x[train]),scale.transform(x[test])
                pca=PCA(64,svd_solver='full').fit(a);aa,bb=pca.transform(a),pca.transform(b)
                z=StandardScaler().fit(aa).transform(bb)
                rows.append(dict(domain=domain,task=task,route=route,fold=i,
                    smallest_retained_singular=float(pca.singular_values_[-1]),largest_singular=float(pca.singular_values_[0]),
                    min_training_component_std=float(aa.std(axis=0).min()),max_scoring_standardized_component=float(abs(z).max()),
                    pca_variance_retained=float(pca.explained_variance_ratio_.sum())))
    save_csv(HERE/'pca_diagnostics.csv',rows)


if __name__=='__main__':main()
