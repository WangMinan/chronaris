"""Recompute comparison tables from per-subject and per-fold diagnostic outputs."""
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import HERE, ASSETS, REGISTRY, manifest, save_json, save_csv, sha

NAMES={'chronaris':'完整方法','physiology_only':'独立第一流','vehicle_only':'独立第二流',
       'mult':'MulT 多模态变换器','contiformer':'ContiFormer 连续时间变换器','naive_time_sync':'朴素时间同步',
       'constant':'简单参照'}
VARIANTS={'full64':'64 维（扩展正则）','full64_original_grid':'64 维（原正则网格）','first24':'第一流 24 维',
          'second24':'第二流 24 维','cross16':'跨流 16 维','branches48':'两分支 48 维',
          'pca16':'主成分压缩 16 维','pca24':'主成分压缩 24 维','pca48':'主成分压缩 48 维',
          'rp24_17':'随机投影 24 维／种子 17','rp24_29':'随机投影 24 维／种子 29','rp24_43':'随机投影 24 维／种子 43',
          'train_mean':'训练均值','train_median':'训练中位数','train_majority':'训练多数类',
          'zero_change':'零变化（持久性）','history_trend':'历史两秒变化'}


def main():
    folds=pd.concat([pd.read_csv(HERE/f'{d}_folds.csv') for d in ('clare','cogpilot')],ignore_index=True)
    subjects=pd.concat([pd.read_csv(HERE/f'{d}_subjects.csv') for d in ('clare','cogpilot')],ignore_index=True)
    keys=['domain','task','route','method','variant']
    summary=[]
    for key,part in folds.groupby(keys,sort=False):
        assert set(part.fold)=={0,1,2}
        meta=dict(zip(keys,key));g=subjects
        for k,v in meta.items():g=g[g[k]==v]
        assert not g.subject.duplicated().any()
        for f,fp in g.groupby('fold'):
            assert abs(fp.value.mean()-part[part.fold==f].value.iloc[0])<1e-12
        summary.append(meta|dict(mean=part.value.mean(),fold_std=float(np.std(part.value)),folds=3,
            subjects=len(g),samples=int(g.samples.sum()),parameters=json.dumps(part.parameter.fillna('').tolist()),
            fold_values=json.dumps(part.value.tolist())))
    save_csv(HERE/'comparison.csv',summary)
    pairs=[]
    for (domain,task,route),part in subjects.groupby(['domain','task','route']):
        full=part[(part.method=='chronaris')&(part.variant=='full64')].set_index('subject')
        sign=1 if task in ('difficulty','workload_classification') else -1
        for (method,variant),other in part.groupby(['method','variant']):
            other=other.set_index('subject').loc[full.index]
            assert np.array_equal(full.fold,other.fold)
            delta=(full.value-other.value)*sign
            by_fold=delta.groupby(full.fold).mean()
            pairs.append(dict(domain=domain,task=task,route=route,comparator=method,variant=variant,
                full_benefit=by_fold.mean(),subjects_better=int((delta>1e-12).sum()),subjects_worse=int((delta< -1e-12).sum()),
                folds_better=int((by_fold>1e-12).sum()),folds_worse=int((by_fold< -1e-12).sum())))
    save_csv(HERE/'paired_comparison.csv',pairs)
    lines=['# 原窗口完整对照表','',
        '**本表保留原输入诊断。CogPilot 事件回归的严格端点语义存在 28 窗目标区间重叠，相关成绩不作为严格历史输入验收结果；新增同规则输入修正对照见 [严格事件端点对照](BOUNDARY_COMPARISON.md)。分类窗口不受该问题影响。**','',
        '下表比较固定编码器在登记开发受试者上的三折嵌套读出。分类为宏平均 F1（越高越好），回归为均方根误差 RMSE（越低越好）；每格为折均值 ± 折间总体标准差，不是置信区间。先逐受试者评分、再折内和折间等权。CLARE 生理负荷数据为 4 人／794 窗，CogPilot 飞行任务数据为 7 人／2524 窗（两任务有效样本分开）。两个训练路线分别为自监督和任务引导。','',
        '第一流／第二流在 CLARE 指脑电／外周生理，在 CogPilot 指生理／飞行状态。独立单流使用原冻结的独立骨干，不能据此单独归因于融合。朴素时间同步未使用标签训练编码器，两表共享该固定表示。原正则网格上限为 100，扩展到 10000；分类网格相同。所有候选均保留，随机投影不挑最佳种子。','']
    df=pd.DataFrame(summary)
    for domain in ('clare','cogpilot'):
        tasks=('workload_classification','workload_regression') if domain=='clare' else ('difficulty','event_response')
        for route in ('self_supervised','task_guided'):
            lines += [f"## {domain.upper() if domain=='clare' else 'CogPilot'}／{'自监督' if route=='self_supervised' else '任务引导'}",'',
                      '对应数据域和训练路线的全部候选如下；“—”表示该参照不适用该任务。','',
                      '| 表示或参照 | 分类宏平均 F1 ↑ | 回归 RMSE ↓ |','| --- | ---: | ---: |']
            part=df[(df.domain==domain)&(df.route==route)]
            for method,variant in part[['method','variant']].drop_duplicates().itertuples(index=False,name=None):
                values=[]
                for task in tasks:
                    r=part[(part.method==method)&(part.variant==variant)&(part.task==task)]
                    values.append(f"{r.iloc[0]['mean']:.5f} ± {r.iloc[0]['fold_std']:.5f}" if len(r) else '—')
                label=(NAMES[method]+'／' if method!='constant' else '')+VARIANTS[variant]
                lines.append('| '+label+' | '+' | '.join(values)+' |')
            lines.append('')
    lines += ['逐折参数及样本数见 [comparison.csv](comparison.csv)；逐人胜负见 [paired_comparison.csv](paired_comparison.csv)，逐人偏差、预测方差和相关性见两域的 `*_subjects.csv`。原五折内部验证另见 [original_validation_folds.csv](original_validation_folds.csv)，不得与本表混作同一总体。','']
    original=pd.read_csv(HERE/'original_validation_folds.csv')
    old_summary=[]
    for key,p in original.groupby(['domain','task','route','method'],sort=False):
        old_summary.append(dict(zip(['domain','task','route','method'],key))|dict(
            mean=float(p.value.mean()) if p.value.notna().sum()==5 else '',
            completed_folds=int(p.value.notna().sum()),required_folds=5))
    save_csv(HERE/'original_validation_summary.csv',old_summary)
    (HERE/'COMPARISON.md').write_text('\n'.join(lines))
    # No fitting: inspect feature scale shifts using only outer-training statistics.
    feature_rows=[]
    registry=json.loads(REGISTRY.read_text())
    for record in json.loads((HERE/'export_verification.json').read_text())['records']:
        domain=record['domain'];rows=manifest(domain);groups=np.array([r['subject_id'] for r in rows])
        with np.load(record['output']) as z:x=z['pooled'].astype(float)
        for fi,fold in enumerate(registry['domains'][domain]['folds']['development']):
            a=x[np.isin(groups,fold['train_subjects'])];b=x[np.isin(groups,fold['validation_subjects'])]
            scaler=StandardScaler().fit(a);ta=scaler.transform(a);tb=scaler.transform(b)
            eig=np.linalg.svd(ta,compute_uv=False)**2
            feature_rows.append(dict(domain=domain,method=record['method'],route=record['route'],fold=fi,
                effective_rank=float(eig.sum()**2/(eig@eig)),max_validation_abs_z=float(abs(tb).max()),
                mean_validation_abs_z=float(abs(tb).mean()),
                outside_training_range_fraction=float(((b<a.min(0))|(b>a.max(0))).mean())))
    save_csv(HERE/'feature_shift.csv',feature_rows)
    out_of_range=[]
    pred=pd.read_csv(HERE/'clare_predictions.csv.gz')
    for key,p in pred[pred.task=='workload_regression'].groupby(['route','method','variant']):
        out_of_range.append(dict(zip(['route','method','variant'],key))|dict(samples=len(p),
            predictions_outside_1_to_9=int(((p.prediction<1)|(p.prediction>9)).sum()),
            minimum_prediction=p.prediction.min(),maximum_prediction=p.prediction.max()))
    save_csv(HERE/'clare_prediction_range.csv',out_of_range)
    print('summary',len(summary),'comparisons',len(pairs),'folds',len(folds),'subject scores',len(subjects))


if __name__=='__main__':main()
