"""All arms, paired subject differences, and predeclared adoption gates."""
import numpy as np
import pandas as pd
from repair_io import HERE,METHODS,save_csv

TASKS={'workload_standard_classification':'CLARE 标准负荷分类（≥5）',
       'workload_classification':'CLARE 项目较高负荷分类（≥7）','workload_regression':'CLARE 负荷评分回归',
       'difficulty':'CogPilot 难度分类','event_response':'CogPilot 事件响应回归'}
NAMES={'chronaris':'完整方法','constant':'简单参照','physiology_only':'独立第一流','vehicle_only':'独立第二流',
       'naive_time_sync':'朴素时间同步','mult':'MulT 多模态变换器','contiformer':'ContiFormer 连续时间变换器',
       'chronaris_second24':'完整方法内部第二流 24 维','chronaris_pca24':'完整表示主成分压缩 24 维'}
ARMS={'block_selected':'分块正则内层选择／9候选','uniform3':'统一正则／3候选','matched9':'统一正则／9候选',
      'favor_second':'固定第二流优先／3候选','favor_first':'固定第一流优先／3候选','selected':'内层选择／3候选',
      'ridge':'未约束岭回归','ridge_clipped':'同参数限幅至1–9','ridge_bounded_selection':'有界误差内层选参',
      'majority':'训练多数类','mean':'训练均值','median':'训练中位数','zero':'零变化','history':'严格历史趋势'}


def primary_arm(domain,task,method):
    classification='classification' in task or task=='difficulty'
    if classification:return 'block_selected' if method=='chronaris' else 'matched9'
    return 'ridge_bounded_selection' if domain=='clare' else 'ridge'


def main():
    folds=pd.concat([pd.read_csv(HERE/d/'folds.csv') for d in ('clare','cogpilot')],ignore_index=True)
    subjects=pd.concat([pd.read_csv(HERE/d/'subjects.csv',dtype={'subject':str}) for d in ('clare','cogpilot')],ignore_index=True)
    keys=['domain','task','route','method','arm'];summary=[]
    for key,p in folds.groupby(keys,sort=False):
        assert set(p.fold)=={0,1,2}
        summary.append(dict(zip(keys,key))|dict(mean=p.value.mean(),fold_std=float(np.std(p.value)),
            fold_values=p.value.to_list(),parameters=p.parameter.fillna('').to_list(),samples=int(p.samples.sum())))
    save_csv(HERE/'comparison.csv',summary)
    pairs=[];gates=[];s=pd.DataFrame(summary)
    for (domain,task,route),part in subjects.groupby(['domain','task','route']):
        sign=1 if 'classification' in task or task=='difficulty' else -1
        arm=primary_arm(domain,task,'chronaris')
        full=part[(part.method=='chronaris')&(part.arm==arm)].set_index('subject')
        references=[]
        for (method,other_arm),other in part.groupby(['method','arm']):
            if method not in METHODS[1:] and method!='constant':continue
            other=other.set_index('subject').loc[full.index]
            assert np.array_equal(full.fold,other.fold)
            delta=(full.value-other.value)*sign;fd=delta.groupby(full.fold).mean()
            row=dict(domain=domain,task=task,route=route,comparator=method,arm=other_arm,
                main_reference=method=='constant' or other_arm==primary_arm(domain,task,method),
                benefit=float(fd.mean()),subjects_better=int((delta>1e-12).sum()),subjects_worse=int((delta< -1e-12).sum()),
                folds_better=int((fd>1e-12).sum()),mean_pass=bool(fd.mean()>1e-12),
                stable_pass=bool(fd.mean()>1e-12 and (fd>1e-12).sum()>=2 and (delta>1e-12).sum()>len(delta)/2))
            pairs.append(row);references.append(row)
        def passed(predicate,stable=False):
            rows=[r for r in references if predicate(r)];assert rows
            return all(r['stable_pass' if stable else 'mean_pass'] for r in rows)
        mean_simple=passed(lambda r:r['comparator']=='constant')
        mean_single=passed(lambda r:r['main_reference'] and r['comparator'] in ('physiology_only','vehicle_only'))
        stable=passed(lambda r:r['main_reference'] and r['comparator'] in ('constant','physiology_only','vehicle_only'),True)
        gates.append(dict(domain=domain,task=task,route=route,main_arm=arm,mean_beats_all_simple=mean_simple,
            mean_beats_both_single=mean_single,mean_beats_all_main_traditional=passed(lambda r:r['main_reference'] and r['comparator']!='constant'),
            also_beats_all_preserved_control_arms=all(r['mean_pass'] for r in references),
            mean_joint_pass=mean_simple and mean_single,stable_joint_pass=stable,
            legacy_task=task=='workload_classification'))
    save_csv(HERE/'paired_comparisons.csv',pairs);save_csv(HERE/'gates.csv',gates)
    lines=['# 全部读出修复对照','',
        'CLARE 为生理负荷数据，CogPilot 为飞行任务数据。下表采用相同冻结表示和三折受试者隔离读出；分类为宏平均 F1（越高越好），回归为均方根误差（越低越好），每格为折均值 ± 折间总体标准差，不是置信区间。两路线分别报告，标准 ≥5 与项目 ≥7 是不同任务，不能把二者差值当作修复收益。','',
        '完整方法分类主方案为分块正则的9候选选择，传统方法对应9档统一正则；三档结果与固定分块结果是完整保留的控制。CLARE 回归主方案为有界误差内层选参，另列同参数直接限幅以分离作用。CogPilot 回归不设置无来源的界限。独立第一/第二流在 CLARE 指脑电/外周生理，在 CogPilot 指生理/飞行状态。','']
    for task in TASKS:
        part=s[s.task==task]
        lines += ['## '+TASKS[task],'','本任务全部预登记候选如下。训练多数类或常数参照不读取评分受试者标签。','',
                  '| 方法与读出 | 自监督 | 任务引导 |','| --- | ---: | ---: |']
        for method,arm in part[['method','arm']].drop_duplicates().itertuples(index=False,name=None):
            values=[]
            for route in ('self_supervised','task_guided'):
                row=part[(part.method==method)&(part.arm==arm)&(part.route==route)].iloc[0]
                values.append(f"{row['mean']:.5f} ± {row.fold_std:.5f}")
            lines.append('| '+NAMES[method]+'／'+ARMS[arm]+' | '+' | '.join(values)+' |')
        lines.append('')
    (HERE/'COMPARISON.md').write_text('\n'.join(lines))
    print(pd.DataFrame(gates).to_string(index=False))


if __name__=='__main__':main()
