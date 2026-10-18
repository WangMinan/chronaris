"""Write all arms, preserved earlier controls, and preregistered development gates."""
import json
import numpy as np
import pandas as pd
from audit_io import HERE,MATERIALS,save_json


def markdown_table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.5f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    summary=[];gates=[];pairs=[];controls=[]
    for domain in ('clare','cogpilot'):
        root=HERE/('complementarity_'+domain);f=pd.read_csv(root/'folds.csv');s=pd.read_csv(root/'subjects.csv')
        for keys,d in f.groupby(['domain','task','route','method','rule']):
            summary.append(dict(zip(['domain','task','route','method','rule'],keys))|dict(mean=float(d.value.mean()),
                fold_std=float(d.value.std(ddof=0)),folds=json.dumps(d.sort_values('fold').value.tolist())))
        for (task,route),all_folds in f.groupby(['task','route']):
            classes=task=='difficulty' or 'classification' in task;rule='bounded' if domain=='clare' and not classes else 'raw'
            baseline_names=['majority'] if classes else ['mean','median']+(['zero'] if task=='event_response' else [])
            refs=[(x,'constant') for x in baseline_names]+[('first',rule),('second',rule)]
            for method in ('concat128','concat_pca64','equal_prediction','selected_prediction'):
                base=dict(domain=domain,task=task,route=route,method=method,rule=rule)
                candidate=all_folds[(all_folds.method==method)&(all_folds.rule==rule)].set_index('fold').value
                cs=s[(s.task==task)&(s.route==route)&(s.method==method)&(s.rule==rule)].set_index(['fold','subject']).value
                comparisons=[]
                for ref,rr in refs:
                    reference=all_folds[(all_folds.method==ref)&(all_folds.rule==rr)].set_index('fold').value
                    rs=s[(s.task==task)&(s.route==route)&(s.method==ref)&(s.rule==rr)].set_index(['fold','subject']).value
                    direction=1 if classes else -1
                    delta=(candidate-reference)*direction;sd=(cs-rs)*direction
                    row=base|dict(reference=ref,mean_gain=float(delta.mean()),fold_wins=int((delta>1e-12).sum()),
                        subject_wins=int((sd>1e-12).sum()),subjects=len(sd),mean_pass=bool(delta.mean()>1e-12),
                        stable_pass=bool(delta.mean()>1e-12 and (delta>1e-12).sum()>=2 and (sd>1e-12).sum()>len(sd)/2))
                    pairs.append(row);comparisons.append(row)
                gates.append(base|dict(primary_task=task!='workload_classification',
                    exploratory=method=='selected_prediction',joint_mean_pass=all(r['mean_pass'] for r in comparisons),
                    joint_stable_pass=all(r['stable_pass'] for r in comparisons)))
        old=pd.read_csv(HERE.parent/'chapter5-readout-repair-20261018'/domain/'folds.csv')
        name_map={'first':'physiology_only','second':'vehicle_only','full':'chronaris'}
        for method,oldmethod in name_map.items():
            for keys,d in f[f.method==method].groupby(['task','route','rule']):
                task,route,rule=keys;classes=task=='difficulty' or 'classification' in task
                arm='uniform3' if classes else {'raw':'ridge','raw_clipped':'ridge_clipped','bounded':'ridge_bounded_selection'}[rule]
                ref=old[(old.task==task)&(old.route==route)&(old.method==oldmethod)&(old.arm==arm)]
                assert len(ref)==len(d)==3
                delta=np.max(abs(d.sort_values('fold').value.to_numpy()-ref.sort_values('fold').value.to_numpy()))
                assert delta<1e-10
                controls.append(dict(domain=domain,task=task,route=route,method=method,rule=rule,max_difference=float(delta)))
    summaries=pd.DataFrame(summary);summaries.to_csv(HERE/'complementarity_comparison.csv',index=False)
    pd.DataFrame(gates).to_csv(HERE/'complementarity_gates.csv',index=False)
    pd.DataFrame(pairs).to_csv(HERE/'complementarity_pairs.csv',index=False)
    pd.DataFrame(controls).to_csv(HERE/'unchanged_controls.csv',index=False)
    old=pd.read_csv(HERE.parent/'chapter5-readout-repair-20261018/comparison.csv')
    old.to_csv(HERE/'preserved_readout_controls.csv',index=False)
    names={'first':'第一流','second':'第二流','full':'完整表示（统一正则）','concat128':'双流拼接128维',
        'concat_pca64':'拼接压缩64维','equal_prediction':'固定均权预测融合','selected_prediction':'训练内选权重（探索）',
        'majority':'多数类','mean':'训练均值','median':'训练中位数','zero':'零响应'}
    tasks={'workload_standard_classification':'CLARE 标准负荷分类（≥5）','workload_classification':'CLARE 历史分类（≥7）',
        'workload_regression':'CLARE 评分回归','difficulty':'CogPilot 难度分类','event_response':'CogPilot 事件响应回归'}
    lines=['# 冻结表示完整对照表','',
        '本表沿用登记开发三折，先逐受试者评分、折内等权、再折间等权；分类为宏平均 F1，越高越好，回归为均方根误差，越低越好。CLARE 第一流为脑电、第二流为外周生理；CogPilot 第一流为生理、第二流为飞机状态。',
        '', '新拼接、压缩和固定预测融合与单流使用相同基础读出网格。训练内选权重额外比较五个权重，单列探索；全部参数与逐人预测保留。完整表示本轮列统一正则，上一轮分块正则等更强修复对照在后续保留表中逐项列出。','']
    for (domain,task,route),d in summaries.groupby(['domain','task','route']):
        lines += ['## '+tasks[task]+'／'+('自监督' if route=='self_supervised' else '任务引导'),'',
            '下表同时保留原始回归、有界选参以及同参数限幅，防止只报告有利处理；分类使用原始类别预测。','']
        view=d[['method','rule','mean','fold_std']].copy();view['method']=view.method.map(names)
        view['rule']=view.rule.map({'raw':'原始读出','bounded':'有界选参','raw_clipped':'同参数限幅','constant':'常数参照'})
        view.columns=['方法','输出规则','三折均值','折间描述性标准差'];lines +=[markdown_table(view),'']
    lines+=['## 上轮全部保留对照','',
        '下表保留六种方法、分块正则、原三档及扩展九档等全部汇总；其任务、受试者和冻结表示与本轮一致。不能靠只比较退化后的九档基线宣称优势。完整机器表见 preserved_readout_controls.csv。','']
    method_names={'constant':'简单参照','chronaris':'完整方法','physiology_only':'第一流单流','vehicle_only':'第二流单流',
        'naive_time_sync':'朴素时间同步','mult':'跨模态变换器 MulT','contiformer':'连续时间变换器 ContiFormer',
        'chronaris_second24':'完整模型内部第二流24维','chronaris_pca24':'完整表示压缩24维'}
    arm_names={'majority':'多数类','mean':'训练均值','median':'训练中位数','zero':'零响应','history':'历史趋势',
        'block_selected':'分块正则选参','uniform3':'统一正则三档','matched9':'统一正则九档',
        'favor_first':'第一流优先固定分块','favor_second':'第二流优先固定分块',
        'ridge':'岭回归','ridge_clipped':'岭回归同参数限幅','ridge_bounded_selection':'有界选参','selected':'训练内选参'}
    for (domain,task,route),d in old.groupby(['domain','task','route']):
        view=d[['method','arm','mean']].copy();view['method']=view.method.map(method_names);view['arm']=view.arm.map(arm_names)
        assert not view.isna().any().any()
        view.columns=['方法','读出方案','三折均值'];lines += ['### '+tasks[task]+'／'+('自监督' if route=='self_supervised' else '任务引导'),'',markdown_table(view),'']
    (HERE/'COMPLEMENTARITY.md').write_text('\n'.join(lines)+'\n')
    t=pd.read_csv(HERE/'timing_comparison.csv');all_t=[]
    for label,mask in [('非零时钟偏移',t.condition.str.startswith('clock_offset_')&~t.condition.str.contains('0.00')),
                       ('附加响应滞后',t.condition.str.startswith('physiology_lag_')),
                       ('全部已导出条件',np.ones(len(t),dtype=bool))]:
        d=t[mask].groupby(['method','variant','route','target','arm'])[['rmse','profile_mean_rmse']].mean().reset_index();d.insert(0,'condition_group',label);all_t.append(d)
    pd.concat(all_t).to_csv(HERE/'timing_aggregates.csv',index=False)
    stress=pd.read_csv(MATERIALS/'simulation_summary.csv')
    stress[stress.condition.str.startswith(('clock_offset_','clock_drift_','physiology_lag_'))].to_csv(HERE/'timing_downstream_controls.csv',index=False)
    save_json(HERE/'comparison_summary.json',dict(unchanged_controls=len(controls),
        primary_main_stable_passes=sum(r['joint_stable_pass'] for r in gates if r['primary_task'] and not r['exploratory']),
        primary_main_comparisons=sum(r['primary_task'] and not r['exploratory'] for r in gates),
        all_stable_passes=[r for r in gates if r['joint_stable_pass']]))
    print(json.dumps(json.loads((HERE/'comparison_summary.json').read_text()),ensure_ascii=False))


if __name__=='__main__':main()
