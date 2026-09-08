"""Chinese tables and fixed-layout scientific figures from verified final evidence."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.simulation_audit_figures import _configure_chinese_font

METHODS = {'chronaris':'连续融合', 'physiology_only':'生理单流', 'vehicle_only':'航电单流',
           'mult':'多模态变换器', 'contiformer':'连续时间变换器', 'naive_time_sync':'固定网格'}
ROUTES = {'self_supervised':'自监督', 'task_guided':'任务引导'}
TASKS = {'difficulty':'飞行难度', 'event_response':'离线事件条件响应', 'workload_classification':'认知负荷分类',
         'workload_regression':'认知负荷回归', 'maneuver_regression':'未来机动强度',
         'maneuver_classification':'未来机动三分类', 'physiology_regression':'未来生理字段'}


def safety_checks(pressure):
    indexed = {(r['method'],r['route'],r['seed'],r['condition']):r for r in pressure}
    rows = []
    for (method,route,seed,condition), item in sorted(indexed.items()):
        if method!='chronaris': continue
        def scores(record):
            f1 = [r['value'] for r in record['grouped']['profile_metrics'] if
                  r['consumer']=='linear' and r['metric']=='macro_f1' and r['task']=='workload_classification']
            error = [r['rmse'] for r in record['grouped']['regression_tails'] if r['consumer']=='linear' and r['profile_id']=='all_windows']
            if not f1 or len(error)!=1: raise ValueError('safety check lacks full-profile classification or all-window error')
            return float(np.mean(f1)), error[0]
        f1,error=scores(item)
        baseline=[scores(indexed[(name,route,seed,condition)]) for name in ('physiology_only','vehicle_only')]
        best_f1,best_error=max(r[0] for r in baseline),min(r[1] for r in baseline)
        rows.append(dict(route=route,seed=seed,condition=condition,f1=f1,best_single_f1=best_f1,
            f1_margin=.05,f1_passed=f1>=best_f1-.05,rmse=error,best_single_rmse=best_error,
            regression_relative_margin=.05,regression_passed=error<=1.05*best_error))
    return rows


def write_final_report(evidence, root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    public=pd.DataFrame(evidence['public_scalar_rows'])
    native=pd.DataFrame(evidence['native_subject_rows'])
    public.to_csv(root/'public_subject_metrics.csv',index=False)
    native.to_csv(root/'native_all_fields.csv',index=False)
    pd.DataFrame(evidence['simulation_rows']).to_csv(root/'simulation_metrics.csv',index=False)
    pd.DataFrame(evidence['simulation_paired_statistics']).to_csv(root/'simulation_paired_statistics.csv',index=False)
    pd.DataFrame(evidence['public_paired_statistics']).to_csv(root/'public_paired_statistics.csv',index=False)
    pd.DataFrame(evidence.get('mechanism_paired_statistics',[])).to_csv(root/'mechanism_paired_statistics.csv',index=False)
    pd.DataFrame(evidence.get('preselected_cases',[])).to_csv(root/'preselected_cases.csv',index=False)
    pd.DataFrame.from_dict(evidence.get('training_update_ledger',{}),orient='index').to_csv(root/'training_cost.csv')
    guards=safety_checks(evidence['pressure'])
    (root/'safety_checks.json').write_text(json.dumps(guards,ensure_ascii=False,indent=2)+'\n')
    summary=public.groupby(['domain','route','method','consumer','task','metric'],as_index=False).value.mean()
    _configure_chinese_font()
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for axis,(domain,task,title) in zip(axes,(('cogpilot','difficulty','虚拟飞行难度分类'),('clare','workload_classification','认知负荷分类')),strict=True):
        chosen=summary[(summary.domain==domain)&(summary.task==task)&(summary.consumer=='linear')&(summary.metric=='macro_f1')]
        for index,route in enumerate(ROUTES):
            values=chosen[chosen.route==route].set_index('method').value.reindex(METHODS)
            if values.isna().any():raise ValueError('figure requires all six methods and both routes')
            axis.barh(np.arange(6)+(index-.5)*.35,values,height=.35,label=ROUTES[route])
        axis.set_yticks(np.arange(6),list(METHODS.values()));axis.set_xlim(0,1)
        axis.set_xlabel('受试者等权宏平均 F1');axis.set_title(title);axis.legend(loc='lower right')
    fig.savefig(root/'公开分类主表.png',dpi=170);fig.savefig(root/'公开分类主表.svg');plt.close(fig)
    lines=['# 冻结后的分组评价报告','',
        f"本轮完成 {evidence['main_evaluation_units']} 个方法、划分、种子与表示路线评价单元。公开数据按受试者汇总，仿真按参数档案重采样；所有负面结果保留。",'',
        '下表展示公开主任务的固定线性消费者成绩。分类为宏平均 F1，回归为均方根误差；先按受试者评价，再等权汇总三个种子。','',
        '| 数据 | 路线 | 方法 | 任务 | 指标 | 数值 |','| --- | --- | --- | --- | --- | ---: |']
    domains={'cogpilot':'CogPilot 虚拟飞行','clare':'CLARE 认知负荷'}
    for row in summary[summary.consumer=='linear'].itertuples():
        lines.append(f'| {domains[row.domain]} | {ROUTES[row.route]} | {METHODS[row.method]} | {TASKS[row.task]} | {"宏平均 F1" if row.metric=="macro_f1" else "均方根误差"} | {row.value:.4f} |')
    lines+=['','下图比较两条表示路线在两个公开分类任务上的成绩，详细配对区间见统计文件。','',
            '![公开分类主表](公开分类主表.png)','',
            '配对区间均采用 2,000 次重采样。公开数据以受试者为单位，仿真以参数档案为单位并保留其全部轨迹和窗口；折和种子不作为独立受试者。','',
            '鼎新采用保留单记录的时间块评价。各随机种子、任务、生理字段、持久性技能、混淆矩阵和缺失覆盖均保存在字段表，不对单记录作跨架次显著性推断。','',
            '时间机制分开报告时钟幅值、有符号时钟偏移和首个生理字段实际响应时延。下表评价冻结表示的线性可读性，不能单独证明模型识别了精确事件时延；任务引导表示另保留编码器监督来源。','',
            '| 路线 | 方法 | 种子 | 时间目标 | 均方根误差（秒） | 第 95 百分位误差（秒） |','| --- | --- | ---: | --- | ---: | ---: |']
    for record in evidence['mechanisms']:
        for metric in record['metrics']:
            lines.append(f'| {ROUTES[record["route"]]} | {METHODS[record["method"]]} | {record["seed"]} | {metric["label"]} | {metric["rmse"]:.4f} | {metric["p95_absolute_error"]:.4f} |')
    failed=sum(not row['f1_passed'] or not row['regression_passed'] for row in guards)
    lines+=['',f'安全保护共 {len(guards)} 项逐场景逐种子比较，其中 {failed} 项至少一项阈值未满足。阈值保持分类 F1 差 0.05、回归误差相对增加 5%；未满足项不阻止结果归档。','',
        '完整证据文件保留全部压力场景、无观测覆盖、第 95 百分位误差及最差五个窗口的平方误差贡献；未截断长尾。','',
        '三个案例来自配置冻结前按身份哈希选定的不同参数档案，其全部种子和两条路线预测见 `preselected_cases.csv`，没有根据正式成绩挑选。编码器和任务头更新按实际检查点去重计数，见 `training_cost.csv`；消费者与数据准备耗时仍见各运行日志。','',
        '来源索引、完整预测与模型文件哈希见 `evidence.json`；公开配对区间见 `public_paired_statistics.csv`，仿真及核心消融见 `simulation_paired_statistics.csv`，逐项保护见 `safety_checks.json`。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
