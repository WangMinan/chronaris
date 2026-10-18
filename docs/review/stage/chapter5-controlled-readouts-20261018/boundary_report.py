"""Matched-input-version comparison after the exclusive endpoint control completes."""
import json

import numpy as np
import pandas as pd

from common import HERE, ASSETS, save_csv, save_json
from summarize import NAMES, VARIANTS


def main():
    old=pd.read_csv(HERE/'cogpilot_folds.csv')
    new=pd.read_csv(HERE/'cogpilot_boundary_folds.csv')
    keys=['task','route','fold','method','variant']
    joined=old.merge(new,on=keys,suffixes=('_old','_exclusive'),validate='one_to_one')
    assert len(joined)==len(old)==len(new)
    save_csv(HERE/'boundary_fold_comparison.csv',joined.to_dict('records'))
    rows=[]
    for key,part in joined.groupby(['task','route','method','variant'],sort=False):
        rows.append(dict(zip(['task','route','method','variant'],key))|dict(original=part.value_old.mean(),
            exclusive=part.value_exclusive.mean(),exclusive_fold_std=float(np.std(part.value_exclusive)),
            delta=part.value_exclusive.mean()-part.value_old.mean()))
    save_csv(HERE/'boundary_comparison.csv',rows)
    feature_checks=[]
    for r in json.loads((HERE/'export_boundary_verification.json').read_text())['records']:
        filename=f"cogpilot__{r['method']}__{r['route']}.npz"
        with np.load(ASSETS/filename) as a, np.load(ASSETS/'exclusive_endpoint'/filename) as b:
            assert np.array_equal(a['sample_ids'],b['sample_ids'])
            classification=np.array(['::window' in s for s in a['sample_ids']])
            delta=abs(a['pooled']-b['pooled'])
            assert np.array_equal(a['pooled'][classification],b['pooled'][classification])
            feature_checks.append(dict(method=r['method'],route=r['route'],classification_max_delta=float(delta[classification].max()),
                event_max_delta=float(delta[~classification].max()),event_mean_delta=float(delta[~classification].mean())))
    save_json(HERE/'boundary_control_verification.json',dict(classification_exact=True,feature_checks=feature_checks,
        identical_window_ids=True,identical_targets=True,encoder_updates=0,input_change='exclude native event endpoint',
        query_points=96,query_grid_end_s=11.875,query_grid_source='src/chronaris/representation/collation.py'))
    lines=['# 严格事件端点对照','',
        '本表检验 CogPilot 飞行任务数据仅排除事件时刻及以后的输入观测，对冻结表示与嵌套读出的影响。原输入和目标保留，新视图使用独立观测身份；1462 个事件窗口全部保留。所有传统方法与完整方法都重新导出，参数仍只在内层训练受试者中选择。编码器历史训练输入没有改变，因此这是输入边界控制，不是完成重训后的新正式成绩。','',
        '1062 个难度窗口的所有 11 条路线表示逐值不变，分类结果沿用 [原窗口完整对照表](COMPARISON.md)。下表给出全部事件回归候选的原值及修正值；均方根误差 RMSE 越低越好，变化为“修正后减修正前”，单位 μS。','',
        '**端点改动未实质改变本轮方法结论。** 九条传统路线的全部表示逐值不变，完整方法两路线的最大表示差分别约 3.58×10⁻⁷ 和 2.38×10⁻⁷。标准 96 点查询网格止于 11.875 秒，位于被移除点之前，与表示基本不变的现象一致。该结果仅限固定权重导出与读出，不能追溯证明历史训练的辅助目标完全不受端点影响。','']
    for route in ('self_supervised','task_guided'):
        lines += [f"## {'自监督' if route=='self_supervised' else '任务引导'}",'',
            '同一受试者划分和目标下的结果如下，不能与原五折确认总体混合平均。','',
            '| 表示或参照 | 原输入 RMSE | 严格端点 RMSE | 变化 |','| --- | ---: | ---: | ---: |']
        for r in rows:
            if r['route']!=route or r['task']!='event_response':continue
            label=(NAMES[r['method']]+'／' if r['method']!='constant' else '')+VARIANTS[r['variant']]
            lines.append(f"| {label} | {r['original']:.6f} | {r['exclusive']:.6f} | {r['delta']:+.6f} |")
        lines.append('')
    (HERE/'BOUNDARY_COMPARISON.md').write_text('\n'.join(lines))
    print(pd.DataFrame(rows).query("route=='task_guided' and task=='event_response'").to_string(index=False))


if __name__=='__main__':main()
