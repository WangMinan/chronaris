"""Render already computed numerical tables; no fits, model loading, or network."""
import pandas as pd
from audit_io import HERE
from summarize_comparisons import markdown_table


def main():
    methods={'chronaris':'本文方法','naive_time_sync':'朴素时间同步','physiology_only':'生理单流',
        'vehicle_only':'飞机单流','mult':'跨模态变换器 MulT','contiformer':'连续时间变换器 ContiFormer'}
    variants={'reference':'完整／参照','no_continuous_evolution':'移除连续演化','no_physics_residual':'移除物理残差',
        'no_event_semantics':'移除事件语义','no_explicit_shift':'移除显式时移','no_single_stream_bypass':'移除单流旁路'}
    target_names={'clock_magnitude':'偏移幅值误差／秒','signed_clock':'有符号偏移误差／秒','response_lag':'响应时延误差／秒'}
    t=pd.read_csv(HERE/'timing_comparison.csv');aggregates=pd.read_csv(HERE/'timing_aggregates.csv')
    conditions=pd.read_csv(HERE/'timing_conditions.csv').set_index('condition')
    lines=['# 非零时差机制完整对照','',
        '所有数值由原冻结预测复算，均为均方根误差，越低越好。条件组汇总先计算每条件误差、再条件等权；逐参数档案等权值另外保存在 timing_comparison.csv，不与全窗口误差混用。原始预测每条件512窗口、16个确认参数档案，共12种条件。','',
        '训练均值和中位数从原512条训练轨迹与12个既定条件重建；每轨迹四窗口等权，未使用确认标签拟合。朴素同步在两路线共享同一预测，重复列示不增加独立证据。偏移探针衡量表示可解码性，实际校正时差的模型输出尚缺。','']
    def table(d):
        model=d[d.arm=='model'].pivot(index=['method','variant','route'],columns='target',values='rmse').reset_index()
        model['method']=model.method.map(methods);model['variant']=model.variant.map(variants)
        model['route']=model.route.map({'self_supervised':'自监督','task_guided':'任务引导'})
        model=model.rename(columns={'method':'方法','variant':'配置','route':'路线',**target_names})
        rows=[]
        for arm,label in [('zero','固定零'),('train_mean','训练均值'),('train_median','训练中位数')]:
            vals=d[d.arm==arm].groupby('target').rmse.first()
            rows.append({'方法':label,'配置':'常数参照','路线':'共享',**{target_names[k]:v for k,v in vals.items()}})
        return markdown_table(pd.concat([model,pd.DataFrame(rows)],ignore_index=True))
    for name,d in aggregates.groupby('condition_group',sort=False):
        lines+=['## '+name,'','下表汇总该组内全部已导出条件；负荷或分类指标没有混入这些时间误差。','',table(d),'']
    for name,d in t.groupby('condition',sort=False):
        c=conditions.loc[name]
        label=f"偏移 {c.physiology_clock_offset_s:+g} 秒，附加响应滞后 {c.additional_lag_s:g} 秒，生理时间抖动 {c.physiology_jitter_std_ms:g} 毫秒"
        lines+=['## '+label,'',f'既有条件索引：`{name}`。该表保留全部方法与消融，并使用相同真值及窗口。','',table(d),'']
    lines+=['## 条件覆盖与解释','',
        '干净条件带有生理5毫秒、飞机2毫秒时间抖动；显式零偏移条件无该抖动，二者分别保留。六种非零时钟偏移为正负0.25、1、3秒；四种附加响应滞后为2、5、15、30秒。时钟扰动与附加响应滞后采用单因素条件，尚无两者同时变化的组合。','',
        '原时间探针没有时钟漂移条件或直接校正输出。已有时钟偏移、漂移、响应滞后下的全部下游指标另存 timing_downstream_controls.csv；这些是任务表现，不能代替漂移估计误差。','']
    (HERE/'TIMING.md').write_text('\n'.join(lines))
    subjects=pd.read_csv(HERE/'event_subjects.csv');stability=pd.read_csv(HERE/'event_stability.csv')
    clare=pd.read_csv(HERE/'clare_subjects.csv');contrast=pd.read_csv(HERE/'event_matched_contrasts.csv')
    lines=['# 任务有效性完整对照','',
        '全部794个CLARE窗口的三套目标与原标签一致，全部1462个CogPilot事件目标按原始皮电电阻换算为电导后逐值一致。下面的分布和对照按受试者展示；原始分记录、匹配时刻、距离和所有扰动值分别保存在随包CSV。','',
        '## CLARE 原始评分与类别覆盖','',
        '标准分类采用大于等于5，历史分类采用大于等于7；评分单位为量表点。两种阈值全部保留，表中标准差为受试者窗口分布的样本标准差。','',
        markdown_table(clare.rename(columns={'subject':'受试者','windows':'窗口数','score_min':'最低分','score_max':'最高分',
            'score_mean':'平均分','score_std':'评分标准差','standard_high_fraction':'标准高类比例','legacy_high_fraction':'历史高类比例'})),'',
        '## CogPilot 事件与匹配非事件','',
        '目标为事件后8秒与前2秒皮电导中位数之差，单位微西门子。相同受试者、记录内按过去飞行状态、时间位置和皮电采样率匹配；1462事件匹配到1016个唯一控制。控制允许复用，统计以7名受试者为单位。','']
    cols=['subject','events','unique_controls','event_zero_rmse','control_zero_rmse','event_mean_abs','control_mean_abs','paired_signed_difference','paired_absolute_difference']
    view=subjects[cols].copy();view.columns=['受试者','事件数','唯一控制数','事件零预测误差','控制零预测误差','事件平均绝对变化','控制平均绝对变化','有符号配对差','绝对变化配对差']
    lines +=[markdown_table(view),'',
        '事件绝对变化较控制平均高0.01016微西门子，受试者重采样描述性95%区间为[-0.01076,0.02809]，五人正向、两人负向。有符号配对差为-0.02093微西门子，区间[-0.04354,-0.00156]；这说明匹配条件下方向有差异，但不足以把事件解释为稳定增强的皮电幅值，更不等于事件前可预测性或因果效应。','',
        '## 时间扰动稳定性','',
        '每项保持原前2秒／后8秒目标公式；下表先逐人计算再等权。符号翻转比例统计两次响应乘积小于零的窗口；相关系数只作描述，不把窗口数量当独立样本计算显著性。','']
    view=stability.groupby('comparison')[['change_rmse','spearman','sign_flip_fraction']].mean().reset_index()
    view['comparison']=view.comparison.map({'shift_-1.0':'事件提前1秒','shift_-0.5':'事件提前0.5秒','shift_0.5':'事件推迟0.5秒','shift_1.0':'事件推迟1秒','first_half':'后窗前半段','second_half':'后窗后半段'})
    view.columns=['对照','目标差异均方根／微西门子','秩相关','符号翻转比例'];lines +=[markdown_table(view),'',
        '## 尚未解决的任务语义','',
        'CLARE标签文件每列54个评分，缺少逐评分时间戳。15个开发记录的脑电起点约为0或120秒，外周流首观测相对脑电从约-0.75到39.52秒不等。[作者仓库](https://github.com/Prithila05/CLARE)及原README说明10秒评分频率，未给出逐记录评分时间锚点；首观测差也可能来自采集缺失，不能据此擅自平移信号。当前只确认原始值与工程构造一致。','',
        'CogPilot两条皮电记录各含一次时间回退，幅度约2.14与3.13秒。第一次严格顺序检查因此停止；复核确认原目标由时间区间布尔筛选构造，对行顺序不敏感。保留源行顺序与原目标，新增回退计数，未通过删除或排序改变输入。本次没有证明采集时钟本身已经校正。','',
        '官方CogPilot质量摘要另作来源核对，传感器覆盖／采样质量与本任务构造的生理意义分别判断；没有将质量摘要当成事件标签真值。','']
    (HERE/'TASK_VALIDITY.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
