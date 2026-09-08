"""Source-checked Chinese figures for the completed initial v4 diagnostics."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import torch

from chronaris.evaluation.application_tasks.simulation_audit_figures import _configure_chinese_font
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

METHODS = {'physiology_only': '生理单流', 'vehicle_only': '航电单流', 'mult': 'MulT',
           'contiformer': 'ContiFormer', 'chronaris': 'Chronaris'}
ROUTES = {'self_supervised': '自监督', 'task_guided': '任务引导'}
CONDITIONS = dict(zip(('clean_asynchronous', 'random_missing_30pct', 'contiguous_gap_15s', 'contiguous_gap_30s',
    'physiology_missing', 'vehicle_missing', 'physiology_clock_offset_plus_1s', 'physiology_response_lag_plus_15s'),
    ('干净', '随机缺失\n30%', '连续缺失\n15 秒', '连续缺失\n30 秒', '生理全缺失', '航电全缺失', '生理时钟\n偏移 +1 秒', '生理响应\n时延 +15 秒'), strict=True))
METRICS = (('linear', 'macro_f1', '负荷分类宏平均 F1'), ('linear', 'rmse', '负荷回归均方根误差'),
    ('causal_tcn_duration', 'frame_macro_f1', '机动逐点宏平均 F1'),
    ('causal_tcn_duration', 'segmental_f1_iou_0.50', '机动片段 F1（交并比 0.5）'),
    ('causal_tcn_duration', 'boundary_f1_1s', '机动边界 F1（容差 1 秒）'),
    ('causal_tcn_duration', 'boundary_detection_delay_s', '机动检测延迟（秒）'))
BLUE, GRAY = '#356C9B', '#747474'


def _select_scores(frame):
    if (set(frame.seed) != {17} or set(frame.role) != {'validation'} or frame.smoke_only.any()
        or set(frame.fold) != {'v4_simulation_g1_development_g2_confirmation'}):
        raise ValueError('initial figure data changed seed, role or training scope')
    selected = frame.loc[[any(row.consumer == c and row.metric == m for c,m,_ in METRICS) for row in frame.itertuples()]].copy()
    key = ['representation_route', 'optimizer_update', 'method', 'consumer', 'metric']
    if (len(selected) != 180 or selected.duplicated(key).any() or set(selected.optimizer_update) != {50, 200, 500}
        or set(selected.method) != set(METHODS) or set(selected.representation_route) != set(ROUTES)
        or not np.isfinite(selected.value).all() or set(selected.status) != {'available'}):
        raise ValueError('initial six-task stage matrix is incomplete or duplicated')
    return selected


def render_initial_diagnostic_figures(*, output_root, initial_root='docs/artifacts/runs/2026-09-07_v4-initial-diagnostics',
                                     pressure_root='docs/artifacts/runs/2026-09-07_v4-development-pressure',
                                     runtime_root='artifacts/application_evaluation/2026-09-06_v4-learning-curves/simulation'):
    root, initial, pressure, runtime = map(Path, (output_root, initial_root, pressure_root, runtime_root))
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    def read(path):
        path = Path(path)
        sources[str(path)] = sha256_file(path)
        return json.loads(path.read_text())
    def verify(path, digest):
        if sha256_file(path) != digest:
            raise ValueError(f'archived diagnostic evidence changed: {path}')
        sources[str(path)] = digest
    audit = read(initial / 'validation_summary.json')
    if audit['confirmation_opened'] or audit['checkpoint_count'] != 50:
        raise ValueError('unexpected initial diagnostic audit')
    for checkpoint in audit['checkpoints']:
        verify(checkpoint['path'], checkpoint['sha256'])
    frame = pd.read_csv(initial / 'metric_long.csv')
    sources[str(initial / 'metric_long.csv')] = sha256_file(initial / 'metric_long.csv')
    scores = _select_scores(frame)
    for method in METHODS:
        for route in ROUTES:
            for update in (50, 200, 500):
                raw = read(runtime / method / f'{route}_{update}_consumers.json')
                for row in scores[(scores.method == method) & (scores.representation_route == route) & (scores.optimizer_update == update)].itertuples():
                    expected = [v for v in raw['metric_rows'] if (v['consumer'],v['metric']) == (row.consumer,row.metric)]
                    if len(expected) != 1 or not np.isclose(row.value, expected[0]['value'], rtol=0, atol=1e-12):
                        raise ValueError('archived stage table differs from raw consumer results')
    pressure_audit = read(pressure / 'validation_summary.json')
    if pressure_audit['confirmation_opened'] or pressure_audit['consumer_refit'] or pressure_audit['evaluation_units'] != 80:
        raise ValueError('unexpected pressure diagnostic audit')
    tails, encoding = [], []
    for unit in pressure_audit['units']:
        verify(unit['result_path'], unit['result_sha256'])
        result = read(unit['result_path'])
        verify(unit['prediction_path'], unit['prediction_sha256'])
        tails.extend(dict(method=unit['method'],route=unit['route'],condition=unit['condition'], **row)
                     for row in result['grouped']['regression_tails'] if row['consumer'] == 'linear' and row['profile_id'] == 'all_windows')
        if unit['method'] == 'chronaris':
            encoding.append(dict(route=unit['route'], condition=unit['condition'], **result['encoding_diagnostics']))
    tails = pd.DataFrame(tails)
    if len(tails) != 80 or tails.duplicated(['route','condition','method']).any() or set(tails.support) != {256}:
        raise ValueError('pressure figure must retain all 256 windows in every unit')
    _configure_chinese_font()
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    figures = []
    def save(fig, name, title):
        for suffix in ('png', 'svg'):
            path = root / f'{name}.{suffix}'
            fig.savefig(path, dpi=180, facecolor='white')
            figures.append(dict(path=str(path), title=title, sha256=sha256_file(path)))
        plt.close(fig)
    _stage_figures(scores, save)
    _pressure_figures(tails, encoding, save)
    _observation_figure(encoding, save)
    checkpoints = {}
    for route in ROUTES:
        item = next(v for v in audit['checkpoints'] if v['method'] == 'chronaris' and v['route'] == route and v['kind'] == 'last_checkpoint_path')
        checkpoints[route] = torch.load(item['path'], map_location='cpu', weights_only=True)
    training = _training_figure(checkpoints, save)
    scores.to_csv(root / 'stage_values.csv', index=False)
    tails.to_json(root / 'pressure_values.json', orient='records', indent=2)
    training.to_csv(root / 'training_values.csv', index=False)
    manifest = dict(scope='initial_training256_seed17_development_only', training_trajectories=256,
        validation_trajectories=64, validation_profiles=8, validation_windows=256, source_files=sources,
        figures=figures, plotted_stage_values=len(scores), pressure_units=len(tails),
        source_code_sha256=sha256_file(__file__), unavailable_states_are_not_zero=True)
    (root / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def _stage_figures(frame, save):
    for route, label in ROUTES.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for axis, (consumer, metric, title) in zip(axes.flat, METRICS, strict=True):
            values = frame[(frame.representation_route == route) & (frame.consumer == consumer) & (frame.metric == metric)]
            for offset, (update,color,marker) in enumerate(((50,'#ADC4D7','o'),(200,GRAY,'s'),(500,BLUE,'^'))):
                y = values[values.optimizer_update == update].set_index('method').loc[list(METHODS),'value'].to_numpy()
                axis.scatter(np.arange(5)+(offset-1)*.2, y, color=color, marker=marker, s=46, label=f'{update} 次更新')
            axis.set_xticks(range(5), METHODS.values(), rotation=15)
            axis.set_xlim(-.55,4.55)
            axis.set_ylim(0, 1 if 'f1' in metric else float(values.value.max()) * 1.15)
            axis.set_title(title + (' ↑' if 'f1' in metric else ' ↓'))
            axis.grid(axis='y', color='#E6E6E6');axis.set_axisbelow(True)
        fig.suptitle(f'{label}表示：六项任务的阶段成绩', fontsize=18, y=.985)
        subtitle = '预训练更新数' if route == 'self_supervised' else '联合适配更新数；此前另有 500 次预训练和 50 次任务头预热'
        fig.text(.5,.94, f'{subtitle}｜训练 256 条轨迹，开发 256 窗口，种子 17',ha='center')
        fig.legend(*axes[0,0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(.5,.915), frameon=False)
        fig.text(.5,.02,'分类与回归：线性消费者；机动：因果时序卷积及持续时间解码。仅展示三个实测阶段。',ha='center')
        fig.subplots_adjust(top=.84,bottom=.1,hspace=.42,wspace=.26)
        save(fig, f'stages_{route}', f'{label}六项任务阶段成绩')


def _matrix(axis, values, labels, title, *, norm=None, fmt='.2f', cmap='Blues'):
    masked = np.ma.masked_invalid(np.asarray(values,dtype=float))
    image = axis.imshow(masked, aspect='auto', cmap=cmap, norm=norm)
    axis.set_xticks(range(8), CONDITIONS.values(), fontsize=10)
    axis.set_yticks(range(len(labels)),labels);axis.set_title(title,pad=12)
    for (i,j),value in np.ndenumerate(masked.data):
        axis.text(j,i,format(value,fmt) if np.isfinite(value) else '不可用',ha='center',va='center',fontsize=10,
                  color='white' if np.isfinite(value) and image.norm(value) > .62 else '#262626')
    return image


def _pressure_figures(tails, encoding, save):
    fig, axes = plt.subplots(2,1,figsize=(14,8))
    for axis,(route,label) in zip(axes,ROUTES.items(),strict=True):
        matrix=tails[tails.route==route].pivot(index='method',columns='condition',values='rmse').loc[list(METHODS),list(CONDITIONS)]
        ratios=matrix.div(matrix['clean_asynchronous'],axis=0)
        image=_matrix(axis,ratios.values,list(METHODS.values()),label,norm=LogNorm(.5,16),fmt='.2f')
    fig.suptitle('八条件负荷回归误差相对干净场景的倍数',fontsize=17,y=.98)
    fig.text(.5,.935,'同方法、同路线干净误差为 1；线性消费者冻结；每格保留全部 256 个开发窗口，种子 17',ha='center')
    fig.subplots_adjust(top=.85,bottom=.09,left=.12,right=.86,hspace=.48)
    fig.colorbar(image,cax=fig.add_axes([.89,.16,.017,.64]),label='误差倍数（对数色阶）',ticks=[.5,1,2,4,8,16],format='%g')
    save(fig,'pressure_rmse_ratio','八条件回归误差倍数')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for axis,(field,title,factor) in zip(axes,(('p95_absolute_error','第 95 百分位绝对误差',1),('top_five_squared_error_fraction','最大五窗口平方误差占比（%）',100)),strict=True):
        for i,(route,label) in enumerate(ROUTES.items()):
            values=tails[(tails.route==route)&(tails.condition=='contiguous_gap_15s')].set_index('method').loc[list(METHODS),field]*factor
            axis.scatter(values,np.arange(5)+(i-.5)*.2,label=label,color=(BLUE,GRAY)[i],marker=('o','s')[i],s=55)
        axis.set_yticks(range(5),METHODS.values());axis.invert_yaxis();axis.set_xlabel(title);axis.set_xlim(left=0);axis.grid(axis='x',alpha=.2)
    fig.suptitle('15 秒连续缺失下的回归误差与长尾贡献',fontsize=17)
    fig.text(.5,.89,'训练 256 条轨迹；每方法每路线保留 256 个开发窗口；线性消费者；种子 17',ha='center')
    axes[1].legend(frameon=False);fig.subplots_adjust(top=.80,left=.12,right=.98,bottom=.15,wspace=.4)
    save(fig,'regression_tails','连续缺失回归误差分解')
    fig,axes=plt.subplots(3,1,figsize=(14,10))
    for axis,stream,title in zip(axes[:2],('physiology','vehicle'),('生理查询状态范数第 99 百分位 / 干净参照','航电查询状态范数第 99 百分位 / 干净参照'),strict=True):
        values=[]
        for route in ROUTES:
            rows={r['condition']:r['distributions'][stream+'_query_state_norm'] for r in encoding if r['route']==route}
            clean=rows['clean_asynchronous']['p99']
            values.append([rows[c].get('p99',np.nan)/clean for c in CONDITIONS])
        image=_matrix(axis,values,list(ROUTES.values()),title,norm=LogNorm(.5,16))
    attention=[]
    for route in ROUTES:
        rows={r['condition']:r['distributions'].get('attention_scale_0_maximum_weight', {}) for r in encoding if r['route']==route}
        attention.append([rows[c].get('mean',np.nan) for c in CONDITIONS])
    _matrix(axes[2],attention,list(ROUTES.values()),'最短历史尺度：平均最大注意力权重',norm=matplotlib.colors.Normalize(0,1))
    fig.suptitle('Chronaris 查询状态与注意力诊断',fontsize=17,y=.98)
    fig.text(.5,.94,'同一随机种子 17 的两条路线；缺乏有效历史时标为不可用；状态增长尚需第二种子复核',ha='center')
    fig.subplots_adjust(top=.86,bottom=.09,left=.12,right=.96,hspace=.7)
    save(fig,'state_attention','查询状态与注意力诊断')


def _observation_figure(encoding,save):
    clean=[r for r in encoding if r['condition']=='clean_asynchronous']
    labels=('脑电低频相对能量','脑电高频相对能量','脑电复杂度','血氧饱和度','心率','生理变异度','个体基线',
            '速度','高度','垂向速度','滚转角','俯仰角','偏航角','滚转角速度','俯仰角速度','偏航角速度','纵向加速度','横向加速度','法向过载')
    field_labels = dict(zip(('eeg_low_relative','eeg_high_relative','eeg_complexity','spo2_percent','heart_rate_bpm',
        'physiology_variability','individual_baseline','speed_mps','altitude_m','vertical_speed_mps','roll_rad','pitch_rad','yaw_rad',
        'roll_rate_rps','pitch_rate_rps','yaw_rate_rps','longitudinal_acc_mps2','lateral_acc_mps2','normal_load_g'),labels,strict=True))
    fields=[r['field'] for r in clean[0]['observation_fit']]
    if len(fields)!=len(labels) or {field.split('.')[-1] for field in fields}!=set(field_labels):
        raise ValueError('observation field inventory changed')
    labels=[field_labels[field.split('.')[-1]] for field in fields]
    fig,axes=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1.4,1]})
    for i,(route,label) in enumerate(ROUTES.items()):
        row=next(r for r in clean if r['route']==route)
        if [r['field'] for r in row['observation_fit']]!=fields:raise ValueError('observation field order changed')
        axes[0].scatter([r['standardized_rmse'] for r in row['observation_fit']],np.arange(19)+(i-.5)*.2,color=(BLUE,GRAY)[i],marker=('o','s')[i],label=label)
        physical={r['component']:r['mean_normalized_huber_residual'] for r in row['physical_components']}
        axes[1].scatter([physical[name] for name in ('vehicle_rigid_body_translation','vehicle_rigid_body_vertical','vehicle_rigid_body_rotation')],np.arange(3)+(i-.5)*.2,color=(BLUE,GRAY)[i],marker=('o','s')[i],label=label)
    axes[0].set_yticks(range(19),labels);axes[0].invert_yaxis();axes[0].set_xlabel('训练尺度下观测拟合均方根误差')
    axes[1].set_yticks(range(3),('速度—纵向加速度','高度—垂向速度','三个角度—角速度'));axes[1].invert_yaxis();axes[1].set_xlabel('归一化物理残差的平均 Huber 损失');axes[1].legend(frameon=False)
    for axis in axes:axis.set_xlim(left=0);axis.grid(axis='x',alpha=.2)
    fig.suptitle('Chronaris 干净场景的观测拟合与物理残差',fontsize=17,y=.97)
    fig.text(.5,.93,'同一观测解码器；训练 256 条轨迹；开发 256 窗口；种子 17；两类指标各自使用训练尺度',ha='center')
    fig.subplots_adjust(top=.86,left=.14,right=.97,bottom=.09,wspace=.7)
    save(fig,'observation_physics','观测拟合与物理残差')


def _training_figure(checkpoints,save):
    rows=pd.DataFrame(checkpoints['self_supervised']['training_rows'])
    if rows.groupby('step').gradient_norm_before_clip.nunique().max()!=1 or rows.step.nunique()!=500:
        raise ValueError('gradient traces do not map to 500 optimizer updates')
    loss=rows.groupby(['step','term_name']).raw_loss.mean().unstack()
    gradient=rows.drop_duplicates('step').set_index('step').gradient_norm_before_clip.sort_index()
    guided=pd.DataFrame(checkpoints['task_guided']['update_rows'])
    guided=guided[guided.stage=='joint_adaptation'].groupby('optimizer_update')[['task_loss','public_loss','mechanism_loss']].mean()
    guided.index-=50
    if len(guided)!=500:raise ValueError('guided trace does not contain 500 joint updates')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for column,label,color in (('masked_reconstruction','掩码重构',BLUE),('short_horizon_prediction','短时预测',GRAY)):
        axes[0,0].plot(loss.index,loss[column].rolling(25,min_periods=1).mean(),label=label,color=color)
    axes[0,0].set_title('预训练公共目标（25 更新滑动均值）');axes[0,0].legend(frameon=False)
    axes[0,1].plot(gradient.index,gradient,color=BLUE,linewidth=.7);axes[0,1].axhline(1,color=GRAY,linestyle='--',label='裁剪阈值 1')
    axes[0,1].set_yscale('symlog',linthresh=1)
    axes[0,1].set_title('梯度范数（1 以下线性，以上对数）');axes[0,1].legend(frameon=False)
    axes[1,0].plot(guided.index,guided.task_loss.rolling(25,min_periods=1).mean(),color=BLUE)
    axes[1,0].set_title('联合适配任务损失（25 更新滑动均值）')
    for column,label,color in (('public_loss','公共自监督目标',GRAY),('mechanism_loss','机制目标',BLUE)):
        axes[1,1].plot(guided.index,guided[column].rolling(25,min_periods=1).mean(),label=label,color=color)
    axes[1,1].set_title('辅助目标：公共项未乘 0.2，机制项含配置权重');axes[1,1].legend(frameon=False)
    for i,axis in enumerate(axes.flat):
        axis.set_xlabel('预训练更新数' if i<2 else '联合适配更新数');axis.set_xlim(1,500);axis.set_ylim(bottom=0);axis.grid(alpha=.2)
    fig.suptitle('Chronaris 真实更新上的训练诊断',fontsize=17,y=.98)
    fig.text(.5,.93,'种子 17；实际批量 4、有效批量 32；同一更新内微批取均值；完整原始记录保留',ha='center')
    fig.subplots_adjust(top=.84,bottom=.09,hspace=.4,wspace=.3)
    save(fig,'training_trace','真实更新训练诊断')
    return pd.concat([loss.add_prefix('pretraining_'),gradient.rename('gradient_norm_before_clip'),guided.add_prefix('guided_')],axis=1).rename_axis('optimizer_update').reset_index()
