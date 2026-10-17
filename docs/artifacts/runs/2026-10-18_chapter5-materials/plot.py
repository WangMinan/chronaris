"""Pure rendering of supplied CSV values; no experiment imports or metric computation."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font

ROOT=Path(__file__).resolve().parent
STYLE=json.loads((ROOT/'style.json').read_text(encoding='utf-8'))
OUT=ROOT/STYLE['output_dir']


def rows(name):
    with (ROOT/'data'/name).open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def label(row):
    result=STYLE['methods'][row['method']]
    if row.get('variant') not in (None,'reference'):result+=' · '+STYLE['variants'][row['variant']]
    return result


def draw():
    OUT.mkdir(exist_ok=True)
    # Refresh CJK collection registration even when Matplotlib predates the font installation.
    for path in font_manager.findSystemFonts():
        if any(name in Path(path).name.lower() for name in ('notosanscjk','wqy','simhei','msyh')):
            font_manager.fontManager.addfont(path)
    font=font_manager.findfont(font_manager.FontProperties(family=STYLE['fonts']))
    if not all(ord(c) in ft2font.FT2Font(font).get_charmap() for c in '认知负荷'):
        raise ValueError('Install a CJK font listed in style.json before rendering Chinese figures')
    plt.rcParams.update({'font.sans-serif':STYLE['fonts'],'font.size':11,'axes.unicode_minus':False,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.prop_cycle':plt.cycler(color=STYLE['palette'])})
    mapping=[]
    def save(fig,name,inputs,note):
        fig.text(.5,.015,note,ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.05,1,.96))
        for ext in ('png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=STYLE['dpi'])
        plt.close(fig);mapping.append(dict(figure=name,inputs=inputs,script='plot.py',style='style.json'))
    data=rows('native_summary.csv')
    for domain,route,consumer in sorted({(r['domain'],r['route'],r['consumer']) for r in data}):
        selected=[r for r in data if (r['domain'],r['route'],r['consumer'])==(domain,route,consumer)]
        tasks=sorted({r['task'] for r in selected})
        fig,axes=plt.subplots(1,len(tasks),figsize=(STYLE['figure_sizes']['native_panel'][0]*len(tasks),STYLE['figure_sizes']['native_panel'][1]),squeeze=False)
        for ax,task in zip(axes[0],tasks):
            values=[r for r in selected if r['task']==task]
            for y,row in enumerate(values):
                if row['mean']!='':ax.errorbar(float(row['mean']),y,xerr=float(row['std']),fmt='o',capsize=3)
                else:ax.text(.01,y,'指标不可用',transform=ax.get_yaxis_transform())
            ax.set_yticks(range(len(values)),[label(r) for r in values]);ax.invert_yaxis()
            ax.set_title(STYLE['tasks'][task]);ax.set_xlabel(STYLE['metrics'][values[0]['metric']]+('（训练尺度）' if values[0]['error_units']=='training_scale' else ''))
            ax.grid(axis='x',alpha=.2)
        fig.suptitle(STYLE['domains'][domain]+' · '+STYLE['routes'][route]+' · '+STYLE['consumers'][consumer])
        save(fig,f'native_{domain}_{route}_{consumer}',['data/native_summary.csv'],
             '种子 17；点为折均值，误差线为折间描述性标准差；鼎新仅描述单记录时间块。')
    sim=rows('simulation_summary.csv')
    for route in ('self_supervised','task_guided'):
        for task,metric,consumer in (('workload_classification','macro_f1','linear'),('workload_regression','rmse','linear'),
                                    ('maneuver_segmentation','segmental_f1_iou_0.50','causal_tcn_duration')):
            values=[r for r in sim if (r['route'],r['task'],r['metric'],r['consumer'],r['condition'])==(route,task,metric,consumer,'clean_asynchronous')]
            fig,ax=plt.subplots(figsize=STYLE['figure_sizes']['simulation'])
            for y,row in enumerate(values):
                if row['mean']!='':ax.errorbar(float(row['mean']),y,xerr=float(row['std']),fmt='o',capsize=3)
            ax.set_yticks(range(len(values)),[label(r) for r in values]);ax.invert_yaxis()
            ax.set_title('受控仿真 · '+STYLE['tasks'][task]+' · '+STYLE['routes'][route]);ax.set_xlabel(STYLE['metrics'][metric])
            save(fig,f'simulation_{route}_{task}',['data/simulation_summary.csv'],'种子 17；误差线为生成参数档案间描述性标准差；不是置信区间。')
        for consumer in ('linear','minirocket'):
            selected=[r for r in sim if r['route']==route and r['consumer']==consumer and r['task']=='workload_regression' and r['metric']=='rmse'
                and (r['condition'].startswith(('random_missing','contiguous_gap')) or r['condition'] in ('physiology_missing','vehicle_missing','clean_asynchronous'))]
            conditions=list(dict.fromkeys(r['condition'] for r in selected))
            identities=list(dict.fromkeys((r['method'],r['variant']) for r in selected))
            lookup={(r['method'],r['variant'],r['condition']):float(r['mean']) if r['mean'] else float('nan') for r in selected}
            fig,ax=plt.subplots(figsize=STYLE['figure_sizes']['missing'])
            values=[[lookup[m,v,c] for c in conditions] for m,v in identities]
            img=ax.imshow(values,aspect='auto',cmap='viridis');fig.colorbar(img,ax=ax,label='负荷回归均方根误差')
            ax.set_xticks(range(len(conditions)),[STYLE['conditions'][c] for c in conditions],rotation=50,ha='right')
            ax.set_yticks(range(len(identities)),[label(dict(method=m,variant=v)) for m,v in identities])
            ax.set_title('配对缺失扰动 · '+STYLE['routes'][route]+' · '+STYLE['consumers'][consumer])
            save(fig,f'missing_{route}_{consumer}',['data/simulation_summary.csv'],'固定模型、固定配对窗口；无观测窗口保留，尾部误差另见随包数据。')
    sensitivity=rows('sensitivity.csv')
    for route in ('self_supervised','task_guided'):
        fig,axes=plt.subplots(1,2,figsize=STYLE['figure_sizes']['sensitivity'])
        for ax,task,metric in zip(axes,('simulated_future_workload_classification','simulated_future_workload_regression'),('macro_f1','rmse')):
            for consumer in ('linear','minirocket'):
                values=sorted([r for r in sensitivity if (r['route'],r['task'],r['metric'],r['consumer'])==(route,task,metric,consumer)],key=lambda r:float(r['weight']))
                ax.plot([float(r['weight']) for r in values],[float(r['value']) for r in values],marker='o',label=STYLE['consumers'][consumer])
            ax.set_xlabel('物理残差权重');ax.set_ylabel(STYLE['metrics'][metric]);ax.set_title(STYLE['tasks'][task]);ax.legend(fontsize=STYLE['legend_font_size'])
        save(fig,f'sensitivity_{route}',['data/sensitivity.csv'],'开发验证数据，种子 17；完整参照固定为 0.05，不依据留出成绩选权重。')
    causality=rows('causality_summary.csv')
    fig,ax=plt.subplots(figsize=STYLE['figure_sizes']['causality'])
    for route in ('self_supervised','task_guided'):
        for perturbation,title in (('repeat','重复推理'),('future_values','改变未来观测'),('future_missing','删除未来观测')):
            values=[r for r in causality if r['route']==route and r['perturbation']==perturbation]
            ax.plot([float(r['cutoff_s']) for r in values],[float(r['max_delta']) for r in values],marker='o',label=STYLE['routes'][route]+' · '+title)
    ax.axhline(1e-5,color='#b22222',linestyle='--',label='已验证的单精度容差');ax.set_xlabel('历史截止时刻（秒）');ax.set_ylabel('历史表示最大绝对差');ax.legend(fontsize=STYLE['legend_font_size'],ncol=2)
    save(fig,'causality',['data/causality_summary.csv'],'完整仿真参照，固定 32 个内部验证窗口；全部逐查询差异随包交付。')
    costs=rows('inference_summary.csv')
    fig,ax=plt.subplots(figsize=STYLE['figure_sizes']['inference'])
    for y,row in enumerate(costs):ax.errorbar(float(row['mean']),y,xerr=float(row['std']),fmt='o',capsize=3)
    ax.set_yticks(range(len(costs)),[label(r)+' · '+STYLE['routes'][r['route']] for r in costs]);ax.invert_yaxis();ax.set_xlabel('四窗口表示推理用时（秒）')
    save(fig,'inference',['data/inference_summary.csv'],'固定内部验证窗口；预热 2 次，记录 10 次 CUDA 同步计时；不含下游推理。')
    timing=rows('timing_metrics.csv')
    for route in ('self_supervised','task_guided'):
        selected=[r for r in timing if r['route']==route and r['condition']=='clean_asynchronous']
        targets=list(dict.fromkeys(r['target'] for r in selected));fig,axes=plt.subplots(1,len(targets),figsize=(STYLE['figure_sizes']['timing_panel'][0]*len(targets),STYLE['figure_sizes']['timing_panel'][1]),squeeze=False)
        for ax,target in zip(axes[0],targets):
            values=[r for r in selected if r['target']==target]
            ax.barh(range(len(values)),[float(r['rmse']) for r in values]);ax.set_yticks(range(len(values)),[label(dict(method=r['method'],variant=r['unit'].split('/')[1])) for r in values]);ax.invert_yaxis()
            ax.set_title(STYLE['targets'][target]);ax.set_xlabel('均方根误差（秒）')
        save(fig,f'timing_{route}',['data/timing_metrics.csv'],'固定岭回归探针；表示的线性可解码性，不等同于直接识别真实时延。')
    (OUT/'figure_inputs.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':draw()
