"""Render supplied values only; independent of Chronaris, models and servers."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    with path.open(encoding='utf-8',newline='') as handle:
        return list(csv.DictReader(handle))


def draw(data, style_path, output):
    style=json.loads(style_path.read_text(encoding='utf-8'))
    plt.rcParams.update({'font.sans-serif':style['fonts'],'font.size':style['font_size'],
        'axes.unicode_minus':False,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    output.mkdir(parents=True,exist_ok=True)
    summaries=rows(data/'summary.csv'); mapping=[]
    panels=sorted({(r['domain'],r['consumer'],r['route']) for r in summaries})
    for domain,consumer,route in panels:
        selected=[r for r in summaries if (r['domain'],r['consumer'],r['route'])==(domain,consumer,route)]
        tasks=list(dict.fromkeys(r['task'] for r in selected))
        fig,axes=plt.subplots(1,len(tasks),figsize=(style['panel_width']*len(tasks),style['panel_height']),squeeze=False)
        for ax,task in zip(axes[0],tasks):
            values=[r for r in selected if r['task']==task]
            values.sort(key=lambda r:list(style['methods']).index(r['method']))
            for y,row in enumerate(values):
                if row['mean']=='':
                    ax.text(.01,y,'指标不可用',transform=ax.get_yaxis_transform());continue
                color=style['colors']['model'] if row['method']=='chronaris' else style['colors']['reference']
                ax.errorbar(float(row['mean']),y,xerr=float(row['std']),fmt='o',color=color,capsize=4,markersize=6)
            ax.set_yticks(range(len(values)),[style['methods'][r['method']] for r in values])
            ax.invert_yaxis();ax.set_xlim(left=0);ax.grid(axis='x',color='#e2e5e9',linewidth=.7)
            ax.set_title(style['tasks'][task],pad=14)
            unit='训练尺度归一化误差' if values[0]['error_units']=='training_scale' else style['units'].get(task,'')
            ax.set_xlabel('宏平均 F1（越高越好）' if values[0]['metric']=='macro_f1' else f'均方根误差（{unit}，越低越好）')
        fig.suptitle(f"{style['domains'][domain]} · {style['routes'][route]} · {style['consumers'][consumer]}",y=.985)
        fig.text(.5,.025,style['validation_note'],ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.07,1,.93),w_pad=3)
        name=f'validation_{domain}_{consumer}_{route}'
        for suffix in ('png','svg'):fig.savefig(output/f'{name}.{suffix}',dpi=style['dpi'])
        plt.close(fig)
        mapping.append(dict(figure=name,input='data/summary.csv',script='plot.py',style='style.json'))
    costs=rows(data/'cost_summary.csv')
    for domain in style['domains']:
        selected=[r for r in costs if r['domain']==domain]
        selected.sort(key=lambda r:list(style['methods']).index(r['method']))
        fig,ax=plt.subplots(figsize=(style['panel_width']+2,style['panel_height']))
        ax.barh(range(len(selected)),[float(r['mean_hours']) for r in selected],color=style['colors']['reference'],height=.6)
        ax.set_yticks(range(len(selected)),[style['methods'][r['method']] for r in selected]);ax.invert_yaxis()
        ax.set_xlabel('每个已完成训练单元的记录成本（小时）');ax.set_title(style['domains'][domain]+' · 训练与下游累计成本')
        ax.grid(axis='x',color='#e2e5e9',linewidth=.7);ax.set_axisbelow(True)
        fig.text(.5,.025,style['cost_note'],ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.07,1,1))
        name=f'cost_{domain}'
        for suffix in ('png','svg'):fig.savefig(output/f'{name}.{suffix}',dpi=style['dpi'])
        plt.close(fig)
        mapping.append(dict(figure=name,input='data/cost_summary.csv',script='plot.py',style='style.json'))
    (output/'figure_inputs.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=here/'data')
    parser.add_argument('--style',type=Path,default=here/'style.json')
    parser.add_argument('--output-dir',type=Path,default=here/'figures')
    args=parser.parse_args();draw(args.data_dir,args.style,args.output_dir)
