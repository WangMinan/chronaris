"""Plot one selected unit's supplied training history without recomputing losses."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(unit, data, style_path, output):
    with (data/'training_curves.csv').open(encoding='utf-8',newline='') as handle:
        selected=[r for r in csv.DictReader(handle) if int(r['unit_index'])==unit]
    if not selected:raise ValueError('this unit has no recorded optimizer history')
    style=json.loads(style_path.read_text(encoding='utf-8'))
    plt.rcParams.update({'font.sans-serif':style['fonts'],'font.size':style['font_size'],
        'axes.unicode_minus':False,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    routes=list(dict.fromkeys(r['route'] for r in selected))
    fig,axes=plt.subplots(1,len(routes),figsize=(style['panel_width']*len(routes),style['panel_height']),squeeze=False)
    for ax,route in zip(axes[0],routes):
        values=[r for r in selected if r['route']==route]
        field=next((name for name in ('validation_selection_loss','public_selection_loss','loss') if any(r.get(name) for r in values)),None)
        if field is None:raise ValueError('no recorded scalar loss for this route')
        points=[(float(r['optimizer_updates'] or r['update']),float(r[field])) for r in values if r.get(field)]
        ax.plot([p[0] for p in points],[p[1] for p in points],color=style['colors']['model'],marker='o',markersize=3)
        ax.set_xlabel('实际优化器更新数')
        ax.set_ylabel('训练适配损失' if field=='loss' else '训练器验证选择损失')
        ax.set_title(style['routes'][route]);ax.grid(color='#e2e5e9',linewidth=.7)
    row=selected[0]
    fig.suptitle(f"{style['domains'][row['domain']]} · {style['methods'][row['method']]} · 划分 {int(row['fold_index'])+1} · 种子 {row['seed']}")
    fig.text(.5,.025,'按原训练记录逐点绘制；不同方法与路线的损失定义不同，不用于跨方法排名。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.92));output.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):fig.savefig(output/f'training_unit_{unit}.{extension}',dpi=style['dpi'])
    plt.close(fig)
    (output/f'training_unit_{unit}.inputs.json').write_text(json.dumps(dict(
        figure=f'training_unit_{unit}',input='data/training_curves.csv',script='plot_training.py',
        style=style_path.name,unit_index=unit),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit',type=int,required=True)
    parser.add_argument('--data-dir',type=Path,default=here/'data')
    parser.add_argument('--style',type=Path,default=here/'style.json')
    parser.add_argument('--output-dir',type=Path,default=here/'figures')
    args=parser.parse_args();draw(args.unit,args.data_dir,args.style,args.output_dir)
