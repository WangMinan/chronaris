"""Render supplied tables only. No project imports, metric calculation or model access."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw(data_dir, style_path, output_dir):
    style = json.loads(style_path.read_text(encoding='utf-8'))
    def rows(name):
        with (data_dir/name).open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))
    summary, settings = rows('summary.csv'), rows('setting_metrics.csv')
    plt.rcParams.update({'font.sans-serif': style['fonts'], 'font.size': style['font_size'],
        'axes.unicode_minus': False, 'svg.fonttype': 'none', 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.axisbelow': True})
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [('clare','workload_classification'),('clare','workload_regression'),
              ('cogpilot','difficulty'),('cogpilot','event_response')]
    for consumer, title in style['consumers'].items():
        fig, axes = plt.subplots(2, 2, figsize=style['figure_size'])
        for ax, (domain, task) in zip(axes.flat, panels):
            selected = [r for r in summary if (r['consumer'],r['domain'],r['task'])==(consumer,domain,task)]
            if len(selected) != 4:
                raise ValueError(f'expected four supplied summaries: {consumer}/{domain}/{task}')
            reference = 'contiformer' if domain=='clare' else 'vehicle_only'
            for i, route in enumerate(style['routes']):
                for method, offset in [('chronaris',-.16),(reference,.16)]:
                    row = next(r for r in selected if r['route']==route and r['method']==method)
                    color = style['colors']['chronaris' if method=='chronaris' else 'reference']
                    samples = [r for r in settings if (r['consumer'],r['domain'],r['task'],r['route'],r['method'])==(consumer,domain,task,route,method)]
                    if len(samples)!=9:
                        raise ValueError('expected nine supplied setting values')
                    x = i+offset
                    ax.scatter([x+(j-4)*.019 for j in range(9)], [float(r['value']) for r in samples],
                        s=20, color=color, alpha=.45, edgecolors='none', zorder=2)
                    ax.errorbar(x, float(row['mean']), yerr=float(row['std_across_settings']),
                        fmt='_', markersize=22, markeredgewidth=2.5, color=color, capsize=5, linewidth=1.8, zorder=3)
            ax.set_xticks([0,1], list(style['routes'].values()))
            ax.set_xlim(-.5,1.5); ax.set_ylim(bottom=0)
            if selected[0]['metric']=='macro_f1': ax.set_ylim(0,1)
            ax.set_ylabel(style['task_axis_labels'].get(task, style['metrics'][selected[0]['metric']]))
            ax.set_title(style['domains'][domain]+' · '+style['tasks'][task], pad=34, fontsize=12)
            ax.grid(axis='y', color='#e7e7e7', linewidth=.8)
            ax.legend(handles=[Line2D([],[],color=style['colors']['chronaris'],lw=3,label=style['methods']['chronaris']),
                Line2D([],[],color=style['colors']['reference'],lw=3,label=style['methods'][reference])],
                loc='lower left', bbox_to_anchor=(0,1.005), ncol=2, frameon=False, fontsize=8.5)
        fig.suptitle('固定配置跨划分复核｜'+title, fontsize=17, y=.97)
        fig.text(.5,.025,style['footnote'],ha='center',fontsize=10,color='#555555')
        fig.subplots_adjust(left=.085,right=.985,bottom=.1,top=.84,hspace=.58,wspace=.3)
        for extension in ('png','svg'):
            fig.savefig(output_dir/f'{consumer}_comparison.{extension}',dpi=style['dpi'])
        plt.close(fig)


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=here/'data')
    parser.add_argument('--style',type=Path,default=here/'style.json')
    parser.add_argument('--output-dir',type=Path,default=here/'figures')
    args = parser.parse_args()
    draw(args.data_dir,args.style,args.output_dir)
