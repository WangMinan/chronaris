"""Planning extrapolation, not a confidence interval or completed training result."""
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
source = json.loads((root/'measurement-summary.json').read_text())
cases = source['cases']
assert all(v['status']=='completed' for v in cases.values())
sim = source['simulation_mature']
assert sim['status']=='completed' and sim['effective_batch']==32
rows = []
for name, domain in [('clare_batched','clare'), ('clare_no_event','clare'), ('cogpilot_batched','cogpilot')]:
    c = cases[name]
    pre_seconds = cases[domain]['active_pretraining_updates'][-1]
    warm_seconds = max(c['warmup_update_seconds'])
    joint_seconds = statistics.mean(c['joint_update_seconds'][-2:])
    cold_seconds = c['joint_update_seconds'][0] + c['pretraining_update_seconds'][0]
    val_rate = {}
    export_rate = {}
    for route in ('self_supervised','task_guided'):
        t = c['training'][route]
        assert t['validation_checks'] > 0 and t['counts']['validation'] > 0
        val_rate[route] = t['validation_seconds']/t['validation_checks']/t['counts']['validation']
        export_rate[route] = c['phase_seconds'][f'chronaris/{route}/export']/sum(c['sample_counts'].values())
    for fold in range(5):
        count = source['formal_counts'][f'{domain}/{fold}']
        pre = max(300, math.ceil(count['train']/16))
        joint = max(200, math.ceil(count['train']/16))
        phases = dict(pretraining=pre*pre_seconds, head_warmup=50*warm_seconds,
            joint=joint*joint_seconds, initial_transition=cold_seconds,
            validation=count['validation']*(math.ceil(pre/50)*val_rate['self_supervised']+math.ceil(joint/50)*val_rate['task_guided']),
            export=(count['train']+count['validation'])*sum(export_rate.values()),
            consumer_allowance=15*60)
        rows.append(dict(case=name, fold=fold, pre_updates=pre, joint_updates=joint,
            phase_seconds=phases, hours=sum(phases.values())/3600))
sim_pre = sum(sum(v.values()) for v in sim['measurements'][-1])
for name, pre, warm, joint in [('sim_no_continuous',0,0,425),('sim_no_physics',1500,50,500),('sim_no_event',1500,50,500)]:
    c = cases[name]
    joint_seconds = max(c['joint_update_seconds'])*8
    phases = dict(pretraining=pre*sim_pre, head_warmup=warm*max(c['warmup_update_seconds'])*8,
        joint=joint*joint_seconds, validation_and_export_allowance=30*60)
    rows.append(dict(case=name, pre_updates=pre, joint_updates=joint,
        phase_seconds=phases, hours=sum(phases.values())/3600))
groups = {}
for row in rows:
    groups[row['case']] = groups.get(row['case'],0)+row['hours']
hours = sum(groups.values())
result = dict(group_hours=groups, base_compute_hours=hours, rows=rows,
    planning_compute_hours=[hours*.8,hours*1.5],
    integration_evaluation_and_export_allowance_hours=[24,48],
    total_calendar_days=[(hours*.8+24)/24,(hours*1.5+48)/24],
    assumptions=[
        'One GPU, continuous availability, the agreed seed-17 15+3 subset only.',
        'Development batches extrapolated to fixed formal-fold window counts; no full-fold timing claim.',
        'Public no-event pretraining conservatively uses the complete-method mature rate.',
        'Simulation complete-method mature pretraining proxies the two new ablations; effective-32 joint cost scales measured effective-4 cost by eight.',
        'Paused simulation can resume only after source/configuration/replay qualification; otherwise add retraining cost.',
        '15 minutes per public unit for consumer fitting is a planning allowance, not a measurement.',
        'The 0.8 to 1.5 multipliers are scheduling scenarios, not statistical confidence bounds.',
        '24 to 48 hours covers integration, validation, evaluation and deliverable work; full external-model evaluation and missing-weight transfer are not measured.'
    ])
assert len(rows)==18 and all(r['hours']>0 for r in rows)
(root/'budget-estimate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','assumptions')},indent=2))
