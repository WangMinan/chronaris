"""Predeclared Chronaris component removals under each route's frozen configuration."""
from copy import deepcopy
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.v4_candidates import EXPANDED_SIMULATION_ROOT
from chronaris.evaluation.application_tasks.v4_confirmation_training import read_frozen_configuration, _train_confirmation_unit
from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
from chronaris.evaluation.application_tasks.v4_native_confirmation_cohort import run_confirmation_units
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY, simulation_unit_root, seal_simulation_models
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

ABLATIONS = ('no_continuous_evolution','no_physics_residual','no_independent_pair_loss','no_explicit_shift',
             'no_single_stream_bypass','no_quality_gate','no_expanded_missingness')


def build_core_ablation_plan(frozen):
    configurations = {}; skipped = []
    for route in ('self_supervised','task_guided'):
        base = frozen['methods'][route]['chronaris']
        for name in ABLATIONS:
            absent = ((name=='no_independent_pair_loss' and not base['training'].get('independent_pair_weight',0))
                or (name=='no_quality_gate' and not base['training'].get('quality_gate_enabled',False))
                or (name=='no_expanded_missingness' and not base['missingness_mixture']))
            if absent:
                skipped.append(dict(route=route,ablation=name,status='not_applicable_component_not_adopted'));continue
            options = deepcopy(base)
            options['name'] = base['name']+'__'+name
            if name in ('no_continuous_evolution','no_single_stream_bypass'):
                options['variant'] = name
            elif name=='no_physics_residual':
                options['physics_weight'] = 0.
            elif name=='no_independent_pair_loss':
                options['training']['independent_pair_weight'] = 0.
            elif name=='no_explicit_shift':
                options['explicit_shift_enabled'] = False
            elif name=='no_quality_gate':
                options['training']['quality_gate_enabled'] = False
            elif name=='no_expanded_missingness':
                options['missingness_mixture'] = False
            signature=json.dumps(options,sort_keys=True)
            item=configurations.setdefault(signature,dict(ablation=name,options=options,routes=[]))
            item['routes'].append(route)
    units=[dict(method='chronaris',candidate_name=item['options']['name'],ablation=item['ablation'],
        options=item['options'],routes=item['routes'],seed=seed,domain='simulation',fold_index=0,backend='neural')
        for item in configurations.values() for seed in (17,29,43)]
    return dict(format='chronaris.v4_core_ablation_plan.v1',units=units,skipped=skipped,
        evaluation_units=sum(len(unit['routes']) for unit in units),confirmation_feedback_used=False,
        evaluation_conditions=['clean_asynchronous'],observation_anchor_retained=True)


def _ablation_root(root):
    return Path(root)/'core_ablations'


def train_core_ablation(*,freeze_path,freeze_sha256,output_root,ablation,base_candidate,seed):
    frozen=read_frozen_configuration(freeze_path,freeze_sha256)
    plan=build_core_ablation_plan(frozen)
    selected=[unit for unit in plan['units'] if unit['ablation']==ablation and unit['seed']==seed
              and unit['candidate_name']==base_candidate+'__'+ablation]
    if len(selected)!=1:
        raise ValueError('ablation was not applicable to the frozen route configuration')
    unit=selected[0]
    with development_gpu_lock() as acquired:
        if not acquired:return dict(status='waiting_gpu',completed=False)
        _require_diagnostic_device(seed)
        result=_train_confirmation_unit(domain='simulation',fold_index=0,method='chronaris',options=unit['options'],
            routes=unit['routes'],seed=seed,output_root=_ablation_root(output_root),data_root=EXPANDED_SIMULATION_ROOT,
            registry_path=SIMULATION_REGISTRY,freeze_sha256=freeze_sha256,normalizer_root=frozen['normalizer_root'])
        receipt=dict(status='completed',completed=True,freeze_sha256=freeze_sha256,training=result,unit=unit)
        (simulation_unit_root(_ablation_root(output_root),unit)/'training_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt


def seal_core_ablation_models(*,freeze_path,freeze_sha256,output_root,simulation_root=EXPANDED_SIMULATION_ROOT):
    frozen=read_frozen_configuration(freeze_path,freeze_sha256)
    plan=build_core_ablation_plan(frozen)
    return seal_simulation_models(freeze_path=freeze_path,freeze_sha256=freeze_sha256,
        output_root=_ablation_root(output_root),ablation_units=plan['units'],simulation_root=simulation_root)


def evaluate_core_ablation(*,freeze_path,freeze_sha256,output_root,ablation,base_candidate,seed,
                          confirmation_root='artifacts/application_evaluation/2026-09-08_v4-simulation-confirmation'):
    from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import read_simulation_model_freeze
    from chronaris.evaluation.application_tasks.v4_simulation_confirmation_evaluation import _evaluate_simulation_unit
    sealed=seal_core_ablation_models(freeze_path=freeze_path,freeze_sha256=freeze_sha256,output_root=output_root)
    if sealed['status']!='frozen':raise ValueError('core ablation models are not all frozen')
    chosen=[row for row in sealed['records'] if row['ablation']==ablation and row['seed']==seed
            and row['candidate_name']==base_candidate+'__'+ablation]
    if len(chosen)!=1:raise ValueError('ablation evaluation is outside the predeclared inventory')
    main_path=Path(output_root)/'simulation_frozen_models.json';main_hash=sha256_file(main_path)
    models=read_simulation_model_freeze(main_path,main_hash,freeze_path=freeze_path,freeze_sha256=freeze_sha256)
    with development_gpu_lock() as acquired:
        if not acquired:return dict(status='waiting_gpu',completed=False)
        _require_diagnostic_device(seed)
        return _evaluate_simulation_unit(models=models,unit=chosen[0],freeze_sha256=freeze_sha256,
            model_freeze_sha256=main_hash,output_root=_ablation_root(output_root),confirmation_root=confirmation_root,
            include_pressure=False)


def run_core_ablation_cohort(*,freeze_path,freeze_sha256,output_root,stage):
    if stage not in ('train','evaluate'):raise ValueError('unsupported core ablation stage')
    frozen=read_frozen_configuration(freeze_path,freeze_sha256)
    plan=build_core_ablation_plan(frozen) | dict(freeze_sha256=freeze_sha256,stage=stage)
    def unit_args(unit):
        return [f'core-ablation-{stage}','--domain','simulation','--ablation',unit['ablation'],
                '--candidate-name',unit['options']['name'].split('__',1)[0],'--seed',str(unit['seed']),
                '--ablation-parent-root',str(output_root)]
    return run_confirmation_units(plan=plan,freeze_path=freeze_path,freeze_sha256=freeze_sha256,
        output_root=_ablation_root(output_root),backend='neural',queue_prefix='core_'+stage,unit_args=unit_args,
        result_name='training_complete.json' if stage=='train' else 'confirmation_unit.json')
