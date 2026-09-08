import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_simulation_confirmation as module
from chronaris.evaluation.application_tasks import v4_simulation_confirmation_data as generation
from chronaris.evaluation.application_tasks.v4_confirmation_training import CONFIRMATION_BUDGET
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.representation import FoldLineage
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _configuration(root):
    methods={route:{method:candidate_options(method,'reference') for method in
        ('physiology_only','vehicle_only','mult','contiformer','chronaris')} for route in ('self_supervised','task_guided')}
    for route in methods:methods[route]['naive_time_sync']={'name':'reference','nonparametric':True}
    evidence=root/'evidence.json';evidence.write_text('{}')
    frozen=dict(format='chronaris.v4_frozen_configuration.v1',status='frozen',source_code_sha256=module.v4_workflow_source_sha256(),
        budget=CONFIRMATION_BUDGET,confirmation_feedback_used=False,evidence_files={str(evidence):sha256_file(evidence)},
        methods=methods,domain_status={'simulation':'enabled'},normalizer_root=str(root/'normalizers'))
    path=root/'frozen.json';path.write_text(json.dumps(frozen))
    return frozen,dict(freeze_path=path,freeze_sha256=sha256_file(path),output_root=root/'formal')


def test_all_selected_seeds_and_routes_must_finish_before_confirmation_generation(tmp_path,monkeypatch):
    frozen,kwargs=_configuration(tmp_path)
    registry=json.loads(Path(module.SIMULATION_REGISTRY).read_text())
    registry['historical_generation_manifests']=[]  # Engineering fixture has no archived production datasets.
    registry_path=tmp_path/'registry.json';registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(module,'SIMULATION_REGISTRY',str(registry_path))
    monkeypatch.setattr(generation,'SIMULATION_REGISTRY',str(registry_path))
    training_root=tmp_path/'training_data';training_root.mkdir()
    for name in ('v4_generation_audit.json','simulation_manifest.json'):(training_root/name).write_text('{}')
    data_hash=hashlib.sha256((sha256_file(registry_path)+sha256_file(training_root/'v4_generation_audit.json')).encode()).hexdigest()
    kwargs['simulation_root']=training_root
    assert len(module.seal_simulation_models(**kwargs)['pending'])==18
    assert not (kwargs['output_root']/'simulation_frozen_models.json').exists()
    fold=FoldLineage('test__training512',('train',),('validation',),('held',)).to_dict()
    for unit in module.simulation_confirmation_units(frozen):
        root=module.simulation_unit_root(kwargs['output_root'],unit);root.mkdir(parents=True)
        source=dict(freeze_sha256=kwargs['freeze_sha256'],source_code_sha256=module.v4_workflow_source_sha256(),
                    method=unit['method'],seed=unit['seed'],routes=unit['routes'],fold=fold,data_manifest_sha256=data_hash)
        if unit['method']=='naive_time_sync':
            checkpoint=root/'naive.pt';checkpoint.write_text('fixture')
            state=source | dict(unit=unit,completed=True,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint))
            (root/'training_complete.json').write_text(json.dumps(state))
        else:
            source['candidate_options']=candidate_options(unit['method'],unit['candidate_name'])
            state=dict(source=source,completed=True)
            for route in unit['routes']:
                checkpoint=root/(route+'.pt')
                payload=dict(training_status='completed',seed=unit['seed'],method_name=unit['method'],
                    config={'device':'cuda','data_manifest_sha256':data_hash},label_used_for_encoder_training=route=='task_guided')
                if route=='self_supervised':payload['fold']=fold
                else:payload.update(fold_id=fold['fold_id'],role_sample_ids={r:fold[r+'_sample_ids'] for r in ('train','validation','held_out')})
                torch.save(payload,checkpoint)
                state[route+'_training']=dict(best_checkpoint_path=str(checkpoint),optimizer_updates=500 if route=='self_supervised' else 250,
                                             joint_updates=200,head_warmup_updates=50)
            (root/'run_state.json').write_text(json.dumps(state))
    monkeypatch.setattr(module,'load_v4_naive_encoder',lambda path,**kw:(None,None,{'seed':int(Path(path).parent.name.removeprefix('seed'))}))
    result=module.seal_simulation_models(**kwargs)
    assert result['evaluation_units']==36 and not result['confirmation_generated']
    assert module.seal_simulation_models(**kwargs)==result
    path=kwargs['output_root']/'simulation_frozen_models.json'
    read_args=dict(freeze_path=kwargs['freeze_path'],freeze_sha256=kwargs['freeze_sha256'])
    assert generation.read_simulation_model_freeze(path,sha256_file(path),**read_args)==result
    calls=[]
    def generate(config,**kw):
        calls.append(config)
        rows=[dict(trajectory_id=trajectory['trajectory_id'],scenario_id=scenario.scenario_id,observation_seed=trajectory['observation_seed'])
              for trajectory in registry['trajectories'] if trajectory['role']=='confirmation' for scenario in config.observation_scenarios]
        manifest=config.run_root/'simulation_manifest.json';manifest.write_text(json.dumps({'scenario_rows':rows}))
        return SimpleNamespace(simulation_manifest_path=manifest,
            validation_rows=[dict(all_states_present=True,event_count=3,vehicle_values_finite=True,physiology_values_finite=True,
                                  vehicle_clock_mapping_max_error_s=0,physiology_clock_mapping_max_error_s=0)],
            paired_rows=[dict(latent_hash_shared=True,trajectory_id_shared=True,scenario_ids_unique=True)])
    monkeypatch.setattr(generation,'generate_benchmark',generate)
    audit=generation.generate_simulation_confirmation(**read_args,model_freeze_path=path,model_freeze_sha256=sha256_file(path),output_root=tmp_path/'observations')
    assert (audit['trajectory_count'],audit['scenario_count'],audit['contexts_per_scenario'])==(128,36,512)
    assert audit['stress_scenario_count']==35
    assert not audit['model_scores_generated'] and calls[0].split_specs[0].generator_family=='g2_event_spline'
    changed=next(iter(result['files']));Path(changed).write_text('modified')
    with pytest.raises(ValueError,match='evidence changed'):
        generation.generate_simulation_confirmation(**read_args,model_freeze_path=path,model_freeze_sha256=sha256_file(path),output_root=tmp_path/'blocked')
    assert len(calls)==1 and not (tmp_path/'blocked').exists()


def test_simulation_queue_reuses_native_executor_and_separates_gpu_evaluation(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import v4_native_confirmation_cohort as shared
    frozen,kwargs=_configuration(tmp_path);calls=[]
    def execute(**kw):calls.append(kw);return {'status':'captured'}
    monkeypatch.setattr(shared,'run_confirmation_units',execute)
    module.run_simulation_confirmation_cohort(**kwargs,stage='train',backend='nonparametric')
    assert sum(u['backend']=='nonparametric' for u in calls[0]['plan']['units'])==3
    kwargs['output_root'].mkdir();(kwargs['output_root']/'simulation_frozen_models.json').write_text('{}')
    module.run_simulation_confirmation_cohort(**kwargs,stage='evaluate',backend='neural')
    assert all(u['backend']=='neural' for u in calls[1]['plan']['units'])
    assert calls[0]['result_name']=='training_complete.json' and calls[1]['result_name']=='confirmation_unit.json'


def test_actual_g2_context_loading_keeps_identity_and_native_whole_modality_masks(tmp_path,monkeypatch):
    from chronaris.simulation.aviation_dual_stream.benchmark import SimulationBenchmarkConfig,SimulationSplitSpec,generate_benchmark
    from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig
    from chronaris.representation import load_simulation_observed_context
    result=generate_benchmark(SimulationBenchmarkConfig(run_id='engineering_g2',output_root=str(tmp_path),
        split_specs=(SimulationSplitSpec('engineering_g2','g2_event_spline',1,1,73001,74001,75001),),
        observation_scenarios=(ObservationScenarioConfig('clean_asynchronous'),ObservationScenarioConfig('example_missing',physiology_random_missing_rate=.3)),paired_observation_seed=True))
    root=Path(result.run_root)
    rows=json.loads((root/'simulation_manifest.json').read_text())['scenario_rows']
    clean=next(row for row in rows if row['scenario_id']=='clean_asynchronous')
    raw=Path(clean['scenario_manifest_path']).with_name('raw_dual_stream.npz')
    starts=[30.,60.,90.,120.]
    samples=[load_simulation_observed_context(raw,context_start_s=start) for start in starts]
    registry={'context_starts_s':starts,'trajectories':[dict(role='confirmation',profile_id=samples[0].group_id,
        trajectory_id=clean['trajectory_id'],context_sample_ids=[sample.sample_id for sample in samples])]}
    path=tmp_path/'engineering_registry.json';path.write_text(json.dumps(registry))
    monkeypatch.setattr(generation,'SIMULATION_REGISTRY',str(path))
    contract=root/'confirmation_generation_contract.json';contract.write_text('{}')
    audit=dict(status='completed',model_freeze_sha256='m'*64,simulation_manifest_sha256=sha256_file(root/'simulation_manifest.json'),
               registry_sha256=sha256_file(path),contract_sha256=sha256_file(contract))
    (root/'confirmation_generation_audit.json').write_text(json.dumps(audit))
    baseline=generation.load_simulation_confirmation(root,model_freeze_sha256='m'*64)
    for condition,stream in (('physiology_missing','physiology'),('vehicle_missing','vehicle')):
        missing=generation.load_simulation_confirmation(root,model_freeze_sha256='m'*64,condition=condition)
        assert missing.batch.sample_ids==baseline.batch.sample_ids
        assert not getattr(missing.batch,stream+'_feature_mask').any()
        other='vehicle' if stream=='physiology' else 'physiology'
        assert torch.equal(getattr(missing.batch,other+'_values'),getattr(baseline.batch,other+'_values'))
        assert missing.batch.source_sample_hashes!=baseline.batch.source_sample_hashes
    changed=generation.load_simulation_confirmation(root,model_freeze_sha256='m'*64,condition='example_missing')
    assert changed.batch.sample_ids==baseline.batch.sample_ids
    raw.write_bytes(raw.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='observations or targets changed'):
        generation.load_simulation_confirmation(root,model_freeze_sha256='m'*64)
