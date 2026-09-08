"""Freeze reviewed choices and bind them to a complete CUDA validation receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.evaluation.application_tasks.v4_candidate_results import _completed_scores, _pressure_p95
from chronaris.evaluation.application_tasks.v4_confirmation_training import CONFIRMATION_BUDGET
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
from chronaris.evaluation.application_tasks.v4_public_screen import METHODS, ROUTES, development_gpu_lock
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _test_suite_sha256():
    root=Path(__file__).parents[4]
    paths=sorted((root/'tests').rglob('*.py'))+[root/name for name in ('pyproject.toml','pytest.ini','conftest.py') if (root/name).exists()]
    return hashlib.sha256(''.join(str(p.relative_to(root))+sha256_file(p) for p in paths).encode()).hexdigest()


def validate_current_cuda(output_root):
    receipt_path=Path(output_root)/'validation_receipt.json'
    if receipt_path.exists():
        _validation_evidence(receipt_path)
        return json.loads(receipt_path.read_text())
    with development_gpu_lock() as acquired:
        if not acquired:return {'status':'waiting_gpu'}
        _require_diagnostic_device(17)
        root=Path(output_root).resolve();root.mkdir(parents=True,exist_ok=True)
        source=v4_workflow_source_sha256();tests=_test_suite_sha256();xml=root/'pytest.xml';log=root/'pytest.log'
        command=[sys.executable,'-m','pytest','-q',f'--junitxml={xml}']
        project=Path(__file__).parents[4]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(project/'src'))
        env.pop('PYTEST_ADDOPTS',None)
        with _periodic_training_heartbeat('configuration_cuda_validation',30,root=root) as progress, log.open('w') as output:
            child=subprocess.Popen(command,cwd=project,env=env,stdout=output,stderr=subprocess.STDOUT)
            progress['child_pid']=child.pid
            code=child.wait()
        completed=code==0 and source==v4_workflow_source_sha256() and tests==_test_suite_sha256()
        receipt=dict(format='chronaris.v4_cuda_validation.v1',status='completed' if completed else 'failed',
            source_code_sha256=source,test_suite_sha256=tests,device='cuda',exit_code=code,command=command,
            files={str(p):sha256_file(p) for p in (xml,log) if p.exists()})
        (root/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt


def _validation_evidence(path):
    receipt=json.loads(Path(path).read_text())
    if (receipt['format']!='chronaris.v4_cuda_validation.v1' or receipt['status']!='completed' or receipt['exit_code']!=0
        or receipt['device']!='cuda' or receipt['source_code_sha256']!=v4_workflow_source_sha256()
        or receipt['test_suite_sha256']!=_test_suite_sha256()):
        raise ValueError('configuration freeze requires complete CUDA validation of the current source')
    files=dict(receipt['files']);files[str(path)]=sha256_file(path)
    for filename,digest in files.items():
        if sha256_file(filename)!=digest:raise ValueError('CUDA validation evidence changed')
    xml=[filename for filename in receipt['files'] if filename.endswith('.xml')]
    if len(xml)!=1:raise ValueError('CUDA validation requires its complete pytest XML report')
    if receipt['command']!=[sys.executable,'-m','pytest','-q',f'--junitxml={xml[0]}']:
        raise ValueError('CUDA validation must run the complete pytest suite without filters')
    cases=list(ET.parse(xml[0]).getroot().iter('testcase'))
    if (not cases or any(case.find('failure') is not None or case.find('error') is not None for case in cases)
        or not any('cuda' in case.get('name','').lower() and case.find('skipped') is None for case in cases)):
        raise ValueError('CUDA validation did not run and pass CUDA cases')
    return files


def _simulation_freeze_evidence(summary, review_root, pressure_root, selected_chronaris):
    files={};unstable=defaultdict(set)
    for row in summary['completed']:
        method,candidate,route,seed=(row[k] for k in ('method','candidate','route','seed'))
        update=1500 if route=='self_supervised' else 500
        unit=Path(review_root)/'simulation'/method/candidate/'review'/f'seed{seed}'
        state=json.loads((unit/'run_state.json').read_text())
        verified=_completed_scores(unit,state,route,update,method,candidate,phase='review',seed=seed)
        if any(verified[k]!=row[k] for k in verified):raise ValueError('simulation clean review changed before freezing')
        if _pressure_p95(pressure_root,state,row,method,candidate,route,update,phase='review',seed=seed)!=row['missingness_p95']:
            raise ValueError('simulation pressure review changed before freezing')
        path=Path(pressure_root)/method/candidate/'review'/f'seed{seed}'/route/str(update)/'run_state.json'
        pressure=json.loads(path.read_text());files[str(path)]=sha256_file(path)
        conditions={}
        for name,item in pressure['conditions'].items():
            for key in ('result','prediction'):files[item[key+'_path']]=item[key+'_sha256']
            files[str(Path(item['result_path']).parent/'representation/fusion_stream.npz')]=item['representation_sha256']
            if method=='chronaris':conditions[name]=json.loads(Path(item['result_path']).read_text())['encoding_diagnostics']['distributions']
        if method=='chronaris' and candidate==selected_chronaris[route]:
            for condition in ('contiguous_gap_15s','contiguous_gap_30s'):
                for name,clean in conditions['clean_asynchronous'].items():
                    gap=conditions[condition].get(name,{})
                    if ('state_norm' in name and clean.get('status')==gap.get('status')=='completed'
                        and gap['p99']>10*clean['p99']):unstable[(route,candidate,condition,name)].add(seed)
    triggered=[dict(route=k[0],candidate=k[1],condition=k[2],state=k[3],seeds=sorted(seeds))
               for k,seeds in unstable.items() if len(seeds)>=2]
    return files,triggered


def freeze_reviewed_configuration(*, review_root, validation_receipt, output_path,
                                  pressure_root='artifacts/application_evaluation/2026-09-08_v4-review-pressure',
                                  dingxin_audit_path='docs/artifacts/runs/2026-09-08_v4-fixed-native-development/vehicle_content_audit.json',
                                  dingxin_deduplicated_audit_path='artifacts/application_evaluation/2026-09-08_v4-dingxin-deduplicated/data_audit.json',
                                  normalizer_root='artifacts/application_evaluation/2026-09-08_v4-public-confirmation-normalizers',
                                  registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    root=Path(review_root);adoption_path=root/'adoption_decisions.json'
    if not adoption_path.exists():return dict(status='waiting_for_adoption_decisions',configuration_frozen=False)
    adoption=json.loads(adoption_path.read_text())
    if adoption['status']!='single_factor_decisions_ready_not_frozen':
        return dict(status='waiting_for_complete_development',configuration_frozen=False,upstream_status=adoption['status'])
    if adoption['confirmation_feedback_used'] or set(adoption['decisions'])!={f'{m}/{r}' for m in METHODS for r in ROUTES}:
        raise ValueError('freeze requires complete development-only decisions')
    files={str(adoption_path):sha256_file(adoption_path)}
    for name in ('public','simulation'):
        path=root/(name+'_results_summary.json');summary=json.loads(path.read_text())
        digest=hashlib.sha256(json.dumps(summary,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if (digest!=adoption['source_summaries_sha256'][name] or summary['confirmation_feedback_used']
            or summary['selection_plan_sha256']!=adoption['selection_plan_sha256']):
            raise ValueError('review evidence differs from adoption decisions')
        files[str(path)]=sha256_file(path);files.update(summary['files'])
        if name=='simulation':simulation=summary
    pressure_files,instability=_simulation_freeze_evidence(simulation,root,pressure_root,
        {route:adoption['decisions'][f'chronaris/{route}']['recommended_candidate'] for route in ROUTES})
    files.update(pressure_files)
    if instability:return dict(status='waiting_for_continuous_cell_candidate',instability=instability,configuration_frozen=False)
    methods={route:{} for route in ROUTES};pending=[]
    for route in ROUTES:
        for method in METHODS:
            decision=adoption['decisions'][f'{method}/{route}'];name=decision['recommended_candidate']
            if not decision['assessments'][name]['eligible']:raise ValueError('recommended candidate failed its adoption criteria')
            methods[route][method]=candidate_options(method,name)
        chronaris=adoption['decisions'][f'chronaris/{route}']
        changes=[name for name,value in chronaris['assessments'].items() if name!='reference' and value['eligible']]
        if len(changes)>1:pending.append(f'{route}:combination_evaluation')
        common=methods[route]['chronaris']['name']
        if common in ('missingness_mixture','multihorizon'):
            for method in METHODS:
                if method!='chronaris' and common not in adoption['decisions'][f'{method}/{route}']['assessments']:
                    pending.append(f'{method}/{route}:common_objective_comparison')
        methods[route]['naive_time_sync']={'name':'reference','nonparametric':True}
    if pending:return dict(status='waiting_for_remaining_development',pending=pending,configuration_frozen=False)
    if not Path(validation_receipt).exists():return dict(status='waiting_for_current_cuda_validation',configuration_frozen=False)
    files.update(_validation_evidence(validation_receipt))
    files[registry_path]=sha256_file(registry_path)
    registry=json.loads(Path(registry_path).read_text())
    for domain in ('cogpilot','clare'):
        folds=registry['domains'][domain]['folds']['confirmation']
        if len(folds)!=5:raise ValueError('public confirmation requires five fixed folds')
        for fold in folds:
            path=Path(normalizer_root)/domain/fold['fold_id']/'normalization.json'
            if not path.exists():return dict(status='waiting_for_confirmation_normalizers',configuration_frozen=False)
            files[str(path)]=sha256_file(path)
    from chronaris.evaluation.application_tasks.v4_dingxin_data import load_v4_dingxin_development
    from chronaris.evaluation.application_tasks.v4_dingxin_deduplicated import deduplicated_dingxin_data, PROTOCOL
    dingxin=json.loads(Path(dingxin_audit_path).read_text())
    original_dingxin=load_v4_dingxin_development()
    if dingxin['data_manifest_sha256']!=original_dingxin.data_manifest_sha256:
        raise ValueError('Dingxin content audit belongs to another data snapshot')
    files[str(dingxin_audit_path)]=sha256_file(dingxin_audit_path)
    if not Path(dingxin_deduplicated_audit_path).exists():
        return dict(status='waiting_for_dingxin_deduplication_audit',configuration_frozen=False)
    deduplicated=json.loads(Path(dingxin_deduplicated_audit_path).read_text())
    retained,selection=deduplicated_dingxin_data(original_dingxin)
    if (deduplicated['status']!='completed' or deduplicated['selection']!=json.loads(json.dumps(selection))
        or deduplicated['content_audit']['data_manifest_sha256']!=retained.data_manifest_sha256
        or not deduplicated['content_audit']['outer_roles_disjoint']):
        raise ValueError('Dingxin retained-record audit differs from the approved split')
    files[str(dingxin_deduplicated_audit_path)]=sha256_file(dingxin_deduplicated_audit_path)
    for filename,digest in files.items():
        if sha256_file(filename)!=digest:raise ValueError('freeze evidence file changed')
    frozen=dict(format='chronaris.v4_frozen_configuration.v1',status='frozen',configuration_frozen=True,
        source_code_sha256=v4_workflow_source_sha256(),budget=CONFIRMATION_BUDGET,methods=methods,
        public_registry_sha256=sha256_file(registry_path),evidence_files=files,confirmation_feedback_used=False,
        normalizer_root=str(normalizer_root),
        selection_plan_sha256=adoption['selection_plan_sha256'],
        domain_status={'simulation':'enabled','cogpilot':'enabled','clare':'enabled',
                      'dingxin':'enabled'},
        native_fold_counts={'cogpilot':5,'clare':5,'dingxin':1},
        domain_evaluation_protocol={'cogpilot':'subject_five_fold','clare':'subject_five_fold','dingxin':PROTOCOL},
        original_total_evaluation_units=468,amended_total_evaluation_units=432,
        combination_status='not_applicable_fewer_than_two_eligible_changes',
        development_initialization_for_public_confirmation=False)
    frozen=json.loads(json.dumps(frozen));path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('existing frozen configuration cannot be overwritten')
    if not path.exists():path.write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')
    return frozen
