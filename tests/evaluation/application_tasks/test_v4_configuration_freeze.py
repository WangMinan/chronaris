import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from chronaris.evaluation.application_tasks import v4_configuration_freeze as module
from chronaris.evaluation.application_tasks import v4_dingxin_data
from chronaris.evaluation.application_tasks.v4_confirmation_training import read_frozen_configuration
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _fixture(root,monkeypatch):
    decisions={f'{m}/{r}':{'recommended_candidate':'reference','assessments':{'reference':{'eligible':True}}}
               for m in module.METHODS for r in module.ROUTES}
    adoption={'status':'single_factor_decisions_ready_not_frozen','confirmation_feedback_used':False,
              'decisions':decisions,'selection_plan_sha256':'a'*64,'source_summaries_sha256':{}}
    for name in ('public','simulation'):
        summary={'selection_plan_sha256':'a'*64,'confirmation_feedback_used':False,'files':{},'completed':[]}
        (root/(name+'_results_summary.json')).write_text(json.dumps(summary))
        adoption['source_summaries_sha256'][name]=hashlib.sha256(json.dumps(summary,sort_keys=True).encode()).hexdigest()
    (root/'adoption_decisions.json').write_text(json.dumps(adoption))
    xml=root/'pytest.xml';xml.write_text('<testsuites><testsuite><testcase name="test_example_cuda"/></testsuite></testsuites>')
    log=root/'pytest.log';log.write_text('engineering fixture')
    receipt={'format':'chronaris.v4_cuda_validation.v1','status':'completed','exit_code':0,'device':'cuda',
             'source_code_sha256':module.v4_workflow_source_sha256(),'test_suite_sha256':module._test_suite_sha256(),
             'command':[sys.executable,'-m','pytest','-q',f'--junitxml={xml}'],
             'files':{str(p):sha256_file(p) for p in (xml,log)}}
    receipt_path=root/'validation_receipt.json';receipt_path.write_text(json.dumps(receipt))
    normalizers=root/'normalizers'
    registry=root/'registry.json';roles={'domains':{}}
    for domain in ('cogpilot','clare'):
        folds=[{'fold_id':f'{domain}_{i}'} for i in range(5)]
        roles['domains'][domain]={'folds':{'confirmation':folds}}
        for fold in folds:
            path=normalizers/domain/fold['fold_id']/'normalization.json';path.parent.mkdir(parents=True);path.write_text('{}')
    registry.write_text(json.dumps(roles))
    audit=root/'dingxin_audit.json';audit.write_text(json.dumps({'data_manifest_sha256':'d'*64,'outer_roles_disjoint':False}))
    monkeypatch.setattr(v4_dingxin_data,'load_v4_dingxin_development',lambda:SimpleNamespace(data_manifest_sha256='d'*64))
    monkeypatch.setattr(module,'_simulation_freeze_evidence',lambda *args:({},[]))
    from chronaris.evaluation.application_tasks import v4_dingxin_deduplicated
    monkeypatch.setattr(v4_dingxin_deduplicated,'deduplicated_dingxin_data',lambda data:(SimpleNamespace(data_manifest_sha256='e'*64),{'retained':'first'}))
    dedup=root/'deduplicated.json';dedup.write_text(json.dumps({'status':'completed','selection':{'retained':'first'},'content_audit':{'data_manifest_sha256':'e'*64,'outer_roles_disjoint':True}}))
    return dict(review_root=root,validation_receipt=receipt_path,output_path=root/'frozen.json',registry_path=str(registry),dingxin_audit_path=audit,dingxin_deduplicated_audit_path=dedup,normalizer_root=normalizers)


def test_freeze_binds_validation_data_and_decisions_without_overwriting(tmp_path,monkeypatch):
    kwargs=_fixture(tmp_path,monkeypatch)
    frozen=module.freeze_reviewed_configuration(**kwargs)
    assert frozen['status']=='frozen' and frozen['domain_status']['dingxin']=='enabled'
    assert all(len(methods)==6 for methods in frozen['methods'].values())
    assert not frozen['development_initialization_for_public_confirmation']
    assert read_frozen_configuration(kwargs['output_path'],sha256_file(kwargs['output_path']))==frozen
    assert module.freeze_reviewed_configuration(**kwargs)==frozen
    monkeypatch.setattr(module,'development_gpu_lock',lambda:pytest.fail('verified current CUDA validation must not run twice'))
    assert module.validate_current_cuda(tmp_path)['status']=='completed'
    (tmp_path/'pytest.log').write_text('changed')
    with pytest.raises(ValueError,match='validation evidence changed'):module.freeze_reviewed_configuration(**kwargs)


def test_freeze_waits_for_remaining_scientific_work_and_rejects_cpu_validation(tmp_path,monkeypatch):
    kwargs=_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'_simulation_freeze_evidence',lambda *args:({},[{'seeds':[17,29]}]))
    assert module.freeze_reviewed_configuration(**kwargs)['status']=='waiting_for_continuous_cell_candidate'
    assert not kwargs['output_path'].exists()
    monkeypatch.setattr(module,'_simulation_freeze_evidence',lambda *args:({},[]))
    path=tmp_path/'adoption_decisions.json';adoption=json.loads(path.read_text())
    decision=adoption['decisions']['chronaris/self_supervised']
    decision['recommended_candidate']='multihorizon';decision['assessments']['multihorizon']={'eligible':True}
    path.write_text(json.dumps(adoption))
    assert module.freeze_reviewed_configuration(**kwargs)['status']=='waiting_for_remaining_development'
    decision['recommended_candidate']='reference';decision['assessments'].pop('multihorizon');path.write_text(json.dumps(adoption))
    receipt=json.loads(kwargs['validation_receipt'].read_text());receipt['device']='cpu'
    kwargs['validation_receipt'].write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='complete CUDA validation'):module.freeze_reviewed_configuration(**kwargs)
