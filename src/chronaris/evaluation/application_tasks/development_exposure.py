"""Record development label exposure against every frozen confirmation fold."""
import json
from pathlib import Path
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows


def audit_expanded_exposure(report):
    report=Path(report);roles=Path('docs/requirements/thesis-v4-public-subjects.json')
    original=json.loads(roles.read_text())['domains']['cogpilot']
    fitted=set(json.loads((report/'additional_subjects.json').read_text()))
    requested=set(original['folds']['confirmation'][0]['train_subjects'])
    assert fitted<=requested and len(requested)==18 and len(fitted)==16
    rows=[]
    for fold in original['folds']['confirmation']:
        held=set(fold['held_out_subjects']);validation=set(fold['validation_subjects'])
        rows.append(dict(fold=fold['fold_id'],held_subjects=sorted(held),
                         fitted_held_overlap=sorted(held&fitted),read_only_held_overlap=sorted(held&(requested-fitted)),
                         fitted_validation_overlap=sorted(validation&fitted)))
    first=original['folds']['confirmation'][0]
    assert not requested&(set(first['validation_subjects'])|set(first['held_out_subjects']))
    result=dict(status='audited',original_role_registry=str(roles.resolve()),registry_sha256=file_digest(roles),
                new_target_construction_subjects=sorted(requested),new_model_fit_subjects=sorted(fitted),folds=rows,
                first_fold_new_target_held_reserve=first['held_out_subjects'],first_fold_new_target_selection_reserve=first['validation_subjects'],
                original_results_modified=False,new_results_scope='development_only',optimizer_updates=0)
    save_json(report/'exposure_audit.json',result)
    save_rows(report/'exposure_summary.csv',[dict(fold=r['fold'],held_count=len(r['held_subjects']),
              fitted_held_overlap=len(r['fitted_held_overlap']),read_only_held_overlap=len(r['read_only_held_overlap']),
              fitted_validation_overlap=len(r['fitted_validation_overlap'])) for r in rows])
    return result
