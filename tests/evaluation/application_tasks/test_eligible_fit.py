from chronaris.evaluation.application_tasks.eligible_fit_inputs import eligible_subjects


def test_complete_case_rule_preserves_all_qualified_subjects_and_record_gate():
    coverage=[dict(subject=str(i),qualified_records=8,windows=100) for i in range(13)]
    coverage[-1]['qualified_records']=7
    records=[dict(subject=str(i),accepted=j<8) for i in range(13) for j in range(10)]
    subjects,passed=eligible_subjects(dict(coverage=coverage,records=records))
    assert passed and set(subjects)=={str(i) for i in range(12)}
    records.append(dict(subject='0',accepted=False))
    assert not eligible_subjects(dict(coverage=coverage,records=records))[1]
    coverage[0]['windows']=99
    assert not eligible_subjects(dict(coverage=coverage,records=records))[1]
