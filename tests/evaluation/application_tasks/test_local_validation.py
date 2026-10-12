from chronaris.evaluation.application_tasks.local_validation import calibrated_budget


def test_full_fold_budget_replaces_short_probe_scale_without_claiming_ablation_measurement():
    prior = {'rows': [dict(case=case, fold=fold, hours=hours)
                     for case, hours in [('clare_batched', 1.), ('clare_no_event', 2.), ('cogpilot_batched', 4.)]
                     for fold in range(5)] + [dict(case='sim_no_event', hours=3.)]}
    result = calibrated_budget(prior, {'0': {'wall_seconds': 7200}, '10': {'wall_seconds': 7200}})
    assert result['calibration_factors'] == {'clare_batched': 2., 'clare_no_event': 2., 'cogpilot_batched': .5}
    assert result['first_batch_compute_hours'] == 43
    assert result['remaining_compute_hours'] == 39
    assert result['public_45_unit_compute_hours'] == 120
