import json

import pytest

from chronaris.evaluation.application_tasks.comparison_accounting import attempt_accounting


def test_interrupted_parent_cost_is_a_lower_bound_without_double_counting(tmp_path):
    unit = dict(domain='cogpilot', method='chronaris')
    current, parent = tmp_path/'current.json', tmp_path/'parent.json'
    current.write_text(json.dumps(unit | {'seconds': 100.}))
    parent.write_text(json.dumps(unit | {'seconds': 900.}))
    migration = dict(parent_attempt_costs={}, preserved_training_seconds=800.)
    result = attempt_accounting(unit, [current, current], migration)
    assert result['observed_attempt_seconds'] == 100.
    assert result['historical_seconds_lower_bound'] == 900.
    assert not result['historical_cost_complete']
    migration['parent_attempt_costs'] = {str(parent): 'bound elsewhere'}
    result = attempt_accounting(unit, [current, parent], migration)
    assert result['unreceipted_checkpoint_training_seconds'] == 0.
    assert result['historical_seconds_lower_bound'] == 1000.
    parent.write_text(json.dumps(unit | {'seconds': float('nan')}))
    with pytest.raises(ValueError, match='invalid'):
        attempt_accounting(unit, [parent], migration)


def test_closeout_preserves_group_weighting_and_rejects_incomplete_learning():
    from chronaris.evaluation.application_tasks.comparison_closeout import check_summary, learning_summary
    evaluation = dict(group_metrics=[
        dict(task='score', group_id='a', status='completed', rmse=1.),
        dict(task='score', group_id='a', status='completed', rmse=3.),
        dict(task='score', group_id='b', status='completed', rmse=8.)],
        task_summary=[dict(task='score', metric='rmse', group_count=2, value=5.)])
    check_summary(evaluation)
    evaluation['task_summary'][0]['value'] = 4.
    with pytest.raises(ValueError, match='grouped'):
        check_summary(evaluation)
    row = dict(training=dict(status='completed', optimizer_updates=300,
        epoch_rows=[dict(public_selection_loss=1., mean_gradient_norm_before_clip=float('nan'))]),
        resume_identical=True, nonconstant_dimensions=64)
    with pytest.raises(ValueError, match='nonfinite'):
        learning_summary(row)
