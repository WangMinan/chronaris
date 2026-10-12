from copy import deepcopy

import pytest

from chronaris.evaluation.application_tasks.device_replay import prediction_changes, ranking_changes


def test_replay_preserves_numeric_failure_and_detects_class_flip():
    a = [dict(sample_id='a', task='classification', field=0, target_valid=True, target=0,
              group_id='g', no_observation=False, sample_weight=1., prediction=0, probability=[.51, .49])]
    b = deepcopy(a)
    b[0].update(prediction=1, probability=[.49, .51])
    result = prediction_changes(a, b)
    assert result['class_flips'] == 1 and not result['strict']['passed']
    b[0]['target'] = 1
    with pytest.raises(ValueError, match='context'):
        prediction_changes(a, b)
    with pytest.raises(ValueError, match='coverage'):
        prediction_changes(a, a+a)


def test_rankings_detect_reversal_and_tie_change():
    rows = [dict(domain='clare', route='self_supervised', family='minirocket', method=m,
                 metrics=[dict(task='load', metric='rmse', before=a, after=b)])
            for m, a, b in [('a', 1., 2.), ('b', 2., 1.), ('c', 2., 2.)]]
    result = ranking_changes(rows)[0]
    assert len(result['changed_pairs']) == 3
