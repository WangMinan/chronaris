from copy import deepcopy

import pytest

from chronaris.evaluation.application_tasks.pipeline_pause import paused_state


def test_requested_pause_preserves_completed_work_and_real_failure_history():
    old = {'stage': 'older', 'error': 'original failure'}
    before = dict(completed={'done': {'sha256': 'unchanged'}}, failures=[old], current_stage='active')
    stopped = dict(before, status='failed', children={}, error='signal', failures=[old, dict(
        stage='active', error='Traceback\nKeyboardInterrupt: pipeline termination requested\n')])
    result = paused_state(before, stopped, 'pause.json')
    assert result['status'] == 'paused_by_user' and result['failures'] == [old]
    assert result['completed'] == before['completed'] and 'error' not in result
    assert stopped['status'] == 'failed' and len(stopped['failures']) == 2
    broken = deepcopy(stopped)
    broken['failures'][-1]['error'] = 'RuntimeError: CUDA failure'
    with pytest.raises(ValueError, match='not solely'):
        paused_state(before, broken, 'pause.json')
    broken = deepcopy(stopped)
    broken['completed']['done']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='boundary changed'):
        paused_state(before, broken, 'pause.json')
