import json
import sys
import time

import pytest

from chronaris.evaluation.application_tasks.semantic_qualification import run_bounded


def test_shared_deadline_rejects_expired_work_and_records_timeout(tmp_path):
    deadline = tmp_path/'deadline.json'
    deadline.write_text(json.dumps({'deadline_unix_s': time.time()-1}))
    with pytest.raises(TimeoutError, match='deadline reached'):
        run_bounded([sys.executable, '-c', 'pass'], deadline_path=deadline, log_path=tmp_path/'expired.log')
    assert not (tmp_path/'expired.log').exists()
    deadline.write_text(json.dumps({'deadline_unix_s': time.time()+10}))
    with pytest.raises(RuntimeError, match='command stopped'):
        run_bounded([sys.executable, '-c', 'import time; time.sleep(5)'],
                    deadline_path=deadline, log_path=tmp_path/'bounded.log', max_seconds=.1)
    result = json.loads((tmp_path/'bounded.receipt.json').read_text())
    assert result['timed_out'] and result['exit_code'] != 0
