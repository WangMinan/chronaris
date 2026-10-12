import pytest

from chronaris.evaluation.application_tasks.local_validation_recovery import recovered_timing


def test_recovered_cost_keeps_previous_work_without_counting_powered_off_time():
    manifest = dict(command=['python', 'train'], elapsed_before_outage_lower_bound_s=8100., manifest_path='receipt.json')
    result = recovered_timing(manifest, started=999999., elapsed=3600., peak=6400, exit_code=0, log='run.log')
    assert result['wall_seconds'] == 11700.
    assert result['resumed_wall_seconds'] == 3600.
    assert result['pre_outage_seconds_lower_bound'] == 8100.
    assert result['exit_code'] == 0
    with pytest.raises(ValueError, match='invalid'):
        recovered_timing(manifest, started=999999., elapsed=-1., peak=0, exit_code=1, log='run.log')
