import numpy as np
from chronaris.evaluation.application_tasks.rest_development_inputs import resting_values


def test_rest_reference_retains_level_and_masks_without_changing_waveform_statistics():
    native=np.zeros((2,96,16));native[:,:,1::4]=2;native[:,:,2::4]=3;native[:,:,3::4]=.5
    masks=np.ones_like(native,dtype=bool);masks[:,0]=False
    values=resting_values(native,masks,np.array([[10]*4,[20]*4]),np.ones((2,4))*5,np.ones((2,4))*2,[1]*4)
    np.testing.assert_allclose(values[0,1:,::4],np.arcsinh(2.5));np.testing.assert_allclose(values[1,1:,::4],np.arcsinh(7.5))
    np.testing.assert_array_equal(values[0,:,1::4],values[1,:,1::4])
    assert (values[:,0]==0).all() and (values[:,1:,3::4]==.5).all()
    assert (native[:,:,::4]==0).all()


def test_rest_training_waits_for_verified_predecessor_and_does_not_replace_success(tmp_path,monkeypatch):
    import json,pytest
    from chronaris.evaluation.application_tasks import shape_mult_run as run
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:'test-commit')
    report=tmp_path/'rest';report.mkdir();assets=tmp_path/'assets';assets.mkdir()
    (report/'PLAN.md').write_text('test');(report/'protected_sources.json').write_text('{}')
    (report/'qualification.json').write_text(json.dumps(dict(status='passed',condition='preflight_unlabelled_rest_observation_available')))
    prior=tmp_path/'docs/review/stage/eligible-fit-20261019';prior.mkdir(parents=True)
    (prior/'verification.json').write_text(json.dumps(dict(status='passed')))
    for result in [dict(status='running'),dict(status='completed',representation_success=True,pairing_success=True)]:
        (prior/'run_status.json').write_text(json.dumps(result))
        with pytest.raises(AssertionError):run.run_shape_mult(report,assets,selection_policy='common_readout',rest_calibration=True)
        assert not (report/'training_binding.json').exists() and not (report/'run_status.json').exists()
