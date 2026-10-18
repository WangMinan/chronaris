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
