import torch
from chronaris.modeling.fusion_encoders.chronos_information import pack_observed_states


def test_packing_preserves_observed_level_scale_and_masks_missing_channels():
    values=torch.arange(2*6*5,dtype=torch.float32).reshape(2,6,5)
    mask=torch.ones_like(values,dtype=torch.bool);mask[0,:,0]=False
    embeds=[torch.ones(5,8,3),torch.ones(5,8,3)*2]
    scales=[(torch.arange(5.)[:,None],torch.ones(5,1)*3)]*2
    out=pack_observed_states(embeds,scales,values,mask)
    assert out.shape==(2,29)
    assert torch.equal(out[0,:5],torch.zeros(5))
    torch.testing.assert_close(out[:,-4:],values[:,-1,-4:])
    torch.testing.assert_close(out[1,:25].reshape(5,5)[:,-2],torch.arange(5.))
    torch.testing.assert_close(out[1,:25].reshape(5,5)[:,-1],torch.ones(5)*3)
