"""Pinned REG embeddings retaining observed normalization statistics and history levels."""
import torch
from chronaris.modeling.fusion_encoders.recent_models import Chronos2WindowEncoder


def pack_observed_states(embeds,loc_scales,values,mask):
    states=torch.stack([e[:,-2,:] for e in embeds]).float()
    channels=values.shape[-1]
    loc=torch.stack([p[0].reshape(channels,1) for p in loc_scales]).float()
    scale=torch.stack([p[1].reshape(channels,1) for p in loc_scales]).float()
    valid=mask.cpu().any(1)
    packed=torch.cat((states,loc,scale),-1).masked_fill(~valid[...,None],0).flatten(1)
    observed=values[:,-1,-4:].cpu().masked_fill(~mask[:,-1,-4:].cpu(),0)
    result=torch.cat((packed,observed),1)
    if not torch.isfinite(result).all():raise ValueError('nonfinite complete Chronos representation')
    return result


class ChronosInformationEncoder(Chronos2WindowEncoder):
    def __init__(self,assets,channel_count):
        super().__init__(assets,channel_count)
        self.feature_dim=(self.core.config.d_model+2)*channel_count+4

    @torch.no_grad()
    def forward(self,values,mask,prompts=None):
        channels=values.transpose(1,2).masked_fill(~mask.transpose(1,2),torch.nan)
        embeds,scales=self.pipeline.embed(channels,batch_size=channels.shape[1],context_length=96)
        return pack_observed_states(embeds,scales,values,mask)
