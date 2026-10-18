"""Zero-initialized task correction over shared causal observation backbones."""
from dataclasses import replace
import torch
from torch import nn

from chronaris.modeling.fusion_encoders.single_stream import ContinuousTimeSingleStreamEncoder, SingleStreamEncoderConfig
from chronaris.modeling.fusion_encoders.deep_baselines import (
    CausalContiFormerFusionEncoder, CausalMulTFusionEncoder, DeepBaselineEncoderConfig,
)


class FairResidualModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'aircraft':
            self.encoder = ContinuousTimeSingleStreamEncoder(SingleStreamEncoderConfig('vehicle', 12))
        elif method in ('contiformer', 'mult', 'mult_aircraft', 'control', 'candidate'):
            name = 'mult' if method in ('mult', 'mult_aircraft') else 'contiformer'
            cls = CausalMulTFusionEncoder if name == 'mult' else CausalContiFormerFusionEncoder
            self.encoder = cls(DeepBaselineEncoderConfig(name, 16, 12))
        else:
            raise ValueError(method)
        self.head = nn.Linear(64, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, physiology, vehicle):
        if self.method == 'aircraft':
            output = self.encoder.forward_queried(vehicle)
        else:
            if self.method in ('control', 'mult_aircraft'):
                mask = torch.zeros_like(physiology.feature_mask)
                physiology = replace(physiology, values=torch.zeros_like(physiology.values), feature_mask=mask,
                                     modality_mask=mask.any(-1), observation_age_s=torch.full_like(physiology.values, torch.inf),
                                     source_indices=torch.full_like(physiology.source_indices, -1))
            output = self.encoder.forward_queried(physiology, vehicle)
        # Preserve observed performance levels even when training only a residual.
        sequence = torch.cat((vehicle.values[..., 8:12], output.sequence_embedding[..., :60]), dim=-1)
        if not torch.isfinite(sequence).all():
            raise ValueError('nonfinite residual representation')
        return sequence, self.head(sequence).squeeze(-1)
