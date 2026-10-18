"""Minimal dual-stream baseline: existing strong branches and one projection."""
import torch
from torch import nn

from chronaris.modeling.fusion_encoders.single_stream import (
    ContinuousTimeSingleStreamEncoder, SingleStreamEncoderConfig, SingleStreamEncoding,
)


class SameBackboneFusion(nn.Module):
    def __init__(self, physiology_dim, vehicle_dim):
        super().__init__()
        self.physiology = ContinuousTimeSingleStreamEncoder(SingleStreamEncoderConfig('physiology', physiology_dim))
        self.vehicle = ContinuousTimeSingleStreamEncoder(SingleStreamEncoderConfig('vehicle', vehicle_dim))
        self.projection = nn.Linear(128, 64)

    def forward(self, batch):
        a, b = self.physiology(batch), self.vehicle(batch)
        return self._merge(a, b)

    def forward_queried(self, physiology, vehicle):
        return self._merge(self.physiology.forward_queried(physiology), self.vehicle.forward_queried(vehicle))

    def _merge(self, a, b):
        valid = a.valid_mask | b.valid_mask
        sequence = self.projection(torch.cat((a.sequence_embedding, b.sequence_embedding), dim=-1))
        return SingleStreamEncoding(sequence.masked_fill(~valid.unsqueeze(-1), 0), valid)
