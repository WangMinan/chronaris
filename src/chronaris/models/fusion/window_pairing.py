"""Independent modality histories for window-level cross-stream pairing."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WindowPairingOutput:
    physiology: torch.Tensor
    vehicle: torch.Tensor
    physiology_valid: torch.Tensor
    vehicle_valid: torch.Tensor


class IndependentWindowPairing(nn.Module):
    """Four actual-time bins per modality, followed by separate 32-D projections."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.physiology_projection = nn.Linear(4 * hidden_dim, 32)
        self.vehicle_projection = nn.Linear(4 * hidden_dim, 32)

    def forward(self, alignment, duration_s):
        vectors, validity = {}, {}
        for name in ("physiology", "vehicle"):
            stream = getattr(alignment, name)
            states = stream.updated_hidden_states
            duration = duration_s.to(stream.offsets_s.device)
            if duration.shape != (states.shape[0],) or not torch.isfinite(duration).all() or not (duration > 0).all():
                raise ValueError("pairing requires each window's finite positive duration")
            valid = stream.mask & (stream.offsets_s >= 0) & (stream.offsets_s < duration[:, None])
            if not torch.isfinite(states[valid]).all():
                raise ValueError("valid independent history states are non-finite")
            bins = (stream.offsets_s / duration[:, None] * 4).floor().long().clamp(0, 3)
            weights = F.one_hot(bins, 4).to(states.dtype) * valid[:, :, None]
            pooled = torch.einsum("btq,bth->bqh", weights, states.masked_fill(~valid[:, :, None], 0))
            pooled = pooled / weights.sum(dim=1)[:, :, None].clamp_min(1)
            available = valid.any(dim=1)
            vectors[name] = getattr(self, name + "_projection")(pooled.flatten(1)).masked_fill(~available[:, None], 0)
            validity[name] = available
        return WindowPairingOutput(vectors["physiology"], vectors["vehicle"], validity["physiology"], validity["vehicle"])
