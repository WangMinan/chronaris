from __future__ import annotations

from dataclasses import dataclass

import torch

from chronaris.modeling.training.chronaris_auxiliary import lag_aware_alignment_loss


@dataclass
class _Stream:
    reference_offsets_s: torch.Tensor
    reference_projected_states: torch.Tensor
    reference_valid_mask: torch.Tensor


@dataclass
class _Alignment:
    physiology: _Stream
    vehicle: _Stream


def _make_alignment(*, shift_steps: int) -> _Alignment:
    torch.manual_seed(7)
    batch, timepoints, hidden = 2, 96, 16
    # Vehicle signal; physiology is the SAME signal lagged by shift_steps (causal).
    vehicle_states = torch.randn(batch, timepoints, hidden)
    physiology_states = torch.zeros_like(vehicle_states)
    physiology_states[:, shift_steps:] = vehicle_states[:, :-shift_steps]
    times = torch.linspace(0.0, 30.0, timepoints).unsqueeze(0).expand(batch, -1)
    valid = torch.ones(batch, timepoints, dtype=torch.bool)
    return _Alignment(
        physiology=_Stream(times, physiology_states, valid),
        vehicle=_Stream(times, vehicle_states, valid),
    )


def test_lag_aware_loss_recovers_known_lag_and_beats_same_time_cosine() -> None:
    shift_steps = 10  # physiology lags vehicle by ~3 s on a 30 s / 96 grid
    alignment = _make_alignment(shift_steps=shift_steps)
    result = lag_aware_alignment_loss(alignment, min_lag_s=0.0, max_lag_s=10.0)
    # Most physiology points (t >= shift) find their causal lag and align near-perfectly;
    # the leading zero-padded points (t < shift) have no true match, so the mean loss is
    # small but not zero.
    assert float(result.loss.detach()) < 0.2
    assert result.count > 0
    assert result.best_lag_index is not None

    # Contrast: the original same-time cosine (1 - cos(phys_t, veh_t)) is high because
    # physiology is shifted, so the lag-aware loss is substantially smaller.
    from torch.nn.functional import cosine_similarity

    same_time = (
        1.0 - cosine_similarity(
            alignment.physiology.reference_projected_states,
            alignment.vehicle.reference_projected_states,
            dim=-1,
        )
    ).mean()
    assert float(same_time.detach()) > float(result.loss.detach()) + 0.3


def test_lag_window_too_narrow_misses_the_lag() -> None:
    """If the lag window excludes the true lag, the loss should be larger."""
    shift_steps = 10  # ~3 s lag
    alignment = _make_alignment(shift_steps=shift_steps)
    narrow = lag_aware_alignment_loss(alignment, min_lag_s=0.0, max_lag_s=0.5)
    wide = lag_aware_alignment_loss(alignment, min_lag_s=0.0, max_lag_s=10.0)
    assert float(narrow.loss.detach()) > float(wide.loss.detach())


def test_lag_aware_loss_zero_window_is_error() -> None:
    import pytest

    alignment = _make_alignment(shift_steps=1)
    with pytest.raises(ValueError):
        lag_aware_alignment_loss(alignment, min_lag_s=5.0, max_lag_s=5.0)
