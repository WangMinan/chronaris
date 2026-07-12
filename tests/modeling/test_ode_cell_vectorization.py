from __future__ import annotations

import copy

import torch
from torchdiffeq import odeint

from chronaris.models.alignment.ode_cells import ODERNNCell


def _legacy_rk4(cell, hidden_state, delta_t_s):
    rows = []
    zero = hidden_state.new_zeros(())
    for index in range(len(hidden_state)):
        delta = delta_t_s[index].clamp_min(0)
        if bool(delta <= 0):
            rows.append(hidden_state[index])
            continue
        trajectory = odeint(
            cell.ode_func,
            hidden_state[index].unsqueeze(0),
            torch.stack((zero, delta)),
            method="rk4",
            rtol=cell.ode_rtol,
            atol=cell.ode_atol,
        )
        rows.append(trajectory[-1, 0])
    return torch.stack(rows)


def _legacy_gru(cell, hidden_state, embedding, mask):
    rows = []
    for index in range(len(hidden_state)):
        if bool(mask[index]):
            rows.append(cell.observation_update(embedding[index], hidden_state[index]))
        else:
            rows.append(hidden_state[index])
    return torch.stack(rows)


def _cell() -> ODERNNCell:
    return ODERNNCell(
        5,
        hidden_dim=4,
        dynamics_hidden_dim=7,
        activation="gelu",
        ode_method="rk4",
    )


def test_batched_rk4_matches_per_sample_torchdiffeq_output_and_gradients() -> None:
    torch.manual_seed(17)
    batched = _cell()
    legacy = copy.deepcopy(batched)
    hidden = torch.randn(6, 4, requires_grad=True)
    legacy_hidden = hidden.detach().clone().requires_grad_(True)
    delta = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0, -0.2])

    actual = batched.evolve_hidden_state(hidden, delta)
    expected = _legacy_rk4(legacy, legacy_hidden, delta)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(),
        (hidden, *batched.ode_func.parameters()),
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (legacy_hidden, *legacy.ode_func.parameters()),
    )
    for value, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(value, reference, rtol=3e-5, atol=2e-6)


def test_batched_gru_update_matches_per_sample_output_and_gradients() -> None:
    torch.manual_seed(29)
    batched = _cell()
    legacy = copy.deepcopy(batched)
    hidden = torch.randn(6, 4, requires_grad=True)
    embedding = torch.randn(6, 5, requires_grad=True)
    legacy_hidden = hidden.detach().clone().requires_grad_(True)
    legacy_embedding = embedding.detach().clone().requires_grad_(True)
    mask = torch.tensor([True, False, True, True, False, True])

    actual = batched.update_hidden_state(hidden, embedding, mask)
    expected = _legacy_gru(legacy, legacy_hidden, legacy_embedding, mask)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(),
        (hidden, embedding, *batched.observation_update.parameters()),
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (
            legacy_hidden,
            legacy_embedding,
            *legacy.observation_update.parameters(),
        ),
    )
    for value, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(value, reference, rtol=3e-5, atol=2e-6)
