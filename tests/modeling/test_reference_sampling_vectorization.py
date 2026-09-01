from __future__ import annotations

import copy

import torch

from chronaris.models.alignment.config import AlignmentPrototypeConfig
from chronaris.models.alignment.prototype import SingleStreamODERNNPrototype
from chronaris.models.alignment.torch_batch import TorchAlignmentStreamBatch


def _stream() -> TorchAlignmentStreamBatch:
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    offsets = torch.tensor([[0.0, 1.0, 3.0, 0.0], [0.5, 2.0, 0.0, 0.0]])
    return TorchAlignmentStreamBatch(
        values=torch.zeros((2, 4, 1)),
        mask=mask,
        feature_valid_mask=mask.unsqueeze(-1),
        offsets_ms=(offsets * 1000).to(torch.int64),
        offsets_s=offsets,
        delta_t_s=torch.tensor([[0.0, 1.0, 2.0, 0.0], [0.5, 1.5, 0.0, 0.0]]),
        point_counts=torch.tensor([3, 2]),
        feature_names=("x",),
    )


def _legacy_sample(model, stream, updated_states, references):
    rows = []
    valid_rows = []
    positive = int((((stream.delta_t_s > 0) & stream.mask).sum()).item())
    maximum = float(stream.delta_t_s[stream.mask].max())
    for sample_index in range(len(stream.values)):
        sample_rows = []
        sample_valid = []
        for reference in references[sample_index]:
            eligible = torch.nonzero(
                stream.mask[sample_index]
                & (stream.offsets_s[sample_index] <= reference),
                as_tuple=False,
            ).flatten()
            if len(eligible) == 0:
                sample_rows.append(torch.zeros(model.config.hidden_dim))
                sample_valid.append(torch.tensor(False))
                continue
            source = int(eligible[-1])
            delta = torch.clamp(reference - stream.offsets_s[sample_index, source], min=0)
            if bool(delta > 0):
                positive += 1
                maximum = max(maximum, float(delta))
            state = model.ode_rnn_cell.evolve_hidden_state(
                updated_states[sample_index, source].unsqueeze(0),
                delta.reshape(1),
            )[0]
            sample_rows.append(state)
            sample_valid.append(torch.tensor(True))
        rows.append(torch.stack(sample_rows))
        valid_rows.append(torch.stack(sample_valid))
    return torch.stack(rows), torch.stack(valid_rows), positive, maximum


def test_vectorized_reference_sampling_matches_legacy_output_and_gradients() -> None:
    torch.manual_seed(17)
    config = AlignmentPrototypeConfig(
        hidden_dim=4,
        embedding_dim=3,
        encoder_hidden_dim=5,
        decoder_hidden_dim=5,
        dynamics_hidden_dim=6,
        projection_dim=4,
        ode_method="euler",
    )
    vectorized_model = SingleStreamODERNNPrototype(1, config=config)
    legacy_model = copy.deepcopy(vectorized_model)
    stream = _stream()
    references = torch.tensor(
        [[0.0, 0.5, 1.0, 2.0, 4.0], [0.0, 0.5, 1.0, 2.0, 3.0]]
    )
    states_vectorized = torch.randn((2, 4, 4), requires_grad=True)
    states_legacy = states_vectorized.detach().clone().requires_grad_(True)

    vectorized = vectorized_model._sample_reference_hidden_states(
        stream, states_vectorized, references
    )
    legacy = _legacy_sample(legacy_model, stream, states_legacy, references)

    torch.testing.assert_close(vectorized[0], legacy[0], rtol=1e-6, atol=1e-7)
    assert torch.equal(vectorized[1], legacy[1])
    assert vectorized[2] == legacy[2]
    assert vectorized[3] == legacy[3]
    vectorized_gradients = torch.autograd.grad(
        vectorized[0].square().sum(),
        (states_vectorized, *vectorized_model.ode_rnn_cell.parameters()),
        allow_unused=True,
    )
    legacy_gradients = torch.autograd.grad(
        legacy[0].square().sum(),
        (states_legacy, *legacy_model.ode_rnn_cell.parameters()),
        allow_unused=True,
    )
    for actual, expected in zip(vectorized_gradients, legacy_gradients, strict=True):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_masked_euler_substeps_match_manual_updates_and_keep_gradients_finite() -> None:
    torch.manual_seed(29)
    config = AlignmentPrototypeConfig(
        hidden_dim=4,
        embedding_dim=3,
        encoder_hidden_dim=5,
        decoder_hidden_dim=5,
        dynamics_hidden_dim=6,
        projection_dim=4,
        ode_method="euler",
        max_ode_step_s=0.5,
    )
    model = SingleStreamODERNNPrototype(1, config=config)
    initial = torch.randn(3, 4, requires_grad=True)
    deltas = torch.tensor([0.0, 0.5, 1.2])

    actual = model.ode_rnn_cell.evolve_hidden_state(initial, deltas)
    expected = initial
    counts = torch.tensor([0, 1, 3])
    sizes = torch.tensor([0.0, 0.5, 0.4])
    for step in range(3):
        proposal = expected + sizes[:, None] * model.ode_rnn_cell.ode_func(
            expected.new_zeros(()), expected
        )
        expected = torch.where((counts > step)[:, None], proposal, expected)

    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(
        actual.square().sum(),
        (initial, *model.ode_rnn_cell.ode_func.parameters()),
    )
    assert all(torch.isfinite(value).all() for value in gradients)


def test_batched_observation_update_matches_rowwise_output_and_gradients() -> None:
    torch.manual_seed(43)
    config = AlignmentPrototypeConfig(
        hidden_dim=4,
        embedding_dim=3,
        encoder_hidden_dim=5,
        decoder_hidden_dim=5,
        dynamics_hidden_dim=6,
        projection_dim=4,
        ode_method="euler",
    )
    batched = SingleStreamODERNNPrototype(1, config=config).ode_rnn_cell
    rowwise = copy.deepcopy(batched)
    hidden = torch.randn(4, 4, requires_grad=True)
    hidden_rowwise = hidden.detach().clone().requires_grad_(True)
    observations = torch.randn(4, 3, requires_grad=True)
    observations_rowwise = observations.detach().clone().requires_grad_(True)
    mask = torch.tensor([True, False, True, False])

    actual = batched.update_hidden_state(hidden, observations, mask)
    expected = torch.stack(
        [
            rowwise.observation_update(observations_rowwise[index], hidden_rowwise[index])
            if mask[index]
            else hidden_rowwise[index]
            for index in range(len(mask))
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(),
        (hidden, observations, *batched.parameters()),
        allow_unused=True,
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (hidden_rowwise, observations_rowwise, *rowwise.parameters()),
        allow_unused=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is expected_gradient
        else:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=1e-5,
                atol=1e-6,
            )


def test_batched_observation_diagnostics_match_pointwise_output_and_gradients() -> None:
    torch.manual_seed(47)
    config = AlignmentPrototypeConfig(
        hidden_dim=4,
        embedding_dim=3,
        encoder_hidden_dim=5,
        decoder_hidden_dim=5,
        dynamics_hidden_dim=6,
        projection_dim=4,
    )
    batched = SingleStreamODERNNPrototype(2, config=config)
    pointwise = copy.deepcopy(batched)
    states = torch.randn(3, 5, 4, requires_grad=True)
    states_pointwise = states.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    scale = mask.unsqueeze(-1).to(states.dtype)

    actual = (
        batched.decoder(states) * scale,
        batched.projection_head(states) * scale,
    )
    expected = (
        torch.stack(
            [pointwise.decoder(states_pointwise[:, index]) for index in range(5)],
            dim=1,
        )
        * scale,
        torch.stack(
            [
                pointwise.projection_head(states_pointwise[:, index])
                for index in range(5)
            ],
            dim=1,
        )
        * scale,
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-6, atol=1e-7)
    actual_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in actual),
        (states, *batched.decoder.parameters(), *batched.projection_head.parameters()),
    )
    expected_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in expected),
        (
            states_pointwise,
            *pointwise.decoder.parameters(),
            *pointwise.projection_head.parameters(),
        ),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-5,
            atol=1e-6,
        )
