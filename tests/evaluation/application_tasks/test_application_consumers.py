from __future__ import annotations

import numpy as np
import pytest
import torch

from chronaris.evaluation.application_tasks.application_consumers import (
    CausalTCNEmissionModel,
    LinearConsumerConfig,
    LinearFrozenConsumer,
    MiniRocketConsumerConfig,
    MiniRocketFrozenConsumer,
    TCNConsumerConfig,
    duration_constrained_viterbi_decode,
    fit_causal_tcn_emission,
    fit_duration_viterbi_parameters,
)


def test_linear_consumer_uses_fixed_pooled_state_models() -> None:
    rng = np.random.default_rng(17)
    pooled = rng.normal(size=(12, 8)).astype(np.float32)
    class_target = np.asarray([index % 3 for index in range(12)])
    regression_target = np.linspace(0.0, 1.0, 12)
    prediction = LinearFrozenConsumer().fit(
        pooled[:9], class_target[:9], regression_target[:9]
    ).predict(pooled[9:])

    assert prediction["class_prediction"].shape == (3,)
    assert prediction["class_probability"].shape == (3, 3)
    assert prediction["regression_prediction"].shape == (3,)


def test_linear_consumer_selects_only_from_fixed_validation_grid() -> None:
    rng = np.random.default_rng(19)
    pooled = rng.normal(size=(18, 8)).astype(np.float32)
    classes = np.asarray([index % 3 for index in range(18)])
    regression = np.linspace(-1.0, 1.0, 18)
    config = LinearConsumerConfig(tune_on_validation=True)
    consumer = LinearFrozenConsumer(config).fit(
        pooled[:12],
        classes[:12],
        regression[:12],
        validation_pooled=pooled[12:],
        validation_class_target=classes[12:],
        validation_regression_target=regression[12:],
    )

    assert consumer.selected_classification_c in config.classification_c_grid
    assert consumer.selected_regression_alpha in config.regression_alpha_grid


def test_minirocket_uses_channel_first_transform_and_fixed_consumers() -> None:
    pytest.importorskip("aeon")
    rng = np.random.default_rng(17)
    sequence = rng.normal(size=(12, 32, 4)).astype(np.float32)
    class_target = np.asarray([index % 3 for index in range(12)])
    regression_target = np.linspace(0.0, 1.0, 12)
    consumer = MiniRocketFrozenConsumer(
        MiniRocketConsumerConfig(n_kernels=84, random_state=17)
    ).fit(sequence[:9], class_target[:9], regression_target[:9])
    prediction = consumer.predict(sequence[9:])

    assert prediction["class_prediction"].shape == (3,)
    assert prediction["class_probability"].shape == (3, 3)
    assert prediction["regression_prediction"].shape == (3,)
    assert prediction["transformed_feature_count"] > 0


def test_minirocket_train_only_variance_filter_removes_constant_channels() -> None:
    pytest.importorskip("aeon")
    rng = np.random.default_rng(29)
    sequence = rng.normal(size=(12, 32, 4)).astype(np.float32)
    sequence[:, :, 3] = 2.0
    class_target = np.asarray([index % 3 for index in range(12)])
    regression_target = np.linspace(0.0, 1.0, 12)
    consumer = MiniRocketFrozenConsumer(
        MiniRocketConsumerConfig(n_kernels=84)
    ).fit(sequence[:9], class_target[:9], regression_target[:9])
    prediction = consumer.predict(sequence[9:])

    assert prediction["input_channel_count"] == 3
    assert consumer.channel_indices.tolist() == [0, 1, 2]


def test_causal_tcn_future_perturbation_does_not_change_past_logits() -> None:
    torch.manual_seed(17)
    model = CausalTCNEmissionModel(
        TCNConsumerConfig(input_dim=4, hidden_channels=8, class_count=3, dropout=0.0)
    ).eval()
    values = torch.randn(2, 20, 4)
    changed = values.clone()
    changed[:, 12:] += 10_000
    first = model(values)
    second = model(changed)

    assert torch.allclose(first[:, :12], second[:, :12], atol=1e-6, rtol=1e-6)
    assert not torch.equal(first[:, 12:], second[:, 12:])


def test_tcn_training_and_viterbi_use_train_labels_only() -> None:
    torch.manual_seed(23)
    sequence = torch.randn(6, 24, 4)
    labels = torch.tensor(
        [[0] * 6 + [1] * 6 + [2] * 6 + [0] * 6 for _ in range(6)]
    )
    result = fit_causal_tcn_emission(
        sequence,
        labels,
        config=TCNConsumerConfig(
            input_dim=4,
            hidden_channels=8,
            class_count=3,
            dropout=0.0,
            epochs=2,
        ),
    )
    parameters = fit_duration_viterbi_parameters(labels, class_count=3)
    with torch.inference_mode():
        logits = result.model(sequence[:2])
    decoded = duration_constrained_viterbi_decode(logits, parameters)

    assert len(result.training_rows) == 2
    assert torch.isfinite(logits).all()
    assert parameters.train_sequence_count == 6
    assert decoded.shape == (2, 24)
    assert set(decoded.unique().tolist()).issubset({0, 1, 2})


def test_tcn_uses_validation_loss_for_early_stopping() -> None:
    torch.manual_seed(31)
    train_sequence = torch.randn(6, 24, 4)
    train_labels = torch.tensor(
        [[0] * 8 + [1] * 8 + [2] * 8 for _ in range(6)]
    )
    validation_sequence = torch.randn(2, 24, 4)
    validation_labels = torch.tensor([[2] * 24, [1] * 24])
    result = fit_causal_tcn_emission(
        train_sequence,
        train_labels,
        config=TCNConsumerConfig(
            input_dim=4,
            hidden_channels=8,
            class_count=3,
            dropout=0.0,
            epochs=5,
            patience=2,
        ),
        validation_sequence=validation_sequence,
        validation_labels=validation_labels,
    )

    assert result.best_epoch >= 1
    assert all(row["selection_role"] == "validation" for row in result.training_rows)
    assert len(result.training_rows) <= 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tcn_can_train_on_gpu_and_returns_portable_cpu_model() -> None:
    sequence = torch.randn(4, 16, 4)
    labels = torch.tensor([[0] * 8 + [1] * 8 for _ in range(4)])
    result = fit_causal_tcn_emission(
        sequence,
        labels,
        config=TCNConsumerConfig(
            input_dim=4,
            hidden_channels=8,
            class_count=2,
            dropout=0.0,
            epochs=1,
            device="cuda",
        ),
    )

    assert next(result.model.parameters()).device.type == "cpu"
    assert torch.isfinite(result.model(sequence[:1])).all()


def test_tcn_training_is_seed_reproducible_with_dropout() -> None:
    torch.manual_seed(41)
    sequence = torch.randn(6, 24, 4)
    labels = torch.tensor(
        [[0] * 6 + [1] * 6 + [2] * 6 + [0] * 6 for _ in range(6)]
    )
    config = TCNConsumerConfig(
        input_dim=4,
        hidden_channels=8,
        class_count=3,
        dropout=0.2,
        epochs=2,
        seed=17,
    )
    first = fit_causal_tcn_emission(sequence, labels, config=config)
    torch.rand(100)
    second = fit_causal_tcn_emission(sequence, labels, config=config)

    assert first.training_rows == second.training_rows
    assert all(
        torch.equal(first.model.state_dict()[name], second.model.state_dict()[name])
        for name in first.model.state_dict()
    )
