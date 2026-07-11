"""Method-invariant MiniRocket, causal TCN, and duration-constrained decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chronaris.evaluation.application_tasks.consumer_model_selection import (
    classifier_classes,
    fit_classifier,
    fit_regressor,
)


@dataclass(frozen=True, slots=True)
class MiniRocketConsumerConfig:
    n_kernels: int = 10_000
    random_state: int = 17
    n_jobs: int = 1
    classification_c: float = 1.0
    regression_alpha: float = 1.0
    classification_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    regression_alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    tune_on_validation: bool = False
    minimum_case_channel_std: float = 1e-7
    classification_solver: str = "liblinear_ovr"


class MiniRocketFrozenConsumer:
    """One train-role MiniRocket transform shared by classification/regression."""

    def __init__(self, config: MiniRocketConsumerConfig | None = None) -> None:
        self.config = config or MiniRocketConsumerConfig()
        self.transformer = None
        self.classifier = None
        self.regressor = None
        self.channel_indices = None
        self.selected_classification_c = None
        self.selected_regression_alpha = None

    def fit(
        self,
        sequence,
        class_target,
        regression_target,
        *,
        validation_sequence=None,
        validation_class_target=None,
        validation_regression_target=None,
    ):
        from aeon.transformations.collection.convolution_based import MiniRocket

        values = _as_collection(sequence)
        case_channel_std = values.std(axis=-1)
        self.channel_indices = np.flatnonzero(
            np.all(
                case_channel_std > self.config.minimum_case_channel_std,
                axis=0,
            )
        )
        if len(self.channel_indices) == 0:
            raise ValueError(
                "MiniRocket train-only variance filter removed every channel"
            )
        values = values[:, self.channel_indices]
        self.transformer = MiniRocket(
            n_kernels=self.config.n_kernels,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        transformed = self.transformer.fit_transform(values)
        validation_transformed = None
        if self.config.tune_on_validation:
            if validation_sequence is None:
                raise ValueError("MiniRocket tuning requires validation sequence")
            validation_values = _as_collection(validation_sequence)[:, self.channel_indices]
            validation_transformed = self.transformer.transform(validation_values)
        self.classifier, self.selected_classification_c = fit_classifier(
            transformed,
            class_target,
            validation_transformed,
            validation_class_target,
            c_values=(
                self.config.classification_c_grid
                if self.config.tune_on_validation
                else (self.config.classification_c,)
            ),
            random_state=self.config.random_state,
            scaler_with_mean=False,
            solver=self.config.classification_solver,
        )
        self.regressor, self.selected_regression_alpha = fit_regressor(
            transformed,
            regression_target,
            validation_transformed,
            validation_regression_target,
            alpha_values=(
                self.config.regression_alpha_grid
                if self.config.tune_on_validation
                else (self.config.regression_alpha,)
            ),
            scaler_with_mean=False,
        )
        return self

    def predict(self, sequence):
        if self.transformer is None or self.classifier is None or self.regressor is None:
            raise RuntimeError("MiniRocket consumer must be fitted before predict")
        values = _as_collection(sequence)[:, self.channel_indices]
        transformed = self.transformer.transform(values)
        return {
            "class_prediction": self.classifier.predict(transformed),
            "class_probability": self.classifier.predict_proba(transformed),
            "classes": classifier_classes(self.classifier),
            "regression_prediction": self.regressor.predict(transformed),
            "transformed_feature_count": int(transformed.shape[1]),
            "input_channel_count": int(len(self.channel_indices)),
        }


@dataclass(frozen=True, slots=True)
class LinearConsumerConfig:
    random_state: int = 17
    classification_c: float = 1.0
    regression_alpha: float = 1.0
    classification_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    regression_alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    tune_on_validation: bool = False


class LinearFrozenConsumer:
    """Fixed pooled-state Logistic/Ridge consumer used beside MiniRocket."""

    def __init__(self, config: LinearConsumerConfig | None = None) -> None:
        self.config = config or LinearConsumerConfig()
        self.classifier = None
        self.regressor = None
        self.selected_classification_c = None
        self.selected_regression_alpha = None

    def fit(
        self,
        pooled,
        class_target,
        regression_target,
        *,
        validation_pooled=None,
        validation_class_target=None,
        validation_regression_target=None,
    ):
        values = np.asarray(pooled, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("linear pooled states must have shape [N,D]")
        validation_values = (
            np.asarray(validation_pooled, dtype=np.float32)
            if self.config.tune_on_validation
            else None
        )
        if self.config.tune_on_validation and validation_values is None:
            raise ValueError("linear tuning requires validation pooled states")
        self.classifier, self.selected_classification_c = fit_classifier(
            values,
            class_target,
            validation_values,
            validation_class_target,
            c_values=(
                self.config.classification_c_grid
                if self.config.tune_on_validation
                else (self.config.classification_c,)
            ),
            random_state=self.config.random_state,
            scaler_with_mean=True,
        )
        self.regressor, self.selected_regression_alpha = fit_regressor(
            values,
            regression_target,
            validation_values,
            validation_regression_target,
            alpha_values=(
                self.config.regression_alpha_grid
                if self.config.tune_on_validation
                else (self.config.regression_alpha,)
            ),
            scaler_with_mean=True,
        )
        return self

    def predict(self, pooled):
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("linear consumer must be fitted before predict")
        values = np.asarray(pooled, dtype=np.float32)
        return {
            "class_prediction": self.classifier.predict(values),
            "class_probability": self.classifier.predict_proba(values),
            "classes": self.classifier.named_steps["logisticregression"].classes_,
            "regression_prediction": self.regressor.predict(values),
        }


@dataclass(frozen=True, slots=True)
class TCNConsumerConfig:
    input_dim: int = 64
    hidden_channels: int = 64
    class_count: int = 5
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.1
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 17
    patience: int = 6
    minimum_delta: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("TCN device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("TCN requested unavailable CUDA device")
        if self.patience <= 0 or self.minimum_delta < 0:
            raise ValueError("TCN early-stopping configuration is invalid")


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, *, kernel_size, dilation):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, values):
        return self.conv(F.pad(values, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values):
        transformed = self.dropout(F.gelu(self.conv(values)))
        return transformed + self.residual(values)


class CausalTCNEmissionModel(nn.Module):
    def __init__(self, config: TCNConsumerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCNConsumerConfig()
        layers = []
        input_channels = self.config.input_dim
        for dilation in self.config.dilations:
            layers.append(
                CausalResidualBlock(
                    input_channels,
                    self.config.hidden_channels,
                    kernel_size=self.config.kernel_size,
                    dilation=dilation,
                    dropout=self.config.dropout,
                )
            )
            input_channels = self.config.hidden_channels
        self.network = nn.Sequential(*layers)
        self.emission = nn.Conv1d(
            self.config.hidden_channels,
            self.config.class_count,
            kernel_size=1,
        )

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[-1] != self.config.input_dim:
            raise ValueError("TCN sequence must have shape [B,T,64]")
        hidden = self.network(sequence.transpose(1, 2))
        return self.emission(hidden).transpose(1, 2)


@dataclass(frozen=True, slots=True)
class TCNTrainingResult:
    model: CausalTCNEmissionModel
    training_rows: tuple[dict[str, object], ...]
    class_weights: torch.Tensor
    best_epoch: int
    stopped_early: bool


def fit_causal_tcn_emission(
    train_sequence,
    train_labels,
    *,
    config: TCNConsumerConfig | None = None,
    validation_sequence=None,
    validation_labels=None,
) -> TCNTrainingResult:
    resolved = config or TCNConsumerConfig()
    values = torch.as_tensor(
        train_sequence, dtype=torch.float32, device=resolved.device
    )
    labels = torch.as_tensor(train_labels, dtype=torch.long, device=resolved.device)
    if values.shape[:2] != labels.shape:
        raise ValueError("TCN train sequence/label shape mismatch")
    validation_values = (
        torch.as_tensor(
            validation_sequence, dtype=torch.float32, device=resolved.device
        )
        if validation_sequence is not None
        else None
    )
    validation_targets = (
        torch.as_tensor(validation_labels, dtype=torch.long, device=resolved.device)
        if validation_labels is not None
        else None
    )
    if (validation_values is None) != (validation_targets is None):
        raise ValueError("TCN validation sequence and labels must be supplied together")
    if validation_values is not None and validation_values.shape[:2] != validation_targets.shape:
        raise ValueError("TCN validation sequence/label shape mismatch")
    counts = torch.bincount(labels.flatten(), minlength=resolved.class_count).float()
    weights = torch.where(
        counts > 0,
        counts.sum() / (resolved.class_count * counts.clamp_min(1)),
        torch.zeros_like(counts),
    )
    rng_devices = [torch.cuda.current_device()] if resolved.device == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(resolved.seed)
        model = CausalTCNEmissionModel(resolved).to(resolved.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved.learning_rate,
            weight_decay=resolved.weight_decay,
        )
        rows = []
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        stale_epochs = 0
        for epoch in range(1, resolved.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = F.cross_entropy(
                logits.reshape(-1, resolved.class_count),
                labels.reshape(-1),
                weight=weights,
            )
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                if validation_values is None:
                    selection_loss = float(loss.detach())
                else:
                    validation_logits = model(validation_values)
                    selection_loss = float(
                        F.cross_entropy(
                            validation_logits.reshape(-1, resolved.class_count),
                            validation_targets.reshape(-1),
                            weight=weights,
                        )
                    )
            improved = selection_loss < best_loss - resolved.minimum_delta
            if improved:
                best_loss = selection_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.detach()),
                    "selection_loss": selection_loss,
                    "selection_role": (
                        "validation" if validation_values is not None else "train"
                    ),
                    "improved": improved,
                    "epochs_without_improvement": stale_epochs,
                    "gradient_norm_before_clip": gradient_norm,
                    "valid_query_count": int(labels.numel()),
                }
            )
            if stale_epochs >= resolved.patience:
                break
        if best_state is None:
            raise RuntimeError("TCN training produced no best state")
        model.load_state_dict(best_state)
        model = model.cpu()
        weights = weights.cpu()
    return TCNTrainingResult(
        model,
        tuple(rows),
        weights,
        best_epoch,
        len(rows) < resolved.epochs,
    )


@dataclass(frozen=True, slots=True)
class DurationViterbiParameters:
    initial_log_probability: torch.Tensor
    transition_log_probability: torch.Tensor
    minimum_duration: torch.Tensor
    maximum_duration: torch.Tensor
    train_sequence_count: int


def fit_duration_viterbi_parameters(
    train_labels,
    *,
    class_count: int = 5,
    laplace: float = 1.0,
) -> DurationViterbiParameters:
    labels = torch.as_tensor(train_labels, dtype=torch.long)
    if labels.ndim != 2 or labels.numel() == 0:
        raise ValueError("Viterbi train labels must have shape [N,T]")
    initial = torch.full((class_count,), laplace, dtype=torch.float64)
    transition = torch.full(
        (class_count, class_count),
        laplace,
        dtype=torch.float64,
    )
    durations = [[] for _ in range(class_count)]
    for row in labels.tolist():
        initial[row[0]] += 1
        start = 0
        for index in range(1, len(row) + 1):
            if index == len(row) or row[index] != row[start]:
                durations[row[start]].append(index - start)
                if index < len(row):
                    transition[row[index - 1], row[index]] += 1
                start = index
    initial /= initial.sum()
    transition /= transition.sum(dim=1, keepdim=True)
    minimum = torch.tensor(
        [max(1, min(values)) if values else 1 for values in durations],
        dtype=torch.long,
    )
    maximum = torch.tensor(
        [max(values) if values else labels.shape[1] for values in durations],
        dtype=torch.long,
    )
    return DurationViterbiParameters(
        initial_log_probability=initial.log(),
        transition_log_probability=transition.log(),
        minimum_duration=minimum,
        maximum_duration=maximum,
        train_sequence_count=labels.shape[0],
    )


def duration_constrained_viterbi_decode(
    emission_logits,
    parameters: DurationViterbiParameters,
) -> torch.Tensor:
    logits = torch.as_tensor(emission_logits, dtype=torch.float64)
    if logits.ndim == 3:
        return _decode_batch(logits, parameters)
    if logits.ndim != 2:
        raise ValueError("Viterbi emissions must have shape [T,C] or [B,T,C]")
    return _decode_one(logits, parameters)


def _decode_batch(logits, parameters):
    """Run the same semi-Markov recurrence with samples vectorized together."""
    log_emission = torch.log_softmax(logits, dim=-1)
    batch_size, time_count, class_count = log_emission.shape
    cumulative = torch.cat(
        (
            torch.zeros(
                batch_size,
                1,
                class_count,
                dtype=log_emission.dtype,
                device=log_emission.device,
            ),
            log_emission.cumsum(dim=1),
        ),
        dim=1,
    )
    initial = parameters.initial_log_probability.to(
        dtype=log_emission.dtype,
        device=log_emission.device,
    )
    transition = parameters.transition_log_probability.to(
        dtype=log_emission.dtype,
        device=log_emission.device,
    )
    minimum_duration = parameters.minimum_duration.tolist()
    maximum_duration = parameters.maximum_duration.tolist()
    dp = torch.full(
        (batch_size, time_count + 1, class_count),
        float("-inf"),
        dtype=log_emission.dtype,
        device=log_emission.device,
    )
    previous_time = torch.full(
        (batch_size, time_count + 1, class_count),
        -1,
        dtype=torch.long,
        device=log_emission.device,
    )
    previous_class = torch.full_like(previous_time, -1)
    for end in range(1, time_count + 1):
        for class_id in range(class_count):
            minimum = int(minimum_duration[class_id])
            maximum = min(int(maximum_duration[class_id]), end)
            for duration in range(minimum, maximum + 1):
                start = end - duration
                score = (
                    cumulative[:, end, class_id]
                    - cumulative[:, start, class_id]
                )
                if start == 0:
                    score = score + initial[class_id]
                    prior_class = torch.full(
                        (batch_size,),
                        -1,
                        dtype=torch.long,
                        device=log_emission.device,
                    )
                else:
                    candidates = dp[:, start] + transition[:, class_id]
                    best_score, prior_class = candidates.max(dim=1)
                    score = score + best_score
                better = score > dp[:, end, class_id]
                dp[:, end, class_id] = torch.where(
                    better,
                    score,
                    dp[:, end, class_id],
                )
                previous_time[:, end, class_id] = torch.where(
                    better,
                    torch.full_like(prior_class, start),
                    previous_time[:, end, class_id],
                )
                previous_class[:, end, class_id] = torch.where(
                    better,
                    prior_class,
                    previous_class[:, end, class_id],
                )
    decoded = torch.empty(
        batch_size,
        time_count,
        dtype=torch.long,
        device=log_emission.device,
    )
    for batch_index in range(batch_size):
        end = time_count
        end_class = int(dp[batch_index, time_count].argmax())
        fallback = False
        while end > 0:
            start = int(previous_time[batch_index, end, end_class])
            if start < 0:
                fallback = True
                break
            decoded[batch_index, start:end] = end_class
            prior_class = int(previous_class[batch_index, end, end_class])
            end = start
            if end > 0:
                end_class = prior_class
        if fallback:
            decoded[batch_index] = logits[batch_index].argmax(dim=-1)
    return decoded


def _decode_one(logits, parameters):
    log_emission = torch.log_softmax(logits, dim=-1)
    time_count, class_count = log_emission.shape
    cumulative = torch.cat(
        (torch.zeros(1, class_count, dtype=log_emission.dtype), log_emission.cumsum(dim=0)),
        dim=0,
    )
    negative_infinity = torch.tensor(float("-inf"), dtype=log_emission.dtype)
    dp = torch.full((time_count + 1, class_count), negative_infinity)
    previous_time = torch.full((time_count + 1, class_count), -1, dtype=torch.long)
    previous_class = torch.full((time_count + 1, class_count), -1, dtype=torch.long)
    for end in range(1, time_count + 1):
        for class_id in range(class_count):
            minimum = int(parameters.minimum_duration[class_id])
            maximum = min(int(parameters.maximum_duration[class_id]), end)
            for duration in range(minimum, maximum + 1):
                start = end - duration
                emission_score = cumulative[end, class_id] - cumulative[start, class_id]
                if start == 0:
                    score = parameters.initial_log_probability[class_id] + emission_score
                    prior_class = -1
                else:
                    candidates = dp[start] + parameters.transition_log_probability[:, class_id]
                    best_score, best_class = candidates.max(dim=0)
                    score = best_score + emission_score
                    prior_class = int(best_class)
                if score > dp[end, class_id]:
                    dp[end, class_id] = score
                    previous_time[end, class_id] = start
                    previous_class[end, class_id] = prior_class
    end_class = int(dp[time_count].argmax())
    decoded = torch.empty(time_count, dtype=torch.long)
    end = time_count
    while end > 0:
        start = int(previous_time[end, end_class])
        if start < 0:
            return logits.argmax(dim=-1).to(torch.long)
        decoded[start:end] = end_class
        prior_class = int(previous_class[end, end_class])
        end = start
        if end > 0:
            end_class = prior_class
    return decoded


def _as_collection(sequence):
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("MiniRocket sequence must have shape [N,T,D]")
    return np.ascontiguousarray(values.transpose(0, 2, 1))
