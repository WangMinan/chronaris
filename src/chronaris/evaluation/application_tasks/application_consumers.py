"""Method-invariant MiniRocket, causal TCN, and duration-constrained decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class MiniRocketConsumerConfig:
    n_kernels: int = 10_000
    random_state: int = 17
    n_jobs: int = 1
    classification_c: float = 1.0
    regression_alpha: float = 1.0
    minimum_case_channel_std: float = 1e-7


class MiniRocketFrozenConsumer:
    """One train-role MiniRocket transform shared by classification/regression."""

    def __init__(self, config: MiniRocketConsumerConfig | None = None) -> None:
        self.config = config or MiniRocketConsumerConfig()
        self.transformer = None
        self.classifier = None
        self.regressor = None
        self.channel_indices = None

    def fit(self, sequence, class_target, regression_target):
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
        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.config.classification_c,
                max_iter=500,
                random_state=self.config.random_state,
            ),
        ).fit(transformed, np.asarray(class_target, dtype=np.int64))
        self.regressor = make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.config.regression_alpha),
        ).fit(transformed, np.asarray(regression_target, dtype=np.float64))
        return self

    def predict(self, sequence):
        if self.transformer is None or self.classifier is None or self.regressor is None:
            raise RuntimeError("MiniRocket consumer must be fitted before predict")
        values = _as_collection(sequence)[:, self.channel_indices]
        if bool(
            np.any(
                values.std(axis=-1)
                <= self.config.minimum_case_channel_std
            )
        ):
            raise ValueError(
                "MiniRocket evaluation data violates train-only channel variance contract"
            )
        transformed = self.transformer.transform(values)
        return {
            "class_prediction": self.classifier.predict(transformed),
            "class_probability": self.classifier.predict_proba(transformed),
            "classes": self.classifier.named_steps["logisticregression"].classes_,
            "regression_prediction": self.regressor.predict(transformed),
            "transformed_feature_count": int(transformed.shape[1]),
            "input_channel_count": int(len(self.channel_indices)),
        }


@dataclass(frozen=True, slots=True)
class LinearConsumerConfig:
    random_state: int = 17
    classification_c: float = 1.0
    regression_alpha: float = 1.0


class LinearFrozenConsumer:
    """Fixed pooled-state Logistic/Ridge consumer used beside MiniRocket."""

    def __init__(self, config: LinearConsumerConfig | None = None) -> None:
        self.config = config or LinearConsumerConfig()
        self.classifier = None
        self.regressor = None

    def fit(self, pooled, class_target, regression_target):
        values = np.asarray(pooled, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("linear pooled states must have shape [N,D]")
        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.config.classification_c,
                max_iter=500,
                random_state=self.config.random_state,
            ),
        ).fit(values, np.asarray(class_target, dtype=np.int64))
        self.regressor = make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.config.regression_alpha),
        ).fit(values, np.asarray(regression_target, dtype=np.float64))
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


class CausalTCNEmissionModel(nn.Module):
    def __init__(self, config: TCNConsumerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCNConsumerConfig()
        layers = []
        input_channels = self.config.input_dim
        for dilation in self.config.dilations:
            layers.extend(
                (
                    CausalConv1d(
                        input_channels,
                        self.config.hidden_channels,
                        kernel_size=self.config.kernel_size,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
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


def fit_causal_tcn_emission(
    train_sequence,
    train_labels,
    *,
    config: TCNConsumerConfig | None = None,
) -> TCNTrainingResult:
    resolved = config or TCNConsumerConfig()
    values = torch.as_tensor(train_sequence, dtype=torch.float32)
    labels = torch.as_tensor(train_labels, dtype=torch.long)
    if values.shape[:2] != labels.shape:
        raise ValueError("TCN train sequence/label shape mismatch")
    counts = torch.bincount(labels.flatten(), minlength=resolved.class_count).float()
    weights = torch.where(
        counts > 0,
        counts.sum() / (resolved.class_count * counts.clamp_min(1)),
        torch.zeros_like(counts),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(resolved.seed)
        model = CausalTCNEmissionModel(resolved)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved.learning_rate,
            weight_decay=resolved.weight_decay,
        )
        rows = []
        model.train()
        for epoch in range(1, resolved.epochs + 1):
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
            rows.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "gradient_norm_before_clip": gradient_norm,
                    "valid_query_count": int(labels.numel()),
                }
            )
    return TCNTrainingResult(model, tuple(rows), weights)


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
        return torch.stack(
            tuple(
                _decode_one(logits[index], parameters)
                for index in range(logits.shape[0])
            )
        )
    if logits.ndim != 2:
        raise ValueError("Viterbi emissions must have shape [T,C] or [B,T,C]")
    return _decode_one(logits, parameters)


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
