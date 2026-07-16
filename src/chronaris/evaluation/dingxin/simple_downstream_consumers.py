"""Fixed Ridge/Logistic consumers and task-level aggregation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class SimpleConsumerConfig:
    classification_c: float = 1.0
    regression_alpha: float = 1.0
    random_state: int = 17


@dataclass(slots=True)
class SimpleDownstreamConsumer:
    config: SimpleConsumerConfig
    maneuver_scaler: StandardScaler
    maneuver_regressor: Ridge
    maneuver_classifier: LogisticRegression
    physiology_scaler: StandardScaler
    physiology_regressor: Ridge
    physiology_fields: tuple[str, ...]

    def predict(self, values) -> dict[str, np.ndarray]:
        array = _embedding_array(values)
        maneuver = self.maneuver_scaler.transform(array)
        physiology = self.physiology_scaler.transform(array)
        probability = np.zeros((len(array), 3), dtype=np.float64)
        fitted_probability = self.maneuver_classifier.predict_proba(maneuver)
        probability[:, self.maneuver_classifier.classes_.astype(int)] = fitted_probability
        return {
            "maneuver_score": np.asarray(
                self.maneuver_regressor.predict(maneuver), dtype=np.float64
            ),
            "maneuver_probability": probability,
            "maneuver_class": np.argmax(probability, axis=1).astype(np.int64),
            "physiology_standardized": np.asarray(
                self.physiology_regressor.predict(physiology), dtype=np.float64
            ),
        }


def fit_simple_downstream_consumer(
    *,
    pooled_embedding,
    sample_ids: Sequence[str],
    maneuver_targets: pd.DataFrame,
    physiology_targets: pd.DataFrame,
    config: SimpleConsumerConfig | None = None,
) -> SimpleDownstreamConsumer:
    resolved = config or SimpleConsumerConfig()
    values = _embedding_array(pooled_embedding)
    ordered_ids = tuple(str(value) for value in sample_ids)
    if len(ordered_ids) != len(values) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("consumer representation IDs are invalid")
    position = {sample_id: index for index, sample_id in enumerate(ordered_ids)}
    maneuver = maneuver_targets[
        maneuver_targets["split_role"].astype(str) == "train"
    ].copy()
    if maneuver["context_id"].duplicated().any():
        raise ValueError("maneuver train targets are duplicated")
    maneuver_ids = tuple(maneuver["context_id"].astype(str))
    missing = sorted(set(maneuver_ids) - set(position))
    if missing:
        raise ValueError(f"maneuver representations missing: {missing[:5]}")
    maneuver_values = values[[position[value] for value in maneuver_ids]]
    maneuver_weights = maneuver["sample_weight"].to_numpy(dtype=np.float64)
    maneuver_scaler = StandardScaler().fit(
        maneuver_values, sample_weight=maneuver_weights
    )
    maneuver_scaled = maneuver_scaler.transform(maneuver_values)
    maneuver_regressor = Ridge(alpha=resolved.regression_alpha).fit(
        maneuver_scaled,
        maneuver["future_maneuver_score"].to_numpy(dtype=np.float64),
        sample_weight=maneuver_weights,
    )
    maneuver_classifier = LogisticRegression(
        C=resolved.classification_c,
        class_weight="balanced",
        max_iter=5_000,
        random_state=resolved.random_state,
    ).fit(
        maneuver_scaled,
        maneuver["future_maneuver_class"].to_numpy(dtype=np.int64),
        sample_weight=maneuver_weights,
    )
    if set(maneuver_classifier.classes_.astype(int)) != {0, 1, 2}:
        raise ValueError("maneuver training fold does not contain all three classes")

    physiology = physiology_targets[
        (physiology_targets["split_role"].astype(str) == "train")
        & physiology_targets["selected"].astype(bool)
    ].copy()
    fields = tuple(sorted(physiology["field_name"].astype(str).unique()))
    target_matrix = physiology.pivot(
        index="context_id", columns="field_name", values="future_standardized"
    ).reindex(columns=fields)
    valid_rows = np.isfinite(target_matrix.to_numpy(dtype=np.float64)).all(axis=1)
    target_matrix = target_matrix.loc[valid_rows]
    physiology_ids = tuple(target_matrix.index.astype(str))
    missing = sorted(set(physiology_ids) - set(position))
    if missing:
        raise ValueError(f"physiology representations missing: {missing[:5]}")
    if len(physiology_ids) < 4 or len(fields) < 2:
        raise ValueError("physiology train target matrix is too small")
    physiology_values = values[[position[value] for value in physiology_ids]]
    physiology_scaler = StandardScaler().fit(physiology_values)
    physiology_regressor = Ridge(alpha=resolved.regression_alpha).fit(
        physiology_scaler.transform(physiology_values),
        target_matrix.to_numpy(dtype=np.float64),
    )
    return SimpleDownstreamConsumer(
        config=resolved,
        maneuver_scaler=maneuver_scaler,
        maneuver_regressor=maneuver_regressor,
        maneuver_classifier=maneuver_classifier,
        physiology_scaler=physiology_scaler,
        physiology_regressor=physiology_regressor,
        physiology_fields=fields,
    )


def maneuver_metric_summary(
    targets: pd.DataFrame,
    *,
    sample_ids: Sequence[str],
    score_prediction,
    class_probability,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    frame = _prediction_frame(
        targets,
        sample_ids=sample_ids,
        score_prediction=score_prediction,
        class_probability=class_probability,
    )
    rows = []
    for vehicle_id, group in frame.groupby("vehicle_context_id", sort=True):
        if group["future_maneuver_score"].nunique() != 1:
            raise ValueError(f"inconsistent maneuver target for {vehicle_id}")
        probability = np.mean(
            np.stack(group["class_probability"].to_list()), axis=0
        )
        rows.append(
            {
                "vehicle_context_id": vehicle_id,
                "sortie_id": str(group["sortie_id"].iloc[0]),
                "future_maneuver_score": float(group["future_maneuver_score"].iloc[0]),
                "current_maneuver_score": float(group["current_maneuver_score"].iloc[0]),
                "future_maneuver_class": int(group["future_maneuver_class"].iloc[0]),
                "score_prediction": float(group["score_prediction"].mean()),
                "class_probability": probability,
                "class_prediction": int(np.argmax(probability)),
                "view_count": len(group),
                "train_target_iqr": float(group["train_target_iqr"].iloc[0]),
            }
        )
    aggregated = pd.DataFrame(rows)
    truth = aggregated["future_maneuver_score"].to_numpy(dtype=np.float64)
    prediction = aggregated["score_prediction"].to_numpy(dtype=np.float64)
    persistence = aggregated["current_maneuver_score"].to_numpy(dtype=np.float64)
    target_iqr = float(aggregated["train_target_iqr"].iloc[0])
    denominator = float(np.sum((truth - persistence) ** 2))
    correlation = spearmanr(truth, prediction).statistic
    summary = {
        "independent_vehicle_context_count": len(aggregated),
        "spearman": float(correlation) if np.isfinite(correlation) else None,
        "normalized_mae": float(np.mean(np.abs(truth - prediction)) / target_iqr)
        if target_iqr > 0
        else None,
        "skill_vs_current_maneuver": (
            None
            if denominator <= 0
            else 1.0 - float(np.sum((truth - prediction) ** 2)) / denominator
        ),
        "macro_f1": float(
            f1_score(
                aggregated["future_maneuver_class"],
                aggregated["class_prediction"],
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                aggregated["future_maneuver_class"],
                aggregated["class_prediction"],
            )
        ),
    }
    return summary, aggregated


def physiology_metric_summary(
    targets: pd.DataFrame,
    *,
    sample_ids: Sequence[str],
    standardized_prediction,
    fields: Sequence[str],
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    ids = tuple(str(value) for value in sample_ids)
    prediction = np.asarray(standardized_prediction, dtype=np.float64)
    fields = tuple(str(value) for value in fields)
    if prediction.shape != (len(ids), len(fields)):
        raise ValueError("physiology prediction shape mismatch")
    selected = targets[targets["selected"].astype(bool)].copy()
    future = selected.pivot(
        index="context_id", columns="field_name", values="future_standardized"
    ).reindex(index=ids, columns=fields)
    current = selected.pivot(
        index="context_id", columns="field_name", values="current_standardized"
    ).reindex(index=ids, columns=fields)
    category = (
        selected.drop_duplicates("field_name")
        .set_index("field_name")["semantic_category"]
        .astype(str)
        .to_dict()
    )
    rows = []
    total_model_error = 0.0
    total_persistence_error = 0.0
    for index, field_name in enumerate(fields):
        truth = future[field_name].to_numpy(dtype=np.float64)
        baseline = current[field_name].to_numpy(dtype=np.float64)
        valid = np.isfinite(truth) & np.isfinite(baseline) & np.isfinite(prediction[:, index])
        if not valid.any():
            rows.append(
                {
                    "field_name": field_name,
                    "semantic_category": category.get(field_name),
                    "support": 0,
                    "rmse": None,
                    "mae": None,
                    "skill_vs_persistence": None,
                }
            )
            continue
        residual = truth[valid] - prediction[valid, index]
        persistence_residual = truth[valid] - baseline[valid]
        model_error = float(np.sum(residual**2))
        persistence_error = float(np.sum(persistence_residual**2))
        total_model_error += model_error
        total_persistence_error += persistence_error
        rows.append(
            {
                "field_name": field_name,
                "semantic_category": category.get(field_name),
                "support": int(valid.sum()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "skill_vs_persistence": (
                    None
                    if persistence_error <= 0
                    else 1.0 - model_error / persistence_error
                ),
            }
        )
    per_field = pd.DataFrame(rows)
    available = per_field[per_field["support"] > 0]
    skills = available["skill_vs_persistence"].dropna().to_numpy(dtype=float)
    summary = {
        "view_context_count": len(ids),
        "field_count": len(available),
        "standardized_rmse_macro": float(available["rmse"].mean()),
        "standardized_mae_macro": float(available["mae"].mean()),
        "skill_vs_persistence": (
            None
            if total_persistence_error <= 0
            else 1.0 - total_model_error / total_persistence_error
        ),
        "positive_skill_field_ratio": (
            None if not len(skills) else float(np.mean(skills > 0))
        ),
        "eeg_rmse_macro": _category_mean(available, "eeg", "rmse"),
        "spo2_rmse_macro": _category_mean(available, "spo2", "rmse"),
    }
    return summary, per_field


def _prediction_frame(
    targets, *, sample_ids, score_prediction, class_probability
):
    ids = tuple(str(value) for value in sample_ids)
    score = np.asarray(score_prediction, dtype=np.float64)
    probability = np.asarray(class_probability, dtype=np.float64)
    if score.shape != (len(ids),) or probability.shape != (len(ids), 3):
        raise ValueError("maneuver prediction shape mismatch")
    target = targets.set_index("context_id").reindex(ids)
    if target["vehicle_context_id"].isna().any():
        raise ValueError("maneuver predictions contain unknown context IDs")
    target = target.reset_index().rename(columns={"index": "context_id"})
    target["score_prediction"] = score
    target["class_probability"] = list(probability)
    return target


def _embedding_array(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 64:
        raise ValueError("simple downstream consumer expects pooled [N,64] values")
    if not np.isfinite(array).all():
        raise ValueError("simple downstream embedding contains non-finite values")
    return array


def _category_mean(frame, category, column):
    values = frame[frame["semantic_category"] == category][column].dropna()
    return None if values.empty else float(values.mean())
