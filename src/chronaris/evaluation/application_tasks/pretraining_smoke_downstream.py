"""Guarded smoke-only workload targets and fixed linear downstream consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from chronaris.representation import FusionStreamBatch, FoldLineage
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def load_guarded_pretraining_smoke_targets(
    *,
    data_manifest_rows: Sequence[Mapping[str, object]],
    fold: FoldLineage,
    completed_pretraining_checkpoints: Sequence[str | Path],
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Open workload truth only after all five representation checkpoints exist."""

    _require_completed_pretraining(completed_pretraining_checkpoints)
    rows = []
    oracle_files = []
    role_by_sample = {
        sample_id: role
        for role, values in (
            ("train", fold.train_sample_ids),
            ("validation", fold.validation_sample_ids),
            ("held_out", fold.held_out_sample_ids),
        )
        for sample_id in values
    }
    for item in data_manifest_rows:
        observed_path = Path(str(item["observed_path"]))
        oracle_path = observed_path.with_name("ground_truth.npz")
        if not oracle_path.is_file():
            raise FileNotFoundError(f"simulation workload truth missing: {oracle_path}")
        with np.load(oracle_path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if not {"true_time_s", "workload"}.issubset(keys):
                raise ValueError("simulation truth lacks workload/time fields")
            times = np.asarray(archive["true_time_s"], dtype=np.float64)
            workload = np.asarray(archive["workload"], dtype=np.float64)
        future = (times >= 30.0) & (times < 35.0)
        if not future.any():
            raise ValueError("simulation truth has no 30-35 second future target")
        sample_id = str(item["sample_id"])
        rows.append(
            {
                "sample_id": sample_id,
                "role": role_by_sample[sample_id],
                "future_workload_mean": float(workload[future].mean()),
            }
        )
        oracle_files.append(
            {
                "sample_id": sample_id,
                "oracle_path": str(oracle_path),
                "oracle_sha256": sha256_file(oracle_path),
                "fields_opened": ["true_time_s", "workload"],
            }
        )
    frame = pd.DataFrame(rows)
    train_values = frame.loc[
        frame["role"] == "train",
        "future_workload_mean",
    ].to_numpy()
    lower, upper = np.quantile(train_values, (1 / 3, 2 / 3))
    frame["workload_class"] = np.where(
        frame["future_workload_mean"] <= lower,
        0,
        np.where(frame["future_workload_mean"] <= upper, 1, 2),
    ).astype(int)
    manifest = {
        "format": "chronaris.pretraining_smoke_targets.v1",
        "oracle_opened_after_checkpoint_count": len(
            completed_pretraining_checkpoints
        ),
        "target_window_s": [30.0, 35.0],
        "classification_thresholds_train_only": [float(lower), float(upper)],
        "oracle_files": oracle_files,
        "smoke_only": True,
    }
    return frame, manifest


def run_fixed_linear_smoke_consumers(
    *,
    outputs: Mapping[str, Mapping[str, FusionStreamBatch]],
    targets: pd.DataFrame,
    fold: FoldLineage,
    seed: int = 17,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run method-invariant Logistic/Ridge consumers on frozen pooled states."""

    target_index = targets.set_index("sample_id")
    metric_rows = []
    prediction_rows = []
    for method_name, roles in outputs.items():
        train_output = roles["train"]
        x_train = train_output.pooled_embedding.detach().cpu().numpy()
        y_class_train = target_index.loc[
            list(train_output.sample_ids),
            "workload_class",
        ].to_numpy(dtype=int)
        y_reg_train = target_index.loc[
            list(train_output.sample_ids),
            "future_workload_mean",
        ].to_numpy(dtype=float)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=500,
                random_state=seed,
            ),
        )
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        classifier.fit(x_train, y_class_train)
        regressor.fit(x_train, y_reg_train)
        classes = classifier.named_steps["logisticregression"].classes_
        for role in ("validation", "held_out"):
            output = roles[role]
            x = output.pooled_embedding.detach().cpu().numpy()
            y_class = target_index.loc[
                list(output.sample_ids),
                "workload_class",
            ].to_numpy(dtype=int)
            y_reg = target_index.loc[
                list(output.sample_ids),
                "future_workload_mean",
            ].to_numpy(dtype=float)
            class_pred = classifier.predict(x)
            class_prob = classifier.predict_proba(x)
            reg_pred = regressor.predict(x)
            class_metrics = {
                "macro_f1": f1_score(
                    y_class,
                    class_pred,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(y_class, class_pred),
                "macro_auprc": _macro_auprc(y_class, class_prob, classes),
            }
            correlation = spearmanr(y_reg, reg_pred).statistic
            regression_metrics = {
                "mae": mean_absolute_error(y_reg, reg_pred),
                "rmse": mean_squared_error(y_reg, reg_pred) ** 0.5,
                "spearman": float(correlation) if np.isfinite(correlation) else None,
            }
            for metric_name, value in class_metrics.items():
                metric_rows.append(
                    _metric_row(
                        method_name,
                        role,
                        "simulated_future_workload_classification",
                        metric_name,
                        value,
                    )
                )
            for metric_name, value in regression_metrics.items():
                metric_rows.append(
                    _metric_row(
                        method_name,
                        role,
                        "simulated_future_workload_regression",
                        metric_name,
                        value,
                    )
                )
            for index, sample_id in enumerate(output.sample_ids):
                prediction_rows.append(
                    {
                        "method_name": method_name,
                        "role": role,
                        "sample_id": sample_id,
                        "workload_class_true": int(y_class[index]),
                        "workload_class_pred": int(class_pred[index]),
                        "future_workload_true": float(y_reg[index]),
                        "future_workload_pred": float(reg_pred[index]),
                        "smoke_only": True,
                    }
                )
    return metric_rows, prediction_rows


def _require_completed_pretraining(paths) -> None:
    resolved = tuple(Path(path) for path in paths)
    if len(resolved) != 5:
        raise ValueError("five completed trainable checkpoints are required")
    methods = set()
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"pretraining checkpoint missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != "chronaris.common_pretraining_checkpoint.v1"
            or payload.get("training_status") != "completed"
            or bool(payload.get("label_used_for_encoder_training"))
        ):
            raise ValueError(f"checkpoint is not eligible for target opening: {path}")
        methods.add(str(payload["method_name"]))
    if len(methods) != 5:
        raise ValueError("completed checkpoint method coverage is incomplete")


def _macro_auprc(y_true, probabilities, classes):
    binary = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binary = np.column_stack((1 - binary[:, 0], binary[:, 0]))
    values = []
    for index, _class_id in enumerate(classes):
        if binary[:, index].sum() == 0:
            continue
        values.append(average_precision_score(binary[:, index], probabilities[:, index]))
    return float(np.mean(values)) if values else None


def _metric_row(method_name, role, task_name, metric_name, value):
    return {
        "method_name": method_name,
        "role": role,
        "task_name": task_name,
        "metric_name": metric_name,
        "value": float(value) if value is not None and np.isfinite(value) else None,
        "status": "available" if value is not None and np.isfinite(value) else "unavailable",
        "reason": None if value is not None and np.isfinite(value) else "metric_not_defined",
        "smoke_only": True,
    }
