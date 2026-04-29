"""Metrics and plots for Stage I public-dataset baselines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score, recall_score


def evaluate_classification_predictions(
    predictions: pd.DataFrame,
    *,
    label_order: Sequence[int | float] | None = None,
) -> dict[str, object]:
    """Evaluate one classification prediction table."""

    if predictions.empty:
        raise ValueError("classification predictions are empty.")
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    labels = list(label_order or sorted(set(y_true) | set(y_pred)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "sample_count": int(len(predictions)),
        "fold_count": int(predictions["split_group"].nunique()),
        "label_order": list(labels),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": {
            str(label): float(recall)
            for label, recall in zip(labels, recalls, strict=True)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_regression_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    """Evaluate one regression prediction table."""

    if predictions.empty:
        raise ValueError("regression predictions are empty.")
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    spearman = spearmanr(y_true, y_pred)
    return {
        "sample_count": int(len(predictions)),
        "fold_count": int(predictions["split_group"].nunique()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearman.statistic) if spearman.statistic == spearman.statistic else float("nan"),
    }


def save_confusion_matrix_plot(
    metrics: dict[str, object],
    *,
    path: str | Path,
    title: str,
) -> str:
    """Render one confusion-matrix PNG."""

    labels = [str(label) for label in metrics["label_order"]]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, int(matrix[row_index, column_index]), ha="center", va="center")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def save_regression_plot(
    predictions: pd.DataFrame,
    *,
    path: str | Path,
    title: str,
) -> str:
    """Render one regression diagnostics PNG."""

    fig, axis = plt.subplots(figsize=(5, 4))
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    axis.scatter(y_true, y_pred, alpha=0.8)
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("True")
    axis.set_ylabel("Predicted")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)
