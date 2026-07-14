"""Frozen-expert task gates used only after the learnability gate passes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation.application_tasks.core_feasibility_data import fold_task_data
from chronaris.evaluation.application_tasks.core_feasibility_features import summary_features
from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    PRIMARY_FOLDS,
    SAFE_FUSION_GATES,
    gate_passed,
)


GATE_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def run_frozen_safe_fusion(
    *,
    plans,
    cache,
    target_frame,
    maneuver_scores,
    representation_root,
    random_state,
):
    fold_rows = []
    for fold_id in PRIMARY_FOLDS:
        plan = plans[fold_id]
        task_data = fold_task_data(
            fold_id=fold_id,
            train_ids=tuple(str(value) for value in plan["train_sample_ids"]),
            validation_ids=tuple(str(value) for value in plan["validation_sample_ids"]),
            target_frame=target_frame,
            maneuver_scores=maneuver_scores[fold_id],
        )
        for task, modality in (
            ("maneuver", "vehicle"),
            ("response", "physiology"),
            ("high_response", "dual"),
        ):
            data = task_data[task]
            direct_train = summary_features(
                cache,
                sample_ids=data["train_ids"],
                modality=modality,
                history_s=30.0,
            )
            direct_validation = summary_features(
                cache,
                sample_ids=data["validation_ids"],
                modality=modality,
                history_s=30.0,
            )
            continuous_train = _load_pooled(
                representation_root,
                fold_id=fold_id,
                role="train",
                sample_ids=data["train_ids"],
            )
            continuous_validation = _load_pooled(
                representation_root,
                fold_id=fold_id,
                role="validation",
                sample_ids=data["validation_ids"],
            )
            fold_rows.append(
                _evaluate_task_gate(
                    fold_id=fold_id,
                    task=task,
                    direct_train=direct_train,
                    direct_validation=direct_validation,
                    continuous_train=continuous_train,
                    continuous_validation=continuous_validation,
                    targets=data["targets"],
                    random_state=random_state,
                )
            )
    results = []
    for task, specification in SAFE_FUSION_GATES.items():
        task_rows = [row for row in fold_rows if row["task"] == task]
        fused = float(np.mean([row["fused_metric"] for row in task_rows]))
        direct = float(np.mean([row["direct_metric"] for row in task_rows]))
        if specification["direction"] == "higher":
            no_harm = fused >= direct - 0.01
            fold_pass_count = sum(
                row["fused_metric"] >= row["direct_metric"] - 0.01
                for row in task_rows
            )
        else:
            no_harm = fused <= direct + 0.01
            fold_pass_count = sum(
                row["fused_metric"] <= row["direct_metric"] + 0.01
                for row in task_rows
            )
        passed = (
            gate_passed(fused, specification)
            and no_harm
            and fold_pass_count >= 2
        )
        results.append(
            {
                "task": task,
                "metric": specification["metric"],
                "value": fused,
                "threshold": specification["threshold"],
                "best_direct_value": direct,
                "no_harm_passed": no_harm,
                "fold_no_harm_pass_count": fold_pass_count,
                "gate_passed": passed,
                "status": "completed",
                "outer_test_accessed": False,
            }
        )
    return results, fold_rows


def _evaluate_task_gate(
    *, fold_id, task, direct_train, direct_validation, continuous_train,
    continuous_validation, targets, random_state
):
    if task == "maneuver":
        direct_model = _classifier(direct_train, random_state).fit(
            direct_train, targets.train_maneuver
        )
        continuous_model = _classifier(continuous_train, random_state).fit(
            continuous_train, targets.train_maneuver
        )
        direct_prediction = direct_model.predict_proba(direct_validation)
        continuous_prediction = continuous_model.predict_proba(continuous_validation)

        def metric(values):
            return f1_score(
                targets.validation_maneuver,
                np.argmax(values, axis=1),
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            )

        direction = "higher"
    elif task == "response":
        direct_model = _regressor(direct_train).fit(
            direct_train, targets.train_response
        )
        continuous_model = _regressor(continuous_train).fit(
            continuous_train, targets.train_response
        )
        direct_prediction = direct_model.predict(direct_validation)
        continuous_prediction = continuous_model.predict(continuous_validation)

        def metric(values):
            return mean_squared_error(targets.validation_response, values) ** 0.5

        direction = "lower"
    else:
        direct_model = _classifier(
            direct_train, random_state, binary=True
        ).fit(direct_train, targets.train_high_response)
        continuous_model = _classifier(
            continuous_train, random_state, binary=True
        ).fit(continuous_train, targets.train_high_response)
        direct_prediction = direct_model.predict_proba(direct_validation)[:, 1]
        continuous_prediction = continuous_model.predict_proba(
            continuous_validation
        )[:, 1]

        def metric(values):
            return average_precision_score(targets.validation_high_response, values)

        direction = "higher"
    candidates = []
    for gate in GATE_VALUES:
        prediction = (1.0 - gate) * direct_prediction + gate * continuous_prediction
        candidates.append((gate, float(metric(prediction))))
    selected_gate, fused_metric = (
        max(candidates, key=lambda item: item[1])
        if direction == "higher"
        else min(candidates, key=lambda item: item[1])
    )
    return {
        "task": task,
        "fold_id": fold_id,
        "initial_gate": 0.0,
        "selected_gate": selected_gate,
        "direct_metric": float(metric(direct_prediction)),
        "continuous_metric": float(metric(continuous_prediction)),
        "fused_metric": fused_metric,
        "experts_frozen": True,
        "fit_role": "inner_train",
        "evaluation_role": "inner_validation",
        "outer_test_accessed": False,
        "status": "completed",
    }


def _load_pooled(root, *, fold_id, role, sample_ids):
    path = (
        Path(root)
        / "representations"
        / "seed_17"
        / fold_id
        / "chronaris"
        / fold_id
        / role
        / "fusion_stream.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        identifiers = tuple(str(value) for value in archive["sample_ids"])
        values = archive["pooled_embedding"].astype(np.float32)
    lookup = {sample_id: index for index, sample_id in enumerate(identifiers)}
    missing = sorted(set(sample_ids) - set(lookup))
    if missing:
        raise ValueError(f"frozen Chronaris representation lacks samples: {missing[:3]}")
    return np.asarray([values[lookup[sample_id]] for sample_id in sample_ids])


def _classifier(matrix, random_state, binary=False):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=_k(matrix, 128)),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear" if binary else "lbfgs",
            random_state=random_state,
        ),
    )


def _regressor(matrix):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(f_regression, k=_k(matrix, 128)),
        Ridge(alpha=10.0),
    )


def _k(matrix, maximum):
    return max(1, min(int(maximum), matrix.shape[1], max(len(matrix) - 1, 1)))
