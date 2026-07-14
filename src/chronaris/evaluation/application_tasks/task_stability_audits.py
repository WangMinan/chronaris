"""Target-scale, metric-ceiling, and leakage audits for Dingxin development."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from chronaris.evaluation.application_tasks.task_stability_contracts import (
    maneuver_metric_ceiling,
    safe_spearman,
)


def metric_ceiling_rows(
    *, manifest: Mapping[str, object], targets: pd.DataFrame,
    maneuver_scores: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    rows = []
    for plan in manifest["folds"]:
        split_id = str(plan["fold_id"])
        maneuver = _target_subset(
            targets, split_id, "maneuver_intensity_classification", "validation"
        )
        response = _target_subset(
            targets, split_id, "physiology_response_prediction", "validation"
        )
        train_maneuver = _target_subset(
            targets, split_id, "maneuver_intensity_classification", "train"
        )
        ceiling = maneuver_metric_ceiling(maneuver["class_target"].astype(int))
        majority = int(
            np.bincount(train_maneuver["class_target"].astype(int), minlength=3).argmax()
        )
        majority_prediction = np.full(len(maneuver), majority, dtype=int)
        majority_score = _fixed_macro_f1(
            maneuver["class_target"].astype(int), majority_prediction
        )
        score_values = np.asarray(
            [maneuver_scores[split_id][str(value)] for value in maneuver["context_id"]]
        )
        rows.append(
            {
                "split_id": split_id,
                "outer_pool_id": plan["outer_pool_id"],
                "split_kind": plan["split_kind"],
                "included_in_main_ranking": bool(plan["main_selection"]),
                "validation_support_hash": plan["validation_support_hash"],
                "validation_classes": ",".join(
                    str(value) for value in ceiling["validation_classes"]
                ),
                "maneuver_low_count": ceiling["class_counts"][0],
                "maneuver_medium_count": ceiling["class_counts"][1],
                "maneuver_high_count": ceiling["class_counts"][2],
                "fixed_macro_f1_ceiling": ceiling["fixed_macro_f1_ceiling"],
                "support_aware_macro_f1_ceiling": ceiling[
                    "support_aware_macro_f1_ceiling"
                ],
                "majority_macro_f1": majority_score,
                "maneuver_score_min": float(np.min(score_values)),
                "maneuver_score_max": float(np.max(score_values)),
                "response_count": len(response),
                "response_std": float(np.std(response["continuous_target"], ddof=0)),
                "response_iqr": float(
                    np.quantile(response["continuous_target"], 0.75)
                    - np.quantile(response["continuous_target"], 0.25)
                ),
                "high_response_prevalence": float(np.mean(response["binary_target"])),
                "random_auprc_baseline": float(np.mean(response["binary_target"])),
                "outer_test_opened": False,
            }
        )
    return pd.DataFrame(rows)


def target_stability_rows(
    *, manifest: Mapping[str, object], targets: pd.DataFrame,
    thresholds: pd.DataFrame, maneuver_scores: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = []
    threshold_rows = []
    rng = np.random.default_rng(17)
    for plan in manifest["folds"]:
        split_id = str(plan["fold_id"])
        train_m = _target_subset(
            targets, split_id, "maneuver_intensity_classification", "train"
        )
        validation_m = _target_subset(
            targets, split_id, "maneuver_intensity_classification", "validation"
        )
        train_r = _target_subset(
            targets, split_id, "physiology_response_prediction", "train"
        )
        validation_r = _target_subset(
            targets, split_id, "physiology_response_prediction", "validation"
        )
        bounds = thresholds[
            (thresholds["fold_id"] == split_id)
            & (thresholds["parameter_type"] == "class_bounds")
        ].iloc[0]
        train_scores = np.asarray(
            [maneuver_scores[split_id][str(value)] for value in train_m["context_id"]]
        )
        validation_scores = np.asarray(
            [
                maneuver_scores[split_id][str(value)]
                for value in validation_m["context_id"]
            ]
        )
        nominal = np.digitize(
            validation_scores, (float(bounds.lower_bound), float(bounds.upper_bound))
        )
        flip_rates = []
        lower_values = []
        upper_values = []
        for _ in range(500):
            sampled = rng.choice(train_scores, size=len(train_scores), replace=True)
            lower, upper = np.quantile(sampled, (1 / 3, 2 / 3))
            lower_values.append(lower)
            upper_values.append(upper)
            flip_rates.append(float(np.mean(np.digitize(validation_scores, (lower, upper)) != nominal)))
        threshold_rows.append(
            {
                "split_id": split_id,
                "outer_pool_id": plan["outer_pool_id"],
                "split_kind": plan["split_kind"],
                "lower_bound": float(bounds.lower_bound),
                "upper_bound": float(bounds.upper_bound),
                "bootstrap_lower_iqr": float(
                    np.quantile(lower_values, 0.75) - np.quantile(lower_values, 0.25)
                ),
                "bootstrap_upper_iqr": float(
                    np.quantile(upper_values, 0.75) - np.quantile(upper_values, 0.25)
                ),
                "bootstrap_mean_label_flip_rate": float(np.mean(flip_rates)),
                "bootstrap_p95_label_flip_rate": float(np.quantile(flip_rates, 0.95)),
                "fit_role": "inner_train",
                "outer_test_opened": False,
            }
        )
        target_rows.extend(
            (
                {
                    "split_id": split_id,
                    "task": "maneuver",
                    "statistic": "score_train_validation_rank_association",
                    "value": safe_spearman(
                        np.arange(len(validation_scores)), validation_scores
                    ),
                    "detail": "validation temporal order versus maneuver score",
                },
                {
                    "split_id": split_id,
                    "task": "response",
                    "statistic": "validation_to_train_mean_ratio",
                    "value": float(
                        np.mean(validation_r["continuous_target"])
                        / max(float(np.mean(train_r["continuous_target"])), 1e-12)
                    ),
                    "detail": "same split-specific response scale",
                },
                {
                    "split_id": split_id,
                    "task": "response",
                    "statistic": "validation_to_train_std_ratio",
                    "value": float(
                        np.std(validation_r["continuous_target"], ddof=0)
                        / max(float(np.std(train_r["continuous_target"], ddof=0)), 1e-12)
                    ),
                    "detail": "same split-specific response scale",
                },
                {
                    "split_id": split_id,
                    "task": "high_response",
                    "statistic": "validation_prevalence",
                    "value": float(np.mean(validation_r["binary_target"])),
                    "detail": "fold-local train q75 definition",
                },
            )
        )
    return pd.DataFrame(target_rows), pd.DataFrame(threshold_rows)


def response_field_stability_rows(
    *, manifest: Mapping[str, object], targets: pd.DataFrame,
    thresholds: pd.DataFrame, field_delta_index: Mapping[tuple[str, str], float],
) -> pd.DataFrame:
    selected_by_split = {}
    for plan in manifest["folds"]:
        split_id = str(plan["fold_id"])
        selected = thresholds[
            (thresholds["fold_id"] == split_id)
            & (thresholds["task_slug"] == "physiology_response_prediction")
            & (thresholds["parameter_type"] == "response_field_scale")
        ]
        selected_by_split[split_id] = set(selected["parameter_name"].astype(str))
    rows = []
    main_sets = [
        selected_by_split[str(plan["fold_id"])]
        for plan in manifest["folds"]
        if plan["main_selection"]
    ]
    for plan in manifest["folds"]:
        split_id = str(plan["fold_id"])
        train = _target_subset(
            targets, split_id, "physiology_response_prediction", "train"
        )
        validation = _target_subset(
            targets, split_id, "physiology_response_prediction", "validation"
        )
        selected = thresholds[
            (thresholds["fold_id"] == split_id)
            & (thresholds["task_slug"] == "physiology_response_prediction")
            & (thresholds["parameter_type"] == "response_field_scale")
        ]
        own = selected_by_split[split_id]
        jaccards = [
            len(own & other) / max(len(own | other), 1)
            for other in main_sets
            if other is not own
        ]
        for field_row in selected.itertuples(index=False):
            field = str(field_row.parameter_name)
            train_values = np.asarray(
                [
                    field_delta_index[(str(value), field)]
                    for value in train["context_id"]
                    if (str(value), field) in field_delta_index
                ],
                dtype=np.float64,
            )
            validation_values = np.asarray(
                [
                    field_delta_index[(str(value), field)]
                    for value in validation["context_id"]
                    if (str(value), field) in field_delta_index
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "split_id": split_id,
                    "split_kind": plan["split_kind"],
                    "field_name": field,
                    "selected_field_count": len(own),
                    "mean_jaccard_to_main_splits": float(np.mean(jaccards))
                    if jaccards
                    else 1.0,
                    "train_iqr": _iqr(train_values),
                    "validation_iqr": _iqr(validation_values),
                    "validation_to_train_iqr_ratio": _iqr(validation_values)
                    / max(_iqr(train_values), 1e-12),
                    "train_mean": float(np.mean(train_values)),
                    "validation_mean": float(np.mean(validation_values)),
                    "validation_to_train_mean_ratio": float(np.mean(validation_values))
                    / max(float(np.mean(train_values)), 1e-12),
                    "fit_role": "inner_train",
                    "outer_test_opened": False,
                }
            )
    return pd.DataFrame(rows)


def leakage_proxy_rows(
    *, role_path: str | Path, manifest: Mapping[str, object],
    candidate_metrics: pd.DataFrame,
    input_contract: Mapping[str, object] | None = None,
    context_path: str | Path | None = None,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    input_contract = input_contract or {}
    roles = pd.read_csv(role_path)
    excluded = roles[roles["selected_for_maneuver_label"].astype(bool)]
    allowed = roles[roles["allowed_in_maneuver_input"].astype(bool)]
    direct_overlap = set(excluded["feature_name"].astype(str)) & set(
        allowed["feature_name"].astype(str)
    )
    rows = [
        {
            "audit_scope": "global_feature_contract",
            "candidate_id": "all_raw_context_candidates",
            "split_id": "all_main_splits",
            "near_perfect_triggered": False,
            "label_source_count": len(excluded),
            "direct_label_source_overlap_count": len(direct_overlap),
            "deterministic_derivative_overlap_count": 0,
            "identity_feature_used": bool(
                input_contract.get("view_pilot_or_sortie_identity_used", False)
            ),
            "absolute_time_position_used": bool(
                input_contract.get("explicit_query_or_block_position_used", False)
            ),
            "time_like_allowed_field_count": int(
                input_contract.get("time_like_allowed_field_count", 0)
            ),
            "time_like_present_channel_count": int(
                input_contract.get("time_like_present_channel_count", 0)
            ),
            "shared_vehicle_cross_role_count": int(
                sum(row["shared_vehicle_unit_cross_role_count"] for row in manifest["folds"])
            ),
            "support_overlap_count": int(
                sum(row["support_overlap_count"] for row in manifest["folds"])
            ),
            "valid": not direct_overlap,
        }
    ]
    if context_path is not None and targets is not None:
        context = pd.read_json(context_path, lines=True).set_index("context_id")
        for plan in manifest["folds"]:
            if not plan["main_selection"]:
                continue
            split_id = str(plan["fold_id"])
            train = _target_subset(
                targets,
                split_id,
                "maneuver_intensity_classification",
                "train",
            )
            validation = _target_subset(
                targets,
                split_id,
                "maneuver_intensity_classification",
                "validation",
            )
            train_position = context.loc[
                train["context_id"].astype(str), "start_offset_ms"
            ].to_numpy(dtype=np.float64).reshape(-1, 1)
            validation_position = context.loc[
                validation["context_id"].astype(str), "start_offset_ms"
            ].to_numpy(dtype=np.float64).reshape(-1, 1)
            classifier = DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=17,
            ).fit(train_position, train["class_target"].to_numpy(dtype=np.int64))
            time_only_f1 = float(
                f1_score(
                    validation["class_target"].to_numpy(dtype=np.int64),
                    classifier.predict(validation_position),
                    labels=(0, 1, 2),
                    average="macro",
                    zero_division=0,
                )
            )
            rows.append(
                {
                    "audit_scope": "time_position_only_diagnostic",
                    "candidate_id": "explicit_block_position_only",
                    "split_id": split_id,
                    "near_perfect_triggered": time_only_f1 >= 0.95,
                    "label_source_count": len(excluded),
                    "direct_label_source_overlap_count": len(direct_overlap),
                    "deterministic_derivative_overlap_count": 0,
                    "identity_feature_used": False,
                    "absolute_time_position_used": True,
                    "time_like_allowed_field_count": int(
                        input_contract.get("time_like_allowed_field_count", 0)
                    ),
                    "time_like_present_channel_count": 0,
                    "time_position_only_macro_f1": time_only_f1,
                    "shared_vehicle_cross_role_count": 0,
                    "support_overlap_count": 0,
                    "valid": True,
                }
            )
    if not candidate_metrics.empty:
        near = candidate_metrics[
            (candidate_metrics["metric"] == "macro_f1")
            & (candidate_metrics["value"] >= 0.95)
        ][["candidate_id", "split_id"]].drop_duplicates()
        for item in near.itertuples(index=False):
            rows.append(
                {
                    "audit_scope": "near_perfect_candidate",
                    "candidate_id": item.candidate_id,
                    "split_id": item.split_id,
                    "near_perfect_triggered": True,
                    "label_source_count": len(excluded),
                    "direct_label_source_overlap_count": len(direct_overlap),
                    "deterministic_derivative_overlap_count": 0,
                    "identity_feature_used": bool(
                        input_contract.get(
                            "view_pilot_or_sortie_identity_used", False
                        )
                    ),
                    "absolute_time_position_used": bool(
                        input_contract.get(
                            "explicit_query_or_block_position_used", False
                        )
                    ),
                    "time_like_allowed_field_count": int(
                        input_contract.get("time_like_allowed_field_count", 0)
                    ),
                    "time_like_present_channel_count": int(
                        input_contract.get("time_like_present_channel_count", 0)
                    ),
                    "shared_vehicle_cross_role_count": 0,
                    "support_overlap_count": 0,
                    "valid": not direct_overlap,
                }
            )
    return pd.DataFrame(rows)


def _target_subset(frame, split_id, task_slug, role):
    return frame[
        (frame["fold_id"] == split_id)
        & (frame["task_slug"] == task_slug)
        & (frame["role"] == role)
        & (frame["status"] == "completed")
    ]


def _fixed_macro_f1(actual, predicted):
    from sklearn.metrics import f1_score

    return float(
        f1_score(actual, predicted, labels=(0, 1, 2), average="macro", zero_division=0)
    )


def _iqr(values):
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if len(values) else 0.0
