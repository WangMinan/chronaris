"""Training-internal metrics for the thesis candidate screen."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.pretext import ExplicitTimeShiftHead
from chronaris.representation import (
    build_explicit_time_shift_inputs,
    select_observation_batch,
)


def dingxin_validation_metrics(adapter, provider, fold, targets, batch_size, seed):
    train_embedding = export_pooled_embeddings(
        adapter, provider, fold.train_sample_ids, batch_size
    )
    validation_embedding = export_pooled_embeddings(
        adapter, provider, fold.validation_sample_ids, batch_size
    )
    train_positions = {
        sample_id: index for index, sample_id in enumerate(fold.train_sample_ids)
    }
    validation_positions = {
        sample_id: index
        for index, sample_id in enumerate(fold.validation_sample_ids)
    }
    maneuver = targets[
        targets["task_slug"].astype(str) == "maneuver_intensity_classification"
    ]
    maneuver_train = maneuver[maneuver["role"].astype(str) == "train"]
    maneuver_validation = maneuver[maneuver["role"].astype(str) == "validation"]
    train_rows = [
        train_positions[value] for value in maneuver_train["context_id"].astype(str)
    ]
    validation_rows = [
        validation_positions[value]
        for value in maneuver_validation["context_id"].astype(str)
    ]
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        ),
    ).fit(
        train_embedding[train_rows],
        maneuver_train["class_target"].to_numpy(dtype=int),
    )
    maneuver_prediction = classifier.predict(validation_embedding[validation_rows])
    response = targets[
        targets["task_slug"].astype(str) == "physiology_response_prediction"
    ]
    response_train = response[
        (response["role"].astype(str) == "train")
        & np.isfinite(response["continuous_target"])
    ]
    response_validation = response[
        (response["role"].astype(str) == "validation")
        & np.isfinite(response["continuous_target"])
    ]
    response_train_rows = [
        train_positions[value] for value in response_train["context_id"].astype(str)
    ]
    response_validation_rows = [
        validation_positions[value]
        for value in response_validation["context_id"].astype(str)
    ]
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        train_embedding[response_train_rows],
        response_train["continuous_target"].to_numpy(dtype=float),
    )
    response_prediction = regressor.predict(
        validation_embedding[response_validation_rows]
    )
    response_truth = response_validation["continuous_target"].to_numpy(dtype=float)
    rho = spearmanr(response_truth, response_prediction).correlation
    return {
        "validation_maneuver_macro_f1": f1_score(
            maneuver_validation["class_target"].to_numpy(dtype=int),
            maneuver_prediction,
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        ),
        "validation_response_rmse": math.sqrt(
            mean_squared_error(response_truth, response_prediction)
        ),
        "validation_response_spearman": float(rho) if math.isfinite(rho) else None,
    }


def public_validation_metrics(
    adapter,
    provider,
    fold,
    target_by_id,
    batch_size,
    seed,
    *,
    score_by_id=None,
):
    train_embedding = export_pooled_embeddings(
        adapter, provider, fold.train_sample_ids, batch_size
    )
    validation_embedding = export_pooled_embeddings(
        adapter, provider, fold.validation_sample_ids, batch_size
    )
    train_target = np.asarray(
        [target_by_id[value] for value in fold.train_sample_ids], dtype=int
    )
    validation_target = np.asarray(
        [target_by_id[value] for value in fold.validation_sample_ids], dtype=int
    )
    prediction = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        ),
    ).fit(train_embedding, train_target).predict(validation_embedding)
    metrics = {
        "validation_macro_f1": f1_score(
            validation_target,
            prediction,
            average="macro",
            zero_division=0,
        ),
        "validation_balanced_accuracy": balanced_accuracy_score(
            validation_target, prediction
        ),
    }
    if score_by_id is not None:
        train_score = np.asarray(
            [score_by_id[value] for value in fold.train_sample_ids], dtype=float
        )
        validation_score = np.asarray(
            [score_by_id[value] for value in fold.validation_sample_ids], dtype=float
        )
        score_prediction = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
            train_embedding, train_score
        ).predict(validation_embedding)
        rho = spearmanr(validation_score, score_prediction).correlation
        metrics.update(
            {
                "validation_score_rmse": math.sqrt(
                    mean_squared_error(validation_score, score_prediction)
                ),
                "validation_score_spearman": (
                    float(rho) if math.isfinite(rho) else None
                ),
            }
        )
    return metrics


def summarize_candidate_training(result, payload):
    best_epoch = int(payload["best_epoch"])
    best = next(row for row in payload["epoch_rows"] if row["epoch"] == best_epoch)
    mechanism = best["mechanism_validation"]
    return {
        "best_epoch": best_epoch,
        "validation_self_supervised_loss": float(
            payload["best_public_selection_loss"]
        ),
        "shift_accuracy": mechanism.get("explicit_time_shift_accuracy"),
        "pair_positive_similarity": mechanism.get(
            "event_pair_positive_similarity"
        ),
        "pair_negative_similarity": mechanism.get(
            "event_pair_negative_similarity"
        ),
        "pair_recall_at_1": mechanism.get("event_pair_recall_at_1"),
        "pair_count": mechanism.get("event_pair_count", 0),
        "mechanism_terms": mechanism.get("terms", []),
        "parameter_count": int(payload["parameter_count"]),
        "training_elapsed_s": float(payload["training_elapsed_s"]),
        "protocol_sha256": result.protocol_sha256,
        "checkpoint_path": result.best_checkpoint_path,
    }


def balanced_shift_accuracy(
    adapter,
    payload,
    *,
    batch,
    provider,
    sample_ids,
    batch_size,
):
    device = next(adapter.encoder.parameters()).device
    head = ExplicitTimeShiftHead(64).to(device)
    head.load_state_dict(payload["explicit_time_shift_head_state_dict"], strict=True)
    head.eval()
    correct = 0
    count = 0
    with torch.inference_mode():
        for offset in range(0, len(sample_ids), batch_size):
            ids = sample_ids[offset : offset + batch_size]
            raw = (
                provider(ids)
                if provider is not None
                else select_observation_batch(batch, ids)
            )
            normalized = adapter.normalizer.transform(raw)
            for class_index in range(5):
                inputs = build_explicit_time_shift_inputs(
                    normalized,
                    tuple("0" * 64 for _ in ids),
                    class_indices=(class_index,) * len(ids),
                )
                shifted = adapter.encoder(
                    move_observation_batch(inputs.shifted_batch, device=device)
                )
                prediction = head(
                    shifted.sequence_embedding,
                    shifted.modality_available_mask,
                ).argmax(dim=-1)
                correct += int((prediction == class_index).sum().item())
                count += len(ids)
    return correct / count


def export_pooled_embeddings(adapter, provider, sample_ids, batch_size):
    rows = []
    for offset in range(0, len(sample_ids), batch_size):
        batch = provider(sample_ids[offset : offset + batch_size])
        rows.append(adapter(batch).pooled_embedding.detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


_THESIS_CANDIDATES = (
    "base",
    "explicit_shift",
    "semantic_pair",
    "both_objectives",
)
_EXPECTED_STAGE_ROWS = {
    "simulation": 12,
    "dingxin": 60,
    "cogpilot": 12,
    "clare": 60,
}
_APPLICATION_METRICS = {
    "dingxin": {
        "validation_maneuver_macro_f1": "higher",
        "validation_response_rmse": "lower",
        "validation_response_spearman": "higher",
    },
    "cogpilot": {
        "validation_macro_f1": "higher",
        "validation_balanced_accuracy": "higher",
    },
    "clare": {
        "validation_macro_f1": "higher",
        "validation_balanced_accuracy": "higher",
        "validation_score_rmse": "lower",
        "validation_score_spearman": "higher",
    },
}


def audit_thesis_candidate_screen(state):
    """Audit the frozen v3.2 screen and select only on preregistered gates."""

    rows = tuple(state.get("rows", ()))
    units = [row.get("unit") for row in rows]
    stage_counts = Counter(row.get("stage") for row in rows)
    candidate_counts = Counter(row.get("candidate") for row in rows)
    seeds = sorted({int(row["seed"]) for row in rows if "seed" in row})
    critical_values = [
        row.get(name)
        for row in rows
        for name in ("validation_self_supervised_loss", "training_elapsed_s")
    ]
    protocol_checks = {
        "completed_144_units": (
            bool(state.get("completed_for_requested_stages"))
            and len(rows) == sum(_EXPECTED_STAGE_ROWS.values())
        ),
        "unique_units": len(units) == len(set(units)),
        "expected_stage_counts": dict(stage_counts) == _EXPECTED_STAGE_ROWS,
        "balanced_candidates": candidate_counts
        == Counter({candidate: 36 for candidate in _THESIS_CANDIDATES}),
        "expected_seeds": seeds == [17, 29, 43],
        "outer_results_closed": state.get("outer_results_opened") is False,
        "finite_losses_and_runtime": all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in critical_values
        ),
        "checkpoints_present": all(
            Path(str(row.get("checkpoint_path", ""))).is_file() for row in rows
        ),
        "frozen_source_identity": all(
            isinstance(state.get(name), str) and len(state[name]) == 64
            for name in ("source_code_sha256", "runner_sha256")
        )
        and isinstance(state.get("source_commit"), str)
        and len(state["source_commit"]) == 40,
    }

    simulation = {
        (int(row["seed"]), row["candidate"]): row
        for row in rows
        if row.get("stage") == "simulation"
    }
    shift_rows = []
    for seed in seeds:
        selected = simulation.get((seed, "both_objectives"), {})
        comparator = simulation.get((seed, "semantic_pair"), {})
        accuracy = selected.get("balanced_shift_accuracy")
        baseline = comparator.get("balanced_shift_accuracy")
        shift_rows.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "no_shift_accuracy": baseline,
                "above_random_and_comparator": (
                    accuracy is not None
                    and baseline is not None
                    and accuracy > 0.2
                    and accuracy > baseline
                ),
            }
        )
    shift_pass = sum(row["above_random_and_comparator"] for row in shift_rows) >= 2

    pair_rows = []
    for seed in seeds:
        row = simulation.get((seed, "both_objectives"), {})
        positive = row.get("pair_positive_similarity")
        negative = row.get("pair_negative_similarity")
        pair_rows.append(
            {
                "seed": seed,
                "positive_similarity": positive,
                "negative_similarity": negative,
                "similarity_gap": (
                    None if positive is None or negative is None else positive - negative
                ),
                "recall_at_1": row.get("pair_recall_at_1"),
                "correct_above_mismatch": (
                    positive is not None and negative is not None and positive > negative
                ),
            }
        )
    pair_pass = sum(row["correct_above_mismatch"] for row in pair_rows) >= 2

    physical = {}
    for stage in _EXPECTED_STAGE_ROWS:
        terms = [
            term
            for row in rows
            if row.get("stage") == stage
            for term in row.get("mechanism_terms", ())
            if term.get("term_name") == "chronaris_physical_consistency"
        ]
        active = [term for term in terms if term.get("status") == "active"]
        raw = [
            float(term["raw_loss"])
            for term in active
            if term.get("raw_loss") is not None
        ]
        physical[stage] = {
            "active_count": len(active),
            "unavailable_count": sum(
                term.get("status") == "unavailable" for term in terms
            ),
            "maximum_active_raw_loss": max(raw) if raw else None,
        }
    physical_scope_pass = (
        physical.get("simulation", {}).get("active_count", 0) > 0
        and physical.get("dingxin", {}).get("active_count", 0) > 0
        and physical.get("cogpilot", {}).get("active_count", 0) > 0
        and physical.get("clare", {}).get("active_count", 0) == 0
        and physical.get("clare", {}).get("unavailable_count", 0) == 60
        and all(
            details["active_count"] == 0
            or details["maximum_active_raw_loss"] is not None
            for details in physical.values()
        )
        and all(
            value is None or math.isfinite(value)
            for value in (
                details["maximum_active_raw_loss"] for details in physical.values()
            )
        )
    )

    candidate_gate_counts = {
        "base": 0,
        "explicit_shift": int(shift_pass),
        "semantic_pair": int(pair_pass),
        "both_objectives": int(shift_pass) + int(pair_pass),
    }
    selected_candidate = (
        max(candidate_gate_counts, key=candidate_gate_counts.get)
        if all(protocol_checks.values())
        and physical_scope_pass
        and shift_pass
        and pair_pass
        else None
    )
    metric_rows = _summarize_application_metrics(rows)
    candidate_rows = []
    for candidate in _THESIS_CANDIDATES:
        values = [row for row in rows if row.get("candidate") == candidate]
        candidate_rows.append(
            {
                "candidate": candidate,
                "new_mechanism_gates_passed": candidate_gate_counts[candidate],
                "parameter_count_median": float(
                    np.median([row["parameter_count"] for row in values])
                ),
                "training_elapsed_s_median": float(
                    np.median([row["training_elapsed_s"] for row in values])
                ),
                "training_elapsed_s_total": float(
                    sum(row["training_elapsed_s"] for row in values)
                ),
                "selected": candidate == selected_candidate,
            }
        )
    return {
        "format": "chronaris.thesis_candidate_selection.v1",
        "protocol_version": state.get("protocol_version"),
        "source_commit": state.get("source_commit"),
        "source_code_sha256": state.get("source_code_sha256"),
        "runner_sha256": state.get("runner_sha256"),
        "row_count": len(rows),
        "protocol_checks": protocol_checks,
        "protocol_gate_passed": all(protocol_checks.values()),
        "physical_scope": physical,
        "physical_scope_gate_passed": physical_scope_pass,
        "explicit_shift_gate": {"passed": shift_pass, "rows": shift_rows},
        "event_pair_gate": {"passed": pair_pass, "rows": pair_rows},
        "candidate_rows": candidate_rows,
        "application_metric_rows": metric_rows,
        "selected_candidate": selected_candidate,
        "full_mechanism_gates_pending": [
            "continuous_evolution_ablation",
            "physical_constraint_ablation",
            "safe_single_stream_bypass",
        ],
    }


def _summarize_application_metrics(rows):
    output = []
    for stage, metrics in _APPLICATION_METRICS.items():
        for candidate in _THESIS_CANDIDATES:
            selected = [
                row
                for row in rows
                if row.get("stage") == stage and row.get("candidate") == candidate
            ]
            for metric, direction in metrics.items():
                values = np.asarray(
                    [row[metric] for row in selected if row.get(metric) is not None],
                    dtype=float,
                )
                output.append(
                    {
                        "stage": stage,
                        "candidate": candidate,
                        "metric": metric,
                        "direction": direction,
                        "count": int(len(values)),
                        "median": float(np.median(values)),
                        "q1": float(np.percentile(values, 25)),
                        "q3": float(np.percentile(values, 75)),
                        "worst": float(
                            np.min(values) if direction == "higher" else np.max(values)
                        ),
                    }
                )
    return output
