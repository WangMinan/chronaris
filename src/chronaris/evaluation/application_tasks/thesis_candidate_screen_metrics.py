"""Training-internal metrics for the thesis candidate screen."""

from __future__ import annotations

import math

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
