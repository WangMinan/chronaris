"""Frozen public/Dingxin consumers with partial labels and grouped evaluation."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from chronaris.evaluation.application_tasks.application_consumers import (
    LinearConsumerConfig, MiniRocketConsumerConfig, MiniRocketFrozenConsumer)
from chronaris.evaluation.application_tasks.consumer_model_selection import (
    classifier_classes, fit_classifier, fit_regressor)
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256


def native_consumer_context(domain, data, fold):
    """Take identity and train-fitted target scales from the authoritative data bundle."""
    ids = fold.train_sample_ids + fold.validation_sample_ids
    if domain in {"cogpilot", "clare"}:
        groups = {row["sample_id"]: row["subject_id"] for row in data.sample_manifest}
        return {"domain": domain, "groups": {sample: groups[sample] for sample in ids},
                "regression": {}, "vehicle_groups": {}}
    if domain != "dingxin":
        raise ValueError("native consumers require a public or Dingxin domain")
    maneuver = data.fitted_targets.maneuver_targets
    maneuver = maneuver[maneuver.fold_id == fold.fold_id].set_index("context_id").loc[list(ids)]
    physiology = data.fitted_targets.physiology_targets
    physiology = physiology[(physiology.fold_id == fold.fold_id) & physiology.selected]
    fields = data.targets_by_fold[fold.fold_id].manifest["physiology_field_order"]
    scales = physiology.drop_duplicates("field_name").set_index("field_name").loc[fields]
    current = physiology.pivot(index="context_id", columns="field_name", values="current_value").loc[list(ids), fields]
    return {"domain": domain, "groups": maneuver.sortie_id.astype(str).to_dict(),
        "vehicle_groups": maneuver.vehicle_context_id.astype(str).to_dict(),
        "regression": {
            "maneuver_regression": {"fields": ["maneuver_score"], "scale": [float(maneuver.train_target_iqr.iloc[0])],
                "persistence": {sample: [float(maneuver.loc[sample, "current_maneuver_score"])] for sample in ids}},
            "physiology_regression": {"fields": list(fields), "scale": scales.train_scale.tolist(),
                "persistence": {sample: [float(v) if np.isfinite(v) else None for v in current.loc[sample]] for sample in ids}}}}


def _validate_inputs(outputs, targets, definitions, context):
    if "train" not in outputs or "validation" not in outputs or set(outputs) - {"train", "validation", "held_out"}:
        raise ValueError("consumer roles require train and validation")
    ids = [sample for output in outputs.values() for sample in output.sample_ids]
    if len(ids) != len(set(ids)) or not set(ids) <= set(targets.sample_ids) or not set(ids) <= set(context["groups"]):
        raise ValueError("consumer roles overlap or lack target/group coverage")
    if len({(output.method_name, output.fold_id, output.checkpoint_sha256) for output in outputs.values()}) != 1:
        raise ValueError("consumer representations have inconsistent provenance")
    if set(targets.values) != {task.name for task in definitions} or any(task.kind == "sequence_classification" for task in definitions):
        raise ValueError("native scalar consumers require matching scalar/field tasks")
    if context["domain"] in {"cogpilot", "clare"}:
        groups = [{context["groups"][sample] for sample in output.sample_ids} for output in outputs.values()]
        if any(left & right for i, left in enumerate(groups) for right in groups[i + 1:]):
            raise ValueError("public consumer subjects overlap between roles")
    elif context["domain"] == "dingxin":
        vehicle = [{context["vehicle_groups"][sample] for sample in output.sample_ids} for output in outputs.values()]
        if any(left & right for i, left in enumerate(vehicle) for right in vehicle[i + 1:]):
            raise ValueError("Dingxin shared vehicle contexts overlap between roles")
        if tuple(targets.manifest["fit_sample_ids"]) != outputs["train"].sample_ids:
            raise ValueError("Dingxin target calibration does not match consumer training role")
        if targets.sample_weights is None:
            raise ValueError("Dingxin consumers require explicit shared-view weights")
        index = {sample: i for i, sample in enumerate(targets.sample_ids)}
        for output in outputs.values():
            for group in {context["vehicle_groups"][sample] for sample in output.sample_ids}:
                weight = sum(float(targets.sample_weights[index[sample]]) for sample in output.sample_ids
                             if context["vehicle_groups"][sample] == group)
                if not np.isclose(weight, 1.):
                    raise ValueError("Dingxin view weights must sum to one per vehicle context")
    else:
        raise ValueError("unsupported native consumer domain")


def fit_native_consumers(*, outputs, targets, definitions, context, family="linear", seed=17,
                         minirocket_kernels=10_000):
    """Fit once on clean training representations; stress evaluation reuses this bundle."""
    _validate_inputs(outputs, targets, definitions, context)
    dingxin = context["domain"] == "dingxin"
    if family not in {"linear", "minirocket"} or (dingxin and family != "linear"):
        raise ValueError("Dingxin has only the approved fixed linear consumers")
    config = LinearConsumerConfig(random_state=seed, tune_on_validation=not dingxin)
    transformer = None
    if family == "minirocket":
        transformer = MiniRocketFrozenConsumer(MiniRocketConsumerConfig(n_kernels=minirocket_kernels, random_state=seed))
        train = outputs["train"]
        x = {"train": transformer.fit_features(train.sequence_embedding.numpy(), valid_cases=train.valid_mask.any(dim=1).numpy()),
             "validation": transformer.transform_features(outputs["validation"].sequence_embedding.numpy())}
    else:
        x = {role: output.pooled_embedding.numpy() for role, output in outputs.items() if role != "held_out"}
    index = {sample: i for i, sample in enumerate(targets.sample_ids)}
    position = {role: np.array([index[sample] for sample in outputs[role].sample_ids]) for role in x}
    models, fit_rows = {}, []
    for task in definitions:
        value, valid = targets.values[task.name].numpy(), targets.valid_masks[task.name].numpy()
        if value.ndim == 1:
            value, valid = value[:, None], valid[:, None]
        expected = task.output_dim if task.kind == "regression" else 1
        if value.shape[1:] != (expected,):
            raise ValueError("consumer target dimensions differ from task definition")
        for field in range(expected):
            selected = {role: valid[positions, field] for role, positions in position.items()}
            y = {role: value[positions, field][selected[role]] for role, positions in position.items()}
            if not len(y["train"]) or (task.kind == "classification" and len(np.unique(y["train"])) < 2):
                raise ValueError(f"consumer training labels unavailable: {task.name}:{field}")
            if not dingxin and not len(y["validation"]):
                raise ValueError(f"public consumer validation labels unavailable: {task.name}:{field}")
            train_ids = tuple(np.array(outputs["train"].sample_ids)[selected["train"]])
            weights = targets.sample_weights.numpy()[position["train"]][selected["train"]] if dingxin else None
            validation_groups = None if dingxin else np.array([context["groups"][sample]
                for sample in outputs["validation"].sample_ids])[selected["validation"]]
            args = (x["train"][selected["train"]], y["train"],
                    None if dingxin else x["validation"][selected["validation"]], None if dingxin else y["validation"])
            common = dict(scaler_with_mean=family == "linear", train_sample_weight=weights, validation_groups=validation_groups)
            if task.kind == "classification":
                if np.any(y["train"] < 0) or np.any(y["train"] >= task.output_dim):
                    raise ValueError("consumer training class exceeds declared task classes")
                model, parameter = fit_classifier(*args, c_values=(1.,) if dingxin else config.classification_c_grid,
                    random_state=seed, classification_labels=list(range(task.output_dim)),
                    solver="lbfgs" if family == "linear" else "liblinear_ovr", **common)
            else:
                model, parameter = fit_regressor(*args, alpha_values=(1.,) if dingxin else config.regression_alpha_grid, **common)
            models[(task.name, field)] = model
            fit_rows.append({"task": task.name, "field": field, "selected_parameter": parameter,
                "train_sample_ids": list(train_ids), "train_weight_sum": float(len(train_ids) if weights is None else weights.sum()),
                "selection_role": "fixed_parameter" if dingxin else "validation_subject_mean"})
    train = outputs["train"]
    return {"models": models, "transformer": transformer, "definitions": definitions, "family": family,
        "config": asdict(config), "fit_rows": fit_rows, "train_sample_ids": train.sample_ids,
        "method_name": train.method_name, "fold_id": train.fold_id, "checkpoint_sha256": train.checkpoint_sha256,
        "context": context}


def regression_metrics(truth, prediction, *, sample_ids, weights=None, persistence=None, no_observation=None):
    """Retain every valid target; missing persistence cannot hide model errors."""
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1 or len(sample_ids) != len(truth):
        raise ValueError("regression metric arrays differ")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("valid regression targets/predictions must be finite")
    weights = np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != truth.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("regression metric weights are invalid")
    if not len(truth):
        return {"status": "unavailable_no_valid_targets", "support": 0}
    error = prediction - truth
    contribution = weights * error ** 2
    total = float(contribution.sum())
    worst = np.argsort(-contribution, kind="stable")[:5]
    result = {"status": "completed", "support": len(truth), "weight_sum": float(weights.sum()),
        "rmse": float(np.sqrt(total / weights.sum())), "mae": float(np.average(np.abs(error), weights=weights)),
        "p95_absolute_error": float(np.quantile(np.abs(error), .95)),
        "top_five_squared_error_fraction": float(contribution[worst].sum() / total) if total > 0 else 0.,
        "top_five_windows": [{"sample_id": str(sample_ids[i]), "weighted_squared_error": float(contribution[i])} for i in worst],
        "no_observation_fraction": float(np.average(no_observation, weights=weights)) if no_observation is not None else None,
        "spearman": float(spearmanr(truth, prediction).statistic) if len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(prediction) > 0 else None}
    if persistence is not None:
        persistence = np.asarray(persistence, dtype=float)
        if persistence.shape != truth.shape:
            raise ValueError("persistence target shape differs")
        paired = np.isfinite(persistence)
        baseline = float(np.sum(weights[paired] * (truth[paired] - persistence[paired]) ** 2))
        result.update(persistence_support=int(paired.sum()),
            persistence_rmse=float(np.sqrt(baseline / weights[paired].sum())) if paired.any() else None,
            skill_vs_persistence=1. - float(contribution[paired].sum()) / baseline if baseline > 0 else None)
    return result


def evaluate_native_consumers(bundle, *, output, targets, context=None):
    """Evaluate frozen models without fitting, including zero-observation windows."""
    context = bundle["context"] if context is None else context
    if (output.method_name, output.fold_id, output.checkpoint_sha256) != (
            bundle["method_name"], bundle["fold_id"], bundle["checkpoint_sha256"]):
        raise ValueError("evaluation encoder differs from fitted consumer")
    if set(output.sample_ids) & set(bundle["train_sample_ids"]):
        raise ValueError("evaluation cannot score consumer training windows")
    x = output.pooled_embedding.numpy() if bundle["transformer"] is None else bundle["transformer"].transform_features(output.sequence_embedding.numpy())
    index = {sample: i for i, sample in enumerate(targets.sample_ids)}
    positions = [index[sample] for sample in output.sample_ids]
    groups = np.array([context["groups"][sample] for sample in output.sample_ids])
    weights = np.ones(len(positions)) if targets.sample_weights is None else targets.sample_weights.numpy()[positions]
    empty = ~output.valid_mask.any(dim=1).numpy()
    group_rows, prediction_rows, aggregate_rows = [], [], []
    for task in bundle["definitions"]:
        value, valid = targets.values[task.name].numpy()[positions], targets.valid_masks[task.name].numpy()[positions]
        if value.ndim == 1:
            value, valid = value[:, None], valid[:, None]
        for field in range(value.shape[1]):
            model = bundle["models"][(task.name, field)]
            prediction = model.predict(x)
            probability = None
            if task.kind == "classification":
                probability = np.zeros((len(positions), task.output_dim))
                probability[:, classifier_classes(model)] = model.predict_proba(x)
            if not np.isfinite(prediction).all() or (probability is not None and not np.isfinite(probability).all()):
                raise ValueError("non-finite consumer prediction")
            calibration = context["regression"].get(task.name)
            scale = float(calibration["scale"][field]) if calibration else 1.
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("regression evaluation scale must be positive and train-fitted")
            for i, sample in enumerate(output.sample_ids):
                prediction_rows.append({"sample_id": sample, "group_id": str(groups[i]), "task": task.name, "field": field,
                    "target_valid": bool(valid[i, field]), "target": float(value[i, field]) if valid[i, field] else None,
                    "prediction": float(prediction[i]), "probability": probability[i].tolist() if probability is not None else None,
                    "no_observation": bool(empty[i]), "sample_weight": float(weights[i])})
            for group in np.unique(groups):
                selected = (groups == group) & valid[:, field]
                ids = np.array(output.sample_ids)[selected]
                truth, pred, weight, noobs = value[selected, field], prediction[selected], weights[selected], empty[selected]
                prob = probability[selected] if probability is not None else None
                baseline = np.array([calibration["persistence"][sample][field] for sample in ids], dtype=float) if calibration else None
                if context["domain"] == "dingxin" and task.name.startswith("maneuver_") and len(ids):
                    vehicle = np.array([context["vehicle_groups"][sample] for sample in ids])
                    unique = np.unique(vehicle)
                    if any(np.ptp(truth[vehicle == v]) != 0 for v in unique):
                        raise ValueError("shared vehicle views have inconsistent maneuver targets")
                    truth = np.array([truth[vehicle == v][0] for v in unique])
                    pred = np.array([np.average(pred[vehicle == v], weights=weight[vehicle == v]) for v in unique])
                    if prob is not None:
                        prob = np.array([np.average(prob[vehicle == v], axis=0, weights=weight[vehicle == v]) for v in unique])
                        pred = prob.argmax(axis=1)
                    baseline = np.array([baseline[vehicle == v][0] for v in unique]) if baseline is not None else None
                    noobs = np.array([noobs[vehicle == v].all() for v in unique])
                    ids, weight = unique, np.ones(len(unique))
                    aggregate_rows.extend({"vehicle_context_id": str(sample), "group_id": str(group), "task": task.name,
                        "target": float(truth[i]), "prediction": float(pred[i]),
                        "probability": prob[i].tolist() if prob is not None else None} for i, sample in enumerate(ids))
                if task.kind == "regression":
                    metrics = regression_metrics(truth / scale, pred / scale, sample_ids=ids, weights=weight,
                        persistence=baseline / scale if baseline is not None else None, no_observation=noobs)
                    metrics["target_scale"] = scale
                    if metrics["status"] == "completed":
                        metrics["native_rmse"] = metrics["rmse"] * scale
                elif len(ids):
                    labels = list(range(task.output_dim))
                    metrics = {"status": "completed", "support": len(ids),
                        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
                        "balanced_accuracy": float(recall_score(truth, pred, labels=np.unique(truth), average="macro", zero_division=0)),
                        "confusion_matrix": confusion_matrix(truth, pred, labels=labels).tolist(),
                        "no_observation_fraction": float(np.mean(noobs))}
                else:
                    metrics = {"status": "unavailable_no_valid_targets", "support": 0}
                group_rows.append({"group_id": str(group), "task": task.name, "field": field,
                    "field_name": calibration["fields"][field] if calibration else task.name, **metrics})
    task_summary = []
    for task in bundle["definitions"]:
        metric = "macro_f1" if task.kind == "classification" else "rmse"
        rows = [row for row in group_rows if row["task"] == task.name and row["status"] == "completed"]
        # Fields are equal within each subject/sortie, then subjects/sorties are equal.
        by_group = [float(np.mean([row[metric] for row in rows if row["group_id"] == group]))
                    for group in sorted({row["group_id"] for row in rows})]
        task_summary.append({"task": task.name, "metric": metric, "value": float(np.mean(by_group)) if by_group else None,
            "group_count": len(by_group), "aggregation": "field_mean_then_group_mean",
            "error_units": "training_scale" if task.name in context["regression"] else "native"})
    return {"task_summary": task_summary, "group_metrics": group_rows, "prediction_rows": prediction_rows,
        "vehicle_prediction_rows": aggregate_rows, "family": bundle["family"],
        "independent_unit": "sortie_descriptive_only" if context["domain"] == "dingxin" else "subject",
        "no_observation_fraction": float(np.mean(empty)), "sample_count": len(positions)}


def run_native_method_consumers(*, outputs, targets, definitions, context, output_root,
                                label_used_for_encoder_training, seed=17, minirocket_kernels=10_000):
    """Save reusable clean consumers and all predictions; reject stale cached evidence."""
    _validate_inputs(outputs, targets, definitions, context)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "chronaris.v4_native_grouped_consumers.v1", "context": context,
        "source_code_sha256": v4_workflow_source_sha256(), "target_manifest": targets.manifest,
        "task_definitions": [asdict(task) for task in definitions], "seed": seed, "minirocket_kernels": minirocket_kernels,
        "label_used_for_encoder_training": label_used_for_encoder_training,
        "consumer_selection_role": "fixed_parameters" if context["domain"] == "dingxin" else "validation_subject_mean",
        "roles": {role: {"sample_ids": output.sample_ids, "source_sample_hashes": output.source_sample_hashes,
            "method_name": output.method_name, "fold_id": output.fold_id, "checkpoint_sha256": output.checkpoint_sha256}
            for role, output in outputs.items()}}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode())
    tensors = [tensor for role in sorted(outputs) for tensor in (outputs[role].sequence_embedding,
        outputs[role].valid_mask, outputs[role].pooled_embedding, outputs[role].timestamps_s)]
    tensors += [tensor for name in sorted(targets.values) for tensor in (targets.values[name], targets.valid_masks[name])]
    if targets.sample_weights is not None:
        tensors.append(targets.sample_weights)
    digest.update(json.dumps(targets.sample_ids).encode())
    for tensor in tensors:
        array = tensor.detach().cpu().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    protocol_hash = digest.hexdigest()
    manifest_path = root / "consumer_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["protocol_sha256"] != protocol_hash:
            raise ValueError("native consumer source/data/config changed; use a new output root")
    else:
        manifest_path.write_text(json.dumps(manifest | {"protocol_sha256": protocol_hash}, ensure_ascii=False, indent=2) + "\n")
    results = {}
    for family in ("linear",) if context["domain"] == "dingxin" else ("linear", "minirocket"):
        model_path, result_path = root / f"{family}.joblib", root / f"{family}_results.json"
        if model_path.exists():
            payload = joblib.load(model_path)
            if payload["protocol_sha256"] != protocol_hash:
                raise ValueError("saved native consumer provenance changed")
            bundle, fit_elapsed = payload["consumer"], payload["fit_elapsed_s"]
        else:
            started = time.perf_counter()
            bundle = fit_native_consumers(outputs=outputs, targets=targets, definitions=definitions, context=context,
                                          family=family, seed=seed, minirocket_kernels=minirocket_kernels)
            fit_elapsed = time.perf_counter() - started
            temporary = model_path.with_suffix(".joblib.tmp")
            joblib.dump({"protocol_sha256": protocol_hash, "consumer": bundle, "fit_elapsed_s": fit_elapsed}, temporary)
            temporary.replace(model_path)
        result = {"protocol_sha256": protocol_hash, "fit_rows": bundle["fit_rows"], "fit_elapsed_s": fit_elapsed,
            "evaluations": {role: evaluate_native_consumers(bundle, output=output, targets=targets)
                            for role, output in outputs.items() if role != "train"}}
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(result_path)
        results[family] = {"result_path": str(result_path), "model_path": str(model_path),
            "fit_elapsed_s": fit_elapsed, "task_summary": {role: values["task_summary"] for role, values in result["evaluations"].items()}}
    return {"protocol_sha256": protocol_hash, "manifest_path": str(manifest_path), "components": results}
