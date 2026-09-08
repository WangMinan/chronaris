"""Fit, resume, and evaluate one method under the fixed G4 consumer protocol."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import torch

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    ApplicationConsumerSmokeTargets,
)
from chronaris.evaluation.application_tasks.application_consumers import (
    CausalTCNEmissionModel,
    DurationViterbiParameters,
    LinearConsumerConfig,
    LinearFrozenConsumer,
    MiniRocketConsumerConfig,
    MiniRocketFrozenConsumer,
    TCNConsumerConfig,
    duration_constrained_viterbi_decode,
    fit_causal_tcn_emission,
    fit_duration_viterbi_parameters,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    classification_metrics,
    regression_metrics,
    segmentation_metrics,
)
from chronaris.representation import FusionStreamBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    sha256_file,
    write_deterministic_npz,
)


@dataclass(frozen=True, slots=True)
class ApplicationConsumerProtocol:
    linear: LinearConsumerConfig = LinearConsumerConfig()
    minirocket: MiniRocketConsumerConfig = MiniRocketConsumerConfig()
    tcn: TCNConsumerConfig = TCNConsumerConfig()
    label_used_for_encoder_training: bool = False


@dataclass(frozen=True, slots=True)
class ApplicationMethodConsumerResult:
    method_name: str
    status: str
    protocol_sha256: str
    metric_rows: tuple[Mapping[str, object], ...]
    workload_prediction_rows: tuple[Mapping[str, object], ...]
    unit_score_rows: tuple[Mapping[str, object], ...]
    tcn_training_rows: tuple[Mapping[str, object], ...]
    resource_rows: tuple[Mapping[str, object], ...]
    model_manifest: Mapping[str, object]
    component_status: Mapping[str, str]


def prepare_application_cpu_consumers(*, method_name, outputs, targets, output_root, fold_id, protocol=None, resume=True):
    """Fit/resume the two CPU components, usable while the GPU adapts an encoder."""
    resolved = protocol or ApplicationConsumerProtocol()
    root = Path(output_root) / method_name
    root.mkdir(parents=True, exist_ok=True)
    protocol_hash = _protocol_hash(method_name, outputs, targets, resolved, fold_id)
    loaded = _load_model_components(root=root, manifest_path=root / "consumer_manifest.json",
        protocol_sha256=protocol_hash, resume=resume)
    selected = {role: _select_targets(targets, outputs[role].sample_ids) for role in ("train", "validation")}
    status, elapsed = {}, {}
    for name, constructor, attribute, argument in (
        ("linear", LinearFrozenConsumer, "pooled_embedding", "validation_pooled"),
        ("minirocket", MiniRocketFrozenConsumer, "sequence_embedding", "validation_sequence")):
        if name in loaded:
            status[name], elapsed[name] = "resumed", 0.
            continue
        started = time.perf_counter()
        loaded[name] = constructor(getattr(resolved, name)).fit(
            getattr(outputs["train"], attribute).detach().cpu().numpy(),
            selected["train"]["workload_class"], selected["train"]["future_workload_mean"],
            **{argument: getattr(outputs["validation"], attribute).detach().cpu().numpy()},
            validation_class_target=selected["validation"]["workload_class"],
            validation_regression_target=selected["validation"]["future_workload_mean"])
        elapsed[name], status[name] = time.perf_counter() - started, "completed"
        _save_joblib_consumer(root / f"{name}.joblib", protocol_hash, loaded[name])
        paths = {key: root / ("causal_tcn.pt" if key == "tcn" else f"{key}.joblib") for key in loaded}
        partial = {"format": "chronaris.application_consumer_models.v2", "status": "cpu_prepared",
            "protocol_sha256": protocol_hash, "model_files": {key: {"path": str(path), "sha256": sha256_file(path)}
                for key, path in paths.items()}}
        temporary = root / "consumer_manifest.json.tmp"
        temporary.write_text(json.dumps(partial, indent=2) + "\n")
        temporary.replace(root / "consumer_manifest.json")
    return loaded, status, elapsed, protocol_hash


def run_application_method_consumers(
    *,
    method_name: str,
    outputs: Mapping[str, FusionStreamBatch],
    targets: ApplicationConsumerSmokeTargets,
    output_root: str | Path,
    fold_id: str,
    protocol: ApplicationConsumerProtocol | None = None,
    resume: bool = True,
) -> ApplicationMethodConsumerResult:
    resolved = protocol or ApplicationConsumerProtocol()
    root = Path(output_root) / method_name
    manifest_path = root / "consumer_manifest.json"
    started = time.perf_counter()
    loaded, component_status, component_elapsed, protocol_hash = prepare_application_cpu_consumers(
        method_name=method_name, outputs=outputs, targets=targets, output_root=output_root,
        fold_id=fold_id, protocol=resolved, resume=resume)
    train, validation = outputs["train"], outputs["validation"]
    train_targets, validation_targets = (_select_targets(targets, output.sample_ids) for output in (train, validation))
    paths = {"linear": root / "linear.joblib", "minirocket": root / "minirocket.joblib", "tcn": root / "causal_tcn.pt"}
    linear, minirocket = loaded["linear"], loaded["minirocket"]
    tcn_loaded = loaded.get("tcn")
    if tcn_loaded is None:
        tcn_started = time.perf_counter()
        tcn_result = fit_causal_tcn_emission(
            train.sequence_embedding,
            train_targets["maneuver_state"],
            config=resolved.tcn,
            validation_sequence=validation.sequence_embedding,
            validation_labels=validation_targets["maneuver_state"],
        )
        duration = fit_duration_viterbi_parameters(
            train_targets["maneuver_state"],
            class_count=resolved.tcn.class_count,
        )
        component_elapsed["causal_tcn"] = time.perf_counter() - tcn_started
        tcn_model = tcn_result.model
        training_rows = tuple(dict(row) for row in tcn_result.training_rows)
        _save_tcn_consumer(
            paths["tcn"],
            protocol_sha256=protocol_hash,
            tcn_result=tcn_result,
            duration=duration,
            protocol=resolved,
        )
        component_status["causal_tcn"] = "completed"
    else:
        tcn_model, duration, training_rows = tcn_loaded
        component_elapsed["causal_tcn"] = 0.0
        component_status["causal_tcn"] = "resumed"
    status = (
        "resumed"
        if all(value == "resumed" for value in component_status.values())
        else "completed"
    )
    resources = tuple(
        _resource_row(
            method_name,
            name,
            component_elapsed[name],
            status=component_status[name],
        )
        for name in ("linear", "minirocket", "causal_tcn")
    ) + (
        _resource_row(
            method_name,
            "all_consumers",
            time.perf_counter() - started,
            status=status,
        ),
    )
    metric_rows, workload_rows, unit_rows, prediction_payload = _evaluate_models(
        method_name=method_name,
        outputs=outputs,
        targets=targets,
        fold_id=fold_id,
        linear=linear,
        minirocket=minirocket,
        tcn_model=tcn_model,
        duration=duration,
        seed=resolved.linear.random_state,
    )
    prediction_path = root / "predictions.npz"
    prediction_hash = write_deterministic_npz(prediction_path, prediction_payload)
    model_manifest = {
        "format": "chronaris.application_consumer_models.v2",
        "method_name": method_name,
        "fold_id": fold_id,
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "component_status_current_run": component_status,
        "model_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_hash,
        "label_used_for_encoder_training": resolved.label_used_for_encoder_training,
        "consumer_fit_role": "train",
        "evaluation_roles": [role for role in ("validation", "held_out") if role in outputs],
        "minirocket_input_channel_count_before_filter": int(
            outputs["train"].sequence_embedding.shape[-1]
        ),
        "minirocket_input_channel_count_after_train_filter": int(
            len(minirocket.channel_indices)
        ),
        "minirocket_variance_filter_fit_role": "train",
        "linear_selected_classification_c": linear.selected_classification_c,
        "linear_selected_regression_alpha": linear.selected_regression_alpha,
        "minirocket_selected_classification_c": minirocket.selected_classification_c,
        "minirocket_selected_regression_alpha": minirocket.selected_regression_alpha,
        "hyperparameter_selection_role": (
            "validation" if resolved.linear.tune_on_validation else "fixed"
        ),
    }
    manifest_path.write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ApplicationMethodConsumerResult(
        method_name=method_name,
        status=status,
        protocol_sha256=protocol_hash,
        metric_rows=tuple(metric_rows),
        workload_prediction_rows=tuple(workload_rows),
        unit_score_rows=tuple(unit_rows),
        tcn_training_rows=tuple(training_rows),
        resource_rows=resources,
        model_manifest=model_manifest,
        component_status=component_status,
    )


def _evaluate_models(
    *, method_name, outputs, targets, fold_id, linear, minirocket, tcn_model, duration, seed
):
    smoke_only = bool(targets.manifest.get("smoke_only", True))
    metric_rows = []
    workload_rows = []
    unit_rows = []
    prediction_payload = {}
    tcn_model.eval()
    for role in ("validation", "held_out"):
        if role not in outputs:
            continue
        output = outputs[role]
        selected = _select_targets(targets, output.sample_ids)
        for consumer_name, consumer, values in (
            ("linear", linear, output.pooled_embedding.detach().cpu().numpy()),
            ("minirocket", minirocket, output.sequence_embedding.detach().cpu().numpy()),
        ):
            prediction = consumer.predict(values)
            _append_metrics(
                metric_rows,
                classification_metrics(
                    selected["workload_class"],
                    prediction["class_prediction"],
                    prediction["class_probability"],
                    prediction["classes"],
                ),
                method_name=method_name,
                task="simulated_future_workload_classification",
                consumer=consumer_name,
                seed=seed,
                fold=fold_id,
                role=role,
                smoke_only=smoke_only,
            )
            _append_metrics(
                metric_rows,
                regression_metrics(
                    selected["future_workload_mean"],
                    prediction["regression_prediction"],
                ),
                method_name=method_name,
                task="simulated_future_workload_regression",
                consumer=consumer_name,
                seed=seed,
                fold=fold_id,
                role=role,
                smoke_only=smoke_only,
            )
            for index, sample_id in enumerate(output.sample_ids):
                workload_rows.append(
                    {
                        "method": method_name,
                        "consumer": consumer_name,
                        "role": role,
                        "sample_id": sample_id,
                        "workload_class_true": int(selected["workload_class"][index]),
                        "workload_class_pred": int(prediction["class_prediction"][index]),
                        "future_workload_true": float(selected["future_workload_mean"][index]),
                        "future_workload_pred": float(prediction["regression_prediction"][index]),
                        "smoke_only": smoke_only,
                    }
                )
                unit_rows.extend(
                    (
                        _unit_row(
                            method_name, consumer_name, role, sample_id,
                            "classification_correct",
                            float(prediction["class_prediction"][index] == selected["workload_class"][index]),
                            "higher",
                            smoke_only=smoke_only,
                        ),
                        _unit_row(
                            method_name, consumer_name, role, sample_id,
                            "regression_absolute_error",
                            abs(float(prediction["regression_prediction"][index]) - float(selected["future_workload_mean"][index])),
                            "lower",
                            smoke_only=smoke_only,
                        ),
                    )
                )
            prefix = f"{role}_{consumer_name}"
            prediction_payload[f"{prefix}_class"] = np.asarray(
                prediction["class_prediction"], dtype=np.int64
            )
            prediction_payload[f"{prefix}_regression"] = np.asarray(
                prediction["regression_prediction"], dtype=np.float32
            )
        with torch.inference_mode():
            logits = tcn_model(output.sequence_embedding).detach().cpu()
        raw_prediction = logits.argmax(dim=-1)
        duration_prediction = duration_constrained_viterbi_decode(logits, duration)
        for consumer_name, prediction in (
            ("causal_tcn_raw", raw_prediction),
            ("causal_tcn_duration", duration_prediction),
        ):
            _append_metrics(
                metric_rows,
                segmentation_metrics(selected["maneuver_state"], prediction.numpy()),
                method_name=method_name,
                task="simulated_maneuver_state_segmentation",
                consumer=consumer_name,
                seed=seed,
                fold=fold_id,
                role=role,
                smoke_only=smoke_only,
            )
            for index, sample_id in enumerate(output.sample_ids):
                unit_rows.append(
                    _unit_row(
                        method_name, consumer_name, role, sample_id,
                        "frame_accuracy",
                        float(np.mean(prediction[index].numpy() == selected["maneuver_state"][index])),
                        "higher",
                        smoke_only=smoke_only,
                    )
                )
            prediction_payload[f"{role}_{consumer_name}_state"] = prediction.numpy().astype(np.int64)
        prediction_payload[f"{role}_sample_ids"] = np.asarray(output.sample_ids)
        prediction_payload[f"{role}_state_true"] = selected["maneuver_state"].astype(np.int64)
        prediction_payload[f"{role}_workload_class_true"] = selected["workload_class"].astype(np.int64)
        prediction_payload[f"{role}_workload_true"] = selected["future_workload_mean"].astype(np.float32)
    return metric_rows, workload_rows, unit_rows, prediction_payload


def _select_targets(targets, sample_ids):
    index = {sample_id: position for position, sample_id in enumerate(targets.sample_ids)}
    positions = [index[sample_id] for sample_id in sample_ids]
    return {
        "workload_class": targets.workload_class[positions].numpy(),
        "future_workload_mean": targets.future_workload_mean[positions].numpy(),
        "maneuver_state": targets.maneuver_state[positions].numpy(),
    }


def _append_metrics(
    rows, metrics, *, method_name, task, consumer, seed, fold, role, smoke_only
):
    for name, (value, direction) in metrics.items():
        available = value is not None and np.isfinite(value)
        rows.append(
            {
                "dataset": "model_independent_aviation_simulation",
                "task": task,
                "consumer": consumer,
                "method": method_name,
                "seed": seed,
                "fold": fold,
                "role": role,
                "metric": name,
                "value": float(value) if available else None,
                "direction": direction,
                "status": "available" if available else "unavailable",
                "reason": None if available else "metric_not_defined",
                "smoke_only": smoke_only,
            }
        )


def _unit_row(
    method, consumer, role, sample_id, metric, value, direction, *, smoke_only=True
):
    return {
        "method": method,
        "consumer": consumer,
        "role": role,
        "sample_id": sample_id,
        "metric": metric,
        "value": value,
        "direction": direction,
        "smoke_only": smoke_only,
    }


def _save_joblib_consumer(path, protocol_sha256, consumer):
    joblib.dump(
        {"protocol_sha256": protocol_sha256, "consumer": consumer},
        path,
    )


def _save_tcn_consumer(
    path, *, protocol_sha256, tcn_result, duration, protocol
):
    torch.save(
        {
            "format": "chronaris.application_tcn.v1",
            "protocol_sha256": protocol_sha256,
            "config": {**asdict(protocol.tcn), "device": "cpu"},
            "state_dict": tcn_result.model.state_dict(),
            "training_rows": list(tcn_result.training_rows),
            "best_epoch": tcn_result.best_epoch,
            "stopped_early": tcn_result.stopped_early,
            "class_weights": tcn_result.class_weights,
            "duration": {
                "initial_log_probability": duration.initial_log_probability,
                "transition_log_probability": duration.transition_log_probability,
                "minimum_duration": duration.minimum_duration,
                "maximum_duration": duration.maximum_duration,
                "train_sequence_count": duration.train_sequence_count,
            },
        },
        path,
    )


def _load_model_components(*, root, manifest_path, protocol_sha256, resume):
    if not resume or not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != protocol_sha256:
        return {}
    paths = {name: Path(item["path"]) for name, item in manifest["model_files"].items()}
    result = {}
    for name in ("linear", "minirocket"):
        item = manifest["model_files"].get(name)
        if item is None:
            continue
        if not paths[name].is_file() or sha256_file(paths[name]) != item["sha256"]:
            continue
        payload = joblib.load(paths[name])
        if payload.get("protocol_sha256") == protocol_sha256:
            result[name] = payload["consumer"]
    item = manifest["model_files"].get("tcn")
    if item is None:
        return result
    if not paths["tcn"].is_file() or sha256_file(paths["tcn"]) != item["sha256"]:
        return result
    tcn_payload = torch.load(paths["tcn"], map_location="cpu", weights_only=True)
    if tcn_payload.get("protocol_sha256") != protocol_sha256:
        return result
    config = TCNConsumerConfig(**tcn_payload["config"])
    model = CausalTCNEmissionModel(config)
    model.load_state_dict(tcn_payload["state_dict"])
    duration = DurationViterbiParameters(**tcn_payload["duration"])
    result["tcn"] = (
        model,
        duration,
        tuple(tcn_payload["training_rows"]),
    )
    return result


def _protocol_hash(method_name, outputs, targets, protocol, fold_id):
    payload = {
        "format": "chronaris.application_consumer_protocol.v2",
        "consumer_runtime_revision": "validation_selected_residual_tcn.v3",
        "method_name": method_name,
        "fold_id": fold_id,
        "checkpoint_sha256": outputs["train"].checkpoint_sha256,
        "sample_ids": {role: list(output.sample_ids) for role, output in outputs.items()},
        "target_manifest": targets.manifest,
        "consumer_config": asdict(protocol),
        "fit_role": "train",
        "evaluation_roles": [role for role in ("validation", "held_out") if role in outputs],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _resource_row(method, consumer, elapsed_s, *, status="completed"):
    return {
        "method": method,
        "consumer": consumer,
        "status": status,
        "elapsed_s": float(elapsed_s),
        "smoke_only": True,
    }
