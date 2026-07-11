"""Fit, resume, and evaluate one Dingxin representation method."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from chronaris.evaluation.application_tasks.application_metrics import (
    classification_metrics,
    regression_metrics,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_models import (
    DingxinConsumerConfig,
    fit_dingxin_task_consumer,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_targets import (
    DingxinFoldConsumerTargets,
    MANEUVER_TASK,
    RESPONSE_TASK,
)
from chronaris.representation import FusionStreamBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class DingxinMethodConsumerResult:
    method_name: str
    fold_id: str
    status: str
    protocol_sha256: str
    metric_rows: tuple[Mapping[str, object], ...]
    prediction_rows: tuple[Mapping[str, object], ...]
    resource_rows: tuple[Mapping[str, object], ...]
    component_status: Mapping[str, str]
    manifest: Mapping[str, object]


def run_dingxin_method_consumers(
    *,
    method_name: str,
    fold_id: str,
    outputs: Mapping[str, FusionStreamBatch],
    targets: DingxinFoldConsumerTargets,
    output_root: str | Path,
    config: DingxinConsumerConfig | None = None,
    resume: bool = True,
) -> DingxinMethodConsumerResult:
    resolved = config or DingxinConsumerConfig()
    root = Path(output_root) / fold_id / method_name
    root.mkdir(parents=True, exist_ok=True)
    protocol_hash = _protocol_hash(
        method_name=method_name,
        fold_id=fold_id,
        outputs=outputs,
        targets=targets,
        config=resolved,
    )
    manifest_path = root / "consumer_manifest.json"
    existing = _load_existing_components(
        manifest_path=manifest_path,
        protocol_sha256=protocol_hash,
        resume=resume,
    )
    train = outputs["train"]
    component_status = {}
    resource_rows = []
    consumers = {}
    for consumer_name in ("linear", "minirocket"):
        started = time.perf_counter()
        consumer = existing.get(consumer_name)
        path = root / f"{consumer_name}.joblib"
        if consumer is None:
            consumer = fit_dingxin_task_consumer(
                consumer_name=consumer_name,
                pooled_embedding=train.pooled_embedding.detach().cpu().numpy(),
                sequence_embedding=train.sequence_embedding.detach().cpu().numpy(),
                sample_ids=train.sample_ids,
                targets=targets,
                config=resolved,
            )
            _atomic_joblib_dump(
                path,
                {"protocol_sha256": protocol_hash, "consumer": consumer},
            )
            status = "completed"
        else:
            status = "resumed"
        consumers[consumer_name] = consumer
        component_status[consumer_name] = status
        resource_rows.append(
            {
                "fold_id": fold_id,
                "method": method_name,
                "consumer": consumer_name,
                "status": status,
                "elapsed_s": time.perf_counter() - started,
                "smoke_only": True,
            }
        )
    metric_rows, prediction_rows = _evaluate_consumers(
        method_name=method_name,
        fold_id=fold_id,
        outputs=outputs,
        targets=targets,
        consumers=consumers,
        seed=resolved.random_state,
    )
    prediction_path = root / "prediction_rows.csv"
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    prediction_hash = sha256_file(prediction_path)
    manifest = {
        "format": "chronaris.dingxin_window_consumers.v1",
        "method_name": method_name,
        "fold_id": fold_id,
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "component_status_current_run": component_status,
        "consumer_files": {
            name: {
                "path": str(root / f"{name}.joblib"),
                "sha256": sha256_file(root / f"{name}.joblib"),
                "model": consumers[name].to_manifest(),
            }
            for name in consumers
        },
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_hash,
        "representation_checkpoint_sha256": {
            role: output.checkpoint_sha256 for role, output in outputs.items()
        },
        "target_manifest": targets.to_manifest(),
        "consumer_fit_role": "train",
        "evaluation_roles": ["validation", "held_out"],
        "label_used_for_encoder_training": False,
        "smoke_only": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = (
        "resumed"
        if all(value == "resumed" for value in component_status.values())
        else "completed"
    )
    return DingxinMethodConsumerResult(
        method_name=method_name,
        fold_id=fold_id,
        status=status,
        protocol_sha256=protocol_hash,
        metric_rows=tuple(metric_rows),
        prediction_rows=tuple(prediction_rows),
        resource_rows=tuple(resource_rows),
        component_status=component_status,
        manifest=manifest,
    )


def _evaluate_consumers(
    *, method_name, fold_id, outputs, targets, consumers, seed
):
    metric_rows = []
    prediction_rows = []
    for role in ("validation", "held_out"):
        output = outputs[role]
        positions = {sample_id: index for index, sample_id in enumerate(output.sample_ids)}
        maneuver_ids = targets.sample_ids(role=role, task=MANEUVER_TASK)
        response_ids = targets.sample_ids(role=role, task=RESPONSE_TASK)
        maneuver_positions = np.asarray(
            [positions[sample_id] for sample_id in maneuver_ids], dtype=np.int64
        )
        response_positions = np.asarray(
            [positions[sample_id] for sample_id in response_ids], dtype=np.int64
        )
        maneuver_truth = targets.maneuver_classes(maneuver_ids)
        response_truth = targets.response_values(response_ids)
        high_truth = targets.high_response_classes(response_ids)
        for consumer_name, consumer in consumers.items():
            values = (
                output.pooled_embedding.detach().cpu().numpy()
                if consumer_name == "linear"
                else output.sequence_embedding.detach().cpu().numpy()
            )
            prediction = consumer.predict(values)
            maneuver_metrics = classification_metrics(
                maneuver_truth,
                prediction["maneuver_prediction"][maneuver_positions],
                prediction["maneuver_probability"][maneuver_positions],
                prediction["maneuver_classes"],
            )
            response_metrics = regression_metrics(
                response_truth,
                prediction["response_prediction"][response_positions],
            )
            high_prediction = prediction["high_response_prediction"][response_positions]
            high_metrics = classification_metrics(
                high_truth,
                high_prediction,
                prediction["high_response_probability"][response_positions],
                prediction["high_response_classes"],
            )
            high_metrics["high_response_recall"] = (
                recall_score(high_truth, high_prediction, pos_label=1, zero_division=0),
                "higher",
            )
            _append_metric_rows(
                metric_rows,
                maneuver_metrics,
                method=method_name,
                task=MANEUVER_TASK,
                consumer=consumer_name,
                seed=seed,
                fold=fold_id,
                role=role,
                sample_count=len(maneuver_ids),
            )
            _append_metric_rows(
                metric_rows,
                response_metrics,
                method=method_name,
                task="physiology_response_regression",
                consumer=consumer_name,
                seed=seed,
                fold=fold_id,
                role=role,
                sample_count=len(response_ids),
            )
            _append_metric_rows(
                metric_rows,
                high_metrics,
                method=method_name,
                task="high_physiology_response_classification",
                consumer=consumer_name,
                seed=seed,
                fold=fold_id,
                role=role,
                sample_count=len(response_ids),
            )
            prediction_rows.extend(
                _prediction_rows(
                    method=method_name,
                    consumer=consumer_name,
                    fold=fold_id,
                    role=role,
                    sample_ids=maneuver_ids,
                    task=MANEUVER_TASK,
                    truth=maneuver_truth,
                    prediction=prediction["maneuver_prediction"][maneuver_positions],
                )
            )
            prediction_rows.extend(
                _prediction_rows(
                    method=method_name,
                    consumer=consumer_name,
                    fold=fold_id,
                    role=role,
                    sample_ids=response_ids,
                    task="physiology_response_regression",
                    truth=response_truth,
                    prediction=prediction["response_prediction"][response_positions],
                )
            )
            prediction_rows.extend(
                _prediction_rows(
                    method=method_name,
                    consumer=consumer_name,
                    fold=fold_id,
                    role=role,
                    sample_ids=response_ids,
                    task="high_physiology_response_classification",
                    truth=high_truth,
                    prediction=high_prediction,
                )
            )
    return metric_rows, prediction_rows


def _append_metric_rows(
    rows, metrics, *, method, task, consumer, seed, fold, role, sample_count
):
    for metric, (value, direction) in metrics.items():
        available = value is not None and np.isfinite(value)
        rows.append(
            {
                "dataset": "dingxin_existing_dual_stream",
                "task": task,
                "consumer": consumer,
                "method": method,
                "seed": seed,
                "fold": fold,
                "role": role,
                "metric": metric,
                "value": float(value) if available else None,
                "direction": direction,
                "status": "available" if available else "unavailable",
                "reason": None if available else "metric_not_defined",
                "sample_count": sample_count,
                "threshold_scope": "outer_train_smoke_only",
                "smoke_only": True,
            }
        )


def _prediction_rows(
    *, method, consumer, fold, role, sample_ids, task, truth, prediction
):
    return [
        {
            "method": method,
            "consumer": consumer,
            "fold": fold,
            "role": role,
            "sample_id": sample_id,
            "task": task,
            "truth": float(truth[index]),
            "prediction": float(prediction[index]),
            "threshold_scope": "outer_train_smoke_only",
            "smoke_only": True,
        }
        for index, sample_id in enumerate(sample_ids)
    ]


def _protocol_hash(*, method_name, fold_id, outputs, targets, config):
    digest = hashlib.sha256()
    for role in ("train", "validation", "held_out"):
        output = outputs[role]
        digest.update(role.encode())
        digest.update("\n".join(output.sample_ids).encode())
        digest.update(output.checkpoint_sha256.encode())
        digest.update(output.sequence_embedding.detach().cpu().numpy().tobytes())
    payload = {
        "format": "chronaris.dingxin_window_consumer_protocol.v1",
        "method_name": method_name,
        "fold_id": fold_id,
        "config": asdict(config),
        "target_source_sha256": targets.target_source_sha256,
        "threshold_scope": targets.threshold_scope,
        "representation_digest": digest.hexdigest(),
        "fit_role": "train",
        "evaluation_roles": ["validation", "held_out"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_existing_components(*, manifest_path, protocol_sha256, resume):
    if not resume or not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != protocol_sha256:
        return {}
    result = {}
    for name, item in manifest["consumer_files"].items():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            continue
        payload = joblib.load(path)
        if payload.get("protocol_sha256") == protocol_sha256:
            result[name] = payload["consumer"]
    return result


def _atomic_joblib_dump(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
