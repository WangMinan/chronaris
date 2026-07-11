"""Locked auxiliary end-to-end fine-tuning on model-independent simulation."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_audit import (
    build_paired_unit_statistic_rows,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    build_guarded_application_consumer_targets,
)
from chronaris.evaluation.application_tasks.application_consumers import (
    duration_constrained_viterbi_decode,
    fit_duration_viterbi_parameters,
)
from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel,
    EndToEndFineTuningConfig,
    train_end_to_end_application_method,
)
from chronaris.evaluation.application_tasks.application_finetuning_export import (
    export_finetuned_application_representations,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    classification_metrics,
    compute_fusion_gain_rows,
    regression_metrics,
    segmentation_metrics,
)
from chronaris.evaluation.application_tasks.simulation_locked_context_data import (
    load_simulation_locked_context_data,
)
from chronaris.evaluation.application_tasks.simulation_finetuning_reporting import (
    write_simulation_finetuning_outputs,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.evaluation.application_tasks.simulation_locked_representation_run import (
    require_complete_locked_checkpoint_set,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.fusion_encoders import load_naive_time_sync_checkpoint
from chronaris.modeling.training import (
    TRAINABLE_FUSION_METHODS,
    load_common_pretraining_checkpoint,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_end_to_end_finetuning")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationFineTuningConfig:
    run_id: str = "2026-07-12_simulation-end-to-end-finetuning"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_simulation-locked-pretraining"
    representation_run_id: str = "2026-07-12_simulation-locked-representations"
    frozen_consumer_run_id: str = "2026-07-12_simulation-locked-consumers"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    learning_rate: float = 1e-4
    max_epochs: int = 20
    patience: int = 5
    batch_size: int = 128
    baseline_device: str = "auto"
    chronaris_device: str = "cpu"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationFineTuningResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_seed_count: int
    metric_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_end_to_end_finetuning(config: SimulationFineTuningConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_root = Path(config.heavy_output_root) / config.pretraining_run_id
    representation_root = Path(config.heavy_output_root) / config.representation_run_id
    _require_completed_upstream(config)
    selected = json.loads(Path(config.selected_candidates_path).read_text(encoding="utf-8"))
    selected_ids = {
        method: str(selected[method]["candidate_id"])
        for method in TRAINABLE_FUSION_METHODS
    }
    checkpoints = require_complete_locked_checkpoint_set(
        pretraining_root,
        seeds=config.seeds,
        selected_ids=selected_ids,
    )
    data = load_simulation_locked_context_data(config.simulation_root)
    targets = build_guarded_application_consumer_targets(
        data,
        completed_pretraining_checkpoints=tuple(
            checkpoints[(config.seeds[0], method)] for method in TRAINABLE_FUSION_METHODS
        ),
        smoke_only=False,
    )
    baseline_device = _resolve_device(config.baseline_device)
    chronaris_device = _resolve_device(config.chronaris_device)
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_end_to_end_finetuning",
        logger=LOGGER,
        initial_progress={
            "representation_family": "end_to_end_finetuned_v1",
            "seeds": list(config.seeds),
            "methods": list(APPLICATION_METHODS),
            "task_oracle_opened_after_pretraining": True,
            "held_out_used_for_training_or_selection": False,
        },
    ) as progress:
        result_rows = []
        training_rows = []
        export_rows = []
        metric_rows = []
        unit_rows = []
        for seed in config.seeds:
            seed_outputs = {}
            seed_models = {}
            for method in APPLICATION_METHODS:
                device = chronaris_device if method == "chronaris" else baseline_device
                source_path, model = _load_source_model(
                    seed=seed,
                    method=method,
                    checkpoints=checkpoints,
                    representation_root=representation_root,
                )
                training = train_end_to_end_application_method(
                    model=model,
                    batch=data.batch,
                    targets=targets,
                    role_sample_ids=data.role_sample_ids,
                    source_checkpoint_path=source_path,
                    output_root=heavy_root / "checkpoints" / f"seed_{seed}",
                    config=EndToEndFineTuningConfig(
                        learning_rate=config.learning_rate,
                        max_epochs=config.max_epochs,
                        patience=config.patience,
                        batch_size=config.batch_size,
                        seed=seed,
                        device=device,
                    ),
                    resume=config.resume,
                )
                outputs = export_finetuned_application_representations(
                    model=model,
                    checkpoint_path=training.best_checkpoint_path,
                    batch=data.batch,
                    role_sample_ids=data.role_sample_ids,
                    output_root=heavy_root / "representations" / f"seed_{seed}",
                    batch_size=config.batch_size,
                )
                completed_payload = torch.load(
                    training.best_checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model._finetune_regression_mean = float(
                    completed_payload["regression_train_mean"]
                )
                model._finetune_regression_std = float(
                    completed_payload["regression_train_std"]
                )
                seed_outputs[method] = outputs
                seed_models[method] = model
                result_rows.append(
                    {
                        "seed": seed,
                        "method_name": method,
                        "status": training.status,
                        "best_epoch": training.best_epoch,
                        "completed_epochs": training.completed_epochs,
                        "stopped_early": training.stopped_early,
                        "training_elapsed_s": training.training_elapsed_s,
                        "training_device": device,
                        "encoder_update_mode": training.encoder_update_mode,
                        "source_checkpoint_path": str(source_path),
                        "source_checkpoint_sha256": sha256_file(source_path),
                        "checkpoint_path": training.best_checkpoint_path,
                        "checkpoint_sha256": sha256_file(training.best_checkpoint_path),
                        "representation_family": "end_to_end_finetuned_v1",
                        "label_used_for_encoder_training": True,
                    }
                )
                training_rows.extend(
                    {"seed": seed, "method_name": method, **dict(row)}
                    for row in training.epoch_rows
                )
                for role, output in outputs.items():
                    manifest = (
                        heavy_root
                        / "representations"
                        / f"seed_{seed}"
                        / method
                        / role
                        / "representation_manifest.json"
                    )
                    export_rows.append(
                        {
                            "seed": seed,
                            "method_name": method,
                            "role": role,
                            "sample_count": len(output.sample_ids),
                            "manifest_path": str(manifest),
                            "label_used_for_encoder_training": json.loads(
                                manifest.read_text(encoding="utf-8")
                            )["label_used_for_encoder_training"],
                        }
                    )
                progress.update(
                    "fine_tuned_method_complete",
                    seed=seed,
                    method_name=method,
                    status=training.status,
                    best_epoch=training.best_epoch,
                )
            metrics, units = _evaluate_seed(
                seed=seed,
                models=seed_models,
                outputs=seed_outputs,
                targets=targets,
            )
            metric_rows.extend(metrics)
            unit_rows.extend(units)
        gain_rows = compute_fusion_gain_rows(
            metric_rows,
            fusion_methods=("naive_time_sync", "mult", "contiformer", "chronaris"),
        )
        paired_rows = []
        for seed in config.seeds:
            paired_rows.extend(
                {"seed": seed, **row}
                for row in build_paired_unit_statistic_rows(
                    [row for row in unit_rows if row["seed"] == seed],
                    sample_manifest_rows=data.sample_manifest_rows,
                    seed=seed,
                    smoke_only=False,
                )
            )
        acceptance = _acceptance_rows(config, result_rows, export_rows, metric_rows, paired_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = write_simulation_finetuning_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            baseline_device=baseline_device,
            chronaris_device=chronaris_device,
            result_rows=result_rows,
            training_rows=training_rows,
            export_rows=export_rows,
            metric_rows=metric_rows,
            gain_rows=gain_rows,
            paired_rows=paired_rows,
            unit_rows=unit_rows,
            target_manifest=targets.manifest,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            method_seed_count=len(result_rows),
            metric_count=len(metric_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationFineTuningResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        method_seed_count=len(result_rows),
        metric_count=len(metric_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _load_source_model(*, seed, method, checkpoints, representation_root):
    if method == "naive_time_sync":
        path = representation_root / "checkpoints" / f"seed_{seed}" / method / "best.pt"
        encoder = load_naive_time_sync_checkpoint(path)
        return path, EndToEndApplicationModel(
            method_name=method,
            encoder=None,
            normalizer=None,
            naive_encoder=encoder,
        )
    path = checkpoints[(seed, method)]
    encoder, _heads, normalizer, payload = load_common_pretraining_checkpoint(path)
    if bool(payload.get("label_used_for_encoder_training")):
        raise ValueError("fine-tuning source checkpoint already used task labels")
    return path, EndToEndApplicationModel(
        method_name=method,
        encoder=encoder,
        normalizer=normalizer,
        naive_encoder=None,
    )


def _evaluate_seed(*, seed, models, outputs, targets):
    index = {sample_id: position for position, sample_id in enumerate(targets.sample_ids)}
    first_outputs = next(iter(outputs.values()))
    train_positions = [index[value] for value in first_outputs["train"].sample_ids]
    duration = fit_duration_viterbi_parameters(targets.maneuver_state[train_positions])
    metrics = []
    units = []
    for method, model in models.items():
        model.eval()
        for role in ("validation", "held_out"):
            output = outputs[method][role]
            positions = [index[value] for value in output.sample_ids]
            with torch.inference_mode():
                # The exported sequence is exactly the fine-tuned encoder output; heads stay common.
                sequence = output.sequence_embedding.to(next(model.parameters()).device)
                pooled = sequence.mean(dim=1)
                class_logits = model.workload_classifier(pooled).cpu()
                regression_z = model.workload_regressor(pooled).squeeze(-1).cpu()
                state_logits = model.segmentation_head(sequence).cpu()
            regression_mean, regression_std = _regression_scaling(model)
            class_probability = torch.softmax(class_logits, dim=-1).numpy()
            class_prediction = class_logits.argmax(dim=-1).numpy()
            regression_prediction = (regression_z * regression_std + regression_mean).numpy()
            state_raw = state_logits.argmax(dim=-1)
            state_duration = duration_constrained_viterbi_decode(state_logits, duration)
            truth_class = targets.workload_class[positions].numpy()
            truth_regression = targets.future_workload_mean[positions].numpy()
            truth_state = targets.maneuver_state[positions].numpy()
            for task, consumer, values in (
                ("simulated_future_workload_classification", "end_to_end_linear", classification_metrics(truth_class, class_prediction, class_probability, np.arange(3))),
                ("simulated_future_workload_regression", "end_to_end_linear", regression_metrics(truth_regression, regression_prediction)),
                ("simulated_maneuver_state_segmentation", "end_to_end_tcn_raw", segmentation_metrics(truth_state, state_raw.numpy())),
                ("simulated_maneuver_state_segmentation", "end_to_end_tcn_duration", segmentation_metrics(truth_state, state_duration.numpy())),
            ):
                for metric, (value, direction) in values.items():
                    metrics.append(_metric_row(seed, method, role, task, consumer, metric, value, direction))
            for offset, sample_id in enumerate(output.sample_ids):
                units.extend((
                    _unit_row(seed, method, role, sample_id, "end_to_end_linear", "classification_correct", float(class_prediction[offset] == truth_class[offset]), "higher"),
                    _unit_row(seed, method, role, sample_id, "end_to_end_linear", "regression_absolute_error", abs(float(regression_prediction[offset]) - float(truth_regression[offset])), "lower"),
                    _unit_row(seed, method, role, sample_id, "end_to_end_tcn_raw", "frame_accuracy", float(np.mean(state_raw[offset].numpy() == truth_state[offset])), "higher"),
                    _unit_row(seed, method, role, sample_id, "end_to_end_tcn_duration", "frame_accuracy", float(np.mean(state_duration[offset].numpy() == truth_state[offset])), "higher"),
                ))
    return metrics, units


def _regression_scaling(model):
    # Attached by the orchestrator after loading the completed fine-tuning checkpoint.
    return float(model._finetune_regression_mean), float(model._finetune_regression_std)


def _metric_row(seed, method, role, task, consumer, metric, value, direction):
    available = value is not None and np.isfinite(value)
    return {
        "dataset": "model_independent_aviation_simulation",
        "task": task,
        "consumer": consumer,
        "method": method,
        "seed": seed,
        "fold": f"simulation_g1_to_g2_end_to_end__seed_{seed}",
        "role": role,
        "metric": metric,
        "value": float(value) if available else None,
        "direction": direction,
        "status": "available" if available else "unavailable",
        "reason": None if available else "metric_not_defined",
        "smoke_only": False,
        "representation_family": "end_to_end_finetuned_v1",
    }


def _unit_row(seed, method, role, sample_id, consumer, metric, value, direction):
    return {
        "seed": seed,
        "method": method,
        "role": role,
        "sample_id": sample_id,
        "consumer": consumer,
        "metric": metric,
        "value": value,
        "direction": direction,
        "smoke_only": False,
    }


def _require_completed_upstream(config):
    for run_id in (config.pretraining_run_id, config.representation_run_id, config.frozen_consumer_run_id):
        path = Path(config.compact_output_root) / run_id / "evidence_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"end-to-end fine-tuning requires completed upstream: {run_id}")


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"} or (value == "cuda" and not torch.cuda.is_available()):
        raise ValueError("fine-tuning device is unavailable")
    return value


def _acceptance_rows(config, results, exports, metrics, paired):
    expected = len(config.seeds) * len(APPLICATION_METHODS)
    return (
        _check("six_methods_all_seeds", len(results) == expected, len(results), expected),
        _check("all_training_complete", all(row["status"] in {"completed", "resumed"} for row in results), [row["status"] for row in results], "completed_or_resumed"),
        _check("labels_declared", all(row["label_used_for_encoder_training"] for row in results), True, True),
        _check("independent_representation_family", all(row["representation_family"] == "end_to_end_finetuned_v1" for row in results), "end_to_end_finetuned_v1", "end_to_end_finetuned_v1"),
        _check("three_role_exports", len(exports) == expected * 3 and all(row["label_used_for_encoder_training"] for row in exports), len(exports), expected * 3),
        _check("held_out_metrics", bool(metrics) and any(row["role"] == "held_out" for row in metrics), len(metrics), ">0"),
        _check("paired_48_trajectories", bool(paired) and all(row["independent_unit_count"] == 48 for row in paired), len(paired), ">0 with 48 units"),
    )


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}
