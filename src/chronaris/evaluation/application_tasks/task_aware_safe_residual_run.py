"""Orchestrate matched anchors and frozen safe residual screening."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronaris.evaluation.application_tasks.core_feasibility_data import (
    fold_task_data,
    maneuver_scores_by_fold,
)
from chronaris.evaluation.application_tasks.core_feasibility_models import FoldTargets
from chronaris.evaluation.application_tasks.task_aware_safe_residual_model import (
    ResidualTrainingConfig,
    SafeAnchorPredictions,
    evaluate_anchor_predictions,
    fit_safe_anchor_predictions,
    gate_is_non_degenerate,
    summarize_fusion_batch,
    train_frozen_safe_residual,
)
from chronaris.evaluation.application_tasks.task_aware_safe_residual_reporting import (
    write_safe_residual_figures,
    write_safe_residual_report,
)
from chronaris.evaluation.application_tasks.task_aware_run_utils import (
    compatible_frozen_resume as _compatible_frozen_resume,
    preserved_elapsed as _preserved_elapsed,
    resolved_device as _resolved_device,
    stable_hash as _stable_hash,
    write_json as _write_json,
)
from chronaris.representation import load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


MATCHED_METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)


@dataclass(frozen=True, slots=True)
class SafeResidualCandidate:
    candidate_id: str
    maneuver_anchor: str
    response_anchor: str
    gate_mode: str = "scalar"
    learning_rate: float = 1e-2
    gate_penalty: float = 1e-2


CANDIDATES = (
    SafeResidualCandidate(
        "observed_vehicle_scalar",
        maneuver_anchor="naive_time_sync",
        response_anchor="vehicle_only",
    ),
    SafeResidualCandidate(
        "vehicle_vehicle_scalar",
        maneuver_anchor="vehicle_only",
        response_anchor="vehicle_only",
    ),
    SafeResidualCandidate(
        "observed_physiology_scalar",
        maneuver_anchor="naive_time_sync",
        response_anchor="physiology_only",
    ),
    SafeResidualCandidate(
        "vehicle_observed_scalar",
        maneuver_anchor="vehicle_only",
        response_anchor="naive_time_sync",
    ),
    SafeResidualCandidate(
        "observed_vehicle_channel",
        maneuver_anchor="naive_time_sync",
        response_anchor="vehicle_only",
        gate_mode="channel",
    ),
)


@dataclass(frozen=True, slots=True)
class DingxinSafeResidualConfig:
    run_id: str = "2026-07-15_dingxin-task-aware-safe-residual"
    source_workspace_root: str = "/home/wangminan/projects/chronaris"
    stability_workspace_root: str = (
        "/home/wangminan/projects/chronaris-dingxin-task-stability-20260714"
    )
    matched_workspace_root: str = (
        "/home/wangminan/projects/chronaris-dingxin-target-reconstruction-20260715"
    )
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    seed: int = 17
    max_epochs: int = 80
    patience: int = 12
    partial_max_epochs: int = 24
    partial_patience: int = 6
    device: str = "cuda"
    resume: bool = True

    def __post_init__(self) -> None:
        if min(
            self.max_epochs,
            self.patience,
            self.partial_max_epochs,
            self.partial_patience,
        ) <= 0:
            raise ValueError("safe residual epoch configuration must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("safe residual device is invalid")


@dataclass(frozen=True, slots=True)
class DingxinSafeResidualResult:
    compact_root: str
    heavy_root: str
    selected_candidate_id: str | None
    frozen_safety_passed: bool
    allow_partial_unfreeze: bool
    partial_safety_passed: bool
    research_gate_passed: bool
    allow_teacher_distillation: bool
    outer_test_opened: bool


def run_dingxin_task_aware_safe_residual(
    config: DingxinSafeResidualConfig,
) -> DingxinSafeResidualResult:
    """Run the pre-registered frozen safe-residual phase."""

    started = time.perf_counter()
    compact = Path(config.compact_output_root) / config.run_id
    heavy = Path(config.heavy_output_root) / config.run_id
    compact.mkdir(parents=True, exist_ok=True)
    heavy.mkdir(parents=True, exist_ok=True)
    source = Path(config.source_workspace_root)
    stability = (
        Path(config.stability_workspace_root)
        / "artifacts/application_evaluation/2026-07-14_dingxin-task-stability"
        / "development_splits"
    )
    matched = (
        Path(config.matched_workspace_root)
        / "artifacts/application_evaluation/2026-07-15_dingxin-target-reconstruction-confirmation"
    )
    input_paths = _input_paths(source, stability, matched)
    _validate_inputs(input_paths)
    manifest = json.loads(input_paths["split_manifest"].read_text(encoding="utf-8"))
    plans = {
        str(row["fold_id"]): row
        for row in manifest["folds"]
        if bool(row.get("main_selection"))
    }
    if len(plans) != 6:
        raise ValueError("stage 2 requires exactly six unique main-selection splits")
    if len({row["validation_support_hash"] for row in plans.values()}) != 6:
        raise ValueError("stage 2 validation supports are not unique")
    if any(int(row["support_overlap_count"]) != 0 for row in plans.values()):
        raise PermissionError("stage 2 split support overlap must be zero")
    target_frame = pd.read_csv(input_paths["nested_targets"])
    maneuver_scores = maneuver_scores_by_fold(
        plans=plans,
        fixed_root=source / "docs/artifacts/runs/2026-07-10_fixed-data-audit",
        e_run_manifest_path=input_paths["e_manifest"],
        f_run_manifest_path=input_paths["f_manifest"],
    )
    protocol = _protocol(config, input_paths, plans)
    protocol_hash = _stable_hash(protocol)
    protocol["protocol_sha256"] = protocol_hash
    _write_json(compact / "protocol.json", protocol)
    _write_json(
        compact / "architecture_manifest.json",
        _architecture_manifest(),
    )
    _write_json(
        compact / "candidate_grid.json",
        {
            "candidate_count": len(CANDIDATES),
            "maximum_candidate_count": 12,
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "outer_test_opened": False,
        },
    )

    matched_rows: list[dict[str, object]] = []
    frozen_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    per_split_payload: dict[str, dict[str, object]] = {}
    for split_index, (split_id, plan) in enumerate(sorted(plans.items())):
        task_data = fold_task_data(
            fold_id=split_id,
            train_ids=tuple(str(value) for value in plan["train_sample_ids"]),
            validation_ids=tuple(
                str(value) for value in plan["validation_sample_ids"]
            ),
            target_frame=target_frame,
            maneuver_scores=maneuver_scores[split_id],
        )
        targets = _combined_targets(task_data)
        features = _load_split_features(
            matched_root=matched,
            seed=config.seed,
            split_id=split_id,
            task_data=task_data,
        )
        method_anchors: dict[str, SafeAnchorPredictions] = {}
        for method_index, method in enumerate(MATCHED_METHODS):
            anchor = _fit_anchor(
                features=features,
                task_data=task_data,
                targets=targets,
                maneuver_method=method,
                response_method=method,
                seed=config.seed + split_index * 100 + method_index,
            )
            method_anchors[method] = anchor
            metrics = evaluate_anchor_predictions(anchor, targets)
            matched_rows.extend(
                _metric_rows(
                    split_id=split_id,
                    candidate_id=method,
                    variant="matched_task_anchor",
                    metrics=metrics,
                    direct_metrics=None,
                )
            )
        per_split_payload[split_id] = {
            "task_data": task_data,
            "targets": targets,
            "features": features,
            "method_anchors": method_anchors,
        }
        for candidate_index, candidate in enumerate(CANDIDATES):
            state_root = heavy / "frozen" / candidate.candidate_id / split_id
            state_path = state_root / "state.json"
            checkpoint_path = state_root / "best.pt"
            unit_hash = _stable_hash(
                {
                    "protocol_sha256": protocol_hash,
                    "candidate": asdict(candidate),
                    "split_id": split_id,
                }
            )
            if config.resume and state_path.is_file() and checkpoint_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("unit_sha256") != unit_hash and not _compatible_frozen_resume(
                    checkpoint_path,
                    candidate=candidate,
                    split_id=split_id,
                ):
                    raise ValueError("safe residual resume unit changed")
                frozen_rows.extend(state["metric_rows"])
                training_rows.extend(state["training_rows"])
                gate_rows.extend(state["gate_rows"])
                continue
            anchors = _fit_anchor(
                features=features,
                task_data=task_data,
                targets=targets,
                maneuver_method=candidate.maneuver_anchor,
                response_method=candidate.response_anchor,
                seed=config.seed + split_index * 100 + candidate_index,
            )
            chronaris = features["chronaris"]
            result = train_frozen_safe_residual(
                chronaris_train_features={
                    "all": chronaris["train_all"],
                    "maneuver": chronaris["train_maneuver"],
                    "response": chronaris["train_response"],
                },
                chronaris_validation_features={
                    "maneuver": chronaris["validation_maneuver"],
                    "response": chronaris["validation_response"],
                },
                anchors=anchors,
                targets=targets,
                config=ResidualTrainingConfig(
                    learning_rate=candidate.learning_rate,
                    max_epochs=config.max_epochs,
                    patience=config.patience,
                    gate_mode=candidate.gate_mode,
                    gate_penalty=candidate.gate_penalty,
                    seed=config.seed + split_index * 100 + candidate_index,
                    device=_resolved_device(config.device),
                ),
            )
            metric_rows = []
            metric_rows.extend(
                _metric_rows(
                    split_id=split_id,
                    candidate_id=candidate.candidate_id,
                    variant="direct_only",
                    metrics=result.direct_metrics,
                    direct_metrics=None,
                )
            )
            metric_rows.extend(
                _metric_rows(
                    split_id=split_id,
                    candidate_id=candidate.candidate_id,
                    variant="full",
                    metrics=result.full_metrics,
                    direct_metrics=result.direct_metrics,
                )
            )
            unit_training_rows = [
                {
                    "split_id": split_id,
                    "candidate_id": candidate.candidate_id,
                    "training_stage": "frozen_safe_residual",
                    "best_epoch": result.best_epoch,
                    "epoch_count": len(result.epoch_rows),
                    "safety_passed": result.safety_passed,
                    "checkpoint_path": str(checkpoint_path),
                    "outer_test_opened": False,
                    **row,
                }
                for row in result.epoch_rows
            ]
            unit_gate_rows = [
                {
                    "split_id": split_id,
                    "candidate_id": candidate.candidate_id,
                    "task": task,
                    "gate_index": index,
                    "gate_value": value,
                    "non_degenerate": gate_is_non_degenerate(result.gate_values),
                    "outer_test_opened": False,
                }
                for task, values in result.gate_values.items()
                for index, value in enumerate(values)
            ]
            state_root.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": "chronaris.dingxin_safe_residual_checkpoint.v1",
                    "unit_sha256": unit_hash,
                    "candidate": asdict(candidate),
                    "split_id": split_id,
                    "model_state_dict": dict(result.model_state_dict),
                    "feature_mean": result.feature_mean,
                    "feature_scale": result.feature_scale,
                    "best_epoch": result.best_epoch,
                    "task_targets_opened": True,
                    "outer_test_opened": False,
                },
                checkpoint_path,
            )
            _write_json(
                state_path,
                {
                    "unit_sha256": unit_hash,
                    "metric_rows": metric_rows,
                    "training_rows": unit_training_rows,
                    "gate_rows": unit_gate_rows,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                },
            )
            frozen_rows.extend(metric_rows)
            training_rows.extend(unit_training_rows)
            gate_rows.extend(unit_gate_rows)
    matched_frame = pd.DataFrame(matched_rows)
    frozen_frame = pd.DataFrame(frozen_rows)
    training_frame = pd.DataFrame(training_rows)
    gate_frame = pd.DataFrame(gate_rows)
    aggregate = _aggregate_candidates(frozen_frame, gate_frame)
    selected = _select_candidate(aggregate)
    selected_id = None if selected is None else str(selected["candidate_id"])
    frozen_passed = selected is not None
    partial_result = None
    if frozen_passed:
        from chronaris.evaluation.application_tasks.task_aware_partial_run import (
            run_partial_unfreeze_screen,
        )

        frozen_candidate_index = next(
            index
            for index, candidate in enumerate(CANDIDATES)
            if candidate.candidate_id == selected_id
        )
        partial_result = run_partial_unfreeze_screen(
            compact_root=compact,
            heavy_root=heavy,
            source_root=source,
            matched_root=matched,
            seed=config.seed,
            device=_resolved_device(config.device),
            max_epochs=config.partial_max_epochs,
            patience=config.partial_patience,
            resume=config.resume,
            selected_frozen_candidate=asdict(CANDIDATES[frozen_candidate_index]),
            selected_frozen_candidate_index=frozen_candidate_index,
            split_payloads=per_split_payload,
        )
    allowance = {
        "decision": (
            "partial_research_gate_passed"
            if partial_result is not None and partial_result.research_gate_passed
            else "partial_research_gap"
            if partial_result is not None
            else "gap"
        ),
        "selected_candidate_id": selected_id,
        "selected_partial_candidate_id": (
            None if partial_result is None else partial_result.selected_candidate_id
        ),
        "frozen_safety_passed": frozen_passed,
        "allow_partial_unfreeze": frozen_passed,
        "partial_safety_passed": bool(
            partial_result is not None and partial_result.partial_safety_passed
        ),
        "research_gate_passed": bool(
            partial_result is not None and partial_result.research_gate_passed
        ),
        "allow_teacher_distillation": bool(
            partial_result is not None
            and partial_result.allow_teacher_distillation
        ),
        "configuration_locked": False,
        "task_targets_opened": True,
        "outer_test_opened": False,
        "historical_outer_metrics_used_for_selection": False,
        "gate_contract": {
            "maneuver_maximum_drop": 0.005,
            "response_maximum_relative_rmse_increase": 0.01,
            "high_response_maximum_auprc_drop": 0.005,
            "minimum_safe_support_count": 4,
        },
    }
    matched_frame.to_csv(compact / "matched_baseline_metrics.csv", index=False)
    frozen_frame.to_csv(compact / "frozen_safe_residual_metrics.csv", index=False)
    training_frame.to_csv(compact / "candidate_training.csv", index=False)
    gate_frame.to_csv(compact / "gate_statistics.csv", index=False)
    pd.DataFrame(aggregate).to_csv(compact / "candidate_inner_metrics.csv", index=False)
    _write_json(
        compact / "selected_candidates.json",
        {
            "selected": selected,
            "partial_unfreeze_candidates": (
                [
                    {"backbone_learning_rate_ratio": 0.02},
                    {"backbone_learning_rate_ratio": 0.05},
                ]
                if frozen_passed
                else []
            ),
            "outer_test_opened": False,
        },
    )
    _write_json(compact / "allowance.json", allowance)
    _write_json(
        compact / "access_audit.json",
        {
            "outer_observation_request_count": 0,
            "outer_label_request_count": 0,
            "outer_prediction_request_count": 0,
            "outer_metric_request_count": 0,
            "outer_test_opened": False,
        },
    )
    _write_json(
        compact / "progress.json",
        {
            "status": "task_aware_safe_residual_complete",
            "completed_split_count": len(plans),
            "completed_candidate_split_count": len(CANDIDATES) * len(plans),
            "matched_method_split_count": len(MATCHED_METHODS) * len(plans),
            "elapsed_s": _preserved_elapsed(
                compact / "progress.json", time.perf_counter() - started
            ),
            **allowance,
        },
    )
    (compact / "resume_command.txt").write_text(
        "PYTHONPATH=src python "
        "scripts/evaluation/application_tasks/run_dingxin_task_aware_safe_residual.py "
        "--resume\n",
        encoding="utf-8",
    )
    report_name = (
            "acceptance_report.md"
            if allowance["research_gate_passed"]
            else "gap_report.md"
    )
    obsolete_name = (
        "gap_report.md" if report_name == "acceptance_report.md" else "acceptance_report.md"
    )
    (compact / obsolete_name).unlink(missing_ok=True)
    write_safe_residual_report(
        compact / report_name,
        allowance=allowance,
        selected=selected,
        aggregate=aggregate,
        partial_aggregate=(
            pd.read_csv(compact / "partial_candidate_metrics.csv").to_dict(
                orient="records"
            )
            if (compact / "partial_candidate_metrics.csv").is_file()
            else ()
        ),
    )
    write_safe_residual_figures(compact)
    return DingxinSafeResidualResult(
        compact_root=str(compact),
        heavy_root=str(heavy),
        selected_candidate_id=selected_id,
        frozen_safety_passed=frozen_passed,
        allow_partial_unfreeze=frozen_passed,
        partial_safety_passed=bool(allowance["partial_safety_passed"]),
        research_gate_passed=bool(allowance["research_gate_passed"]),
        allow_teacher_distillation=bool(allowance["allow_teacher_distillation"]),
        outer_test_opened=False,
    )


def _input_paths(source: Path, stability: Path, matched: Path):
    return {
        "split_manifest": stability / "split_manifest.json",
        "nested_targets": stability / "nested_targets.csv",
        "matched_protocol": matched.parent.parent.parent
        / "docs/artifacts/runs/2026-07-15_dingxin-target-reconstruction-confirmation/protocol.json",
        "representations": matched / "representations",
        "checkpoints": matched / "checkpoints",
        "e_manifest": source
        / "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json",
        "f_manifest": source
        / "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json",
        "fixed_manifest": source
        / "docs/artifacts/runs/2026-07-10_fixed-data-audit/data_manifest.json",
    }


def _validate_inputs(paths):
    for name, path in paths.items():
        if name in {"representations", "checkpoints"}:
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)


def _load_split_features(*, matched_root, seed, split_id, task_data):
    output = {}
    maneuver_train_ids = task_data["maneuver"]["train_ids"]
    maneuver_validation_ids = task_data["maneuver"]["validation_ids"]
    response_train_ids = task_data["response"]["train_ids"]
    response_validation_ids = task_data["response"]["validation_ids"]
    for method in MATCHED_METHODS:
        root = matched_root / "representations" / f"seed_{seed}" / split_id / method / split_id
        train = load_fusion_stream_batch(root / "train")
        validation = load_fusion_stream_batch(root / "validation")
        output[method] = {
            "train_all": summarize_fusion_batch(train, train.sample_ids),
            "train_maneuver": summarize_fusion_batch(train, maneuver_train_ids),
            "validation_maneuver": summarize_fusion_batch(
                validation, maneuver_validation_ids
            ),
            "train_response": summarize_fusion_batch(train, response_train_ids),
            "validation_response": summarize_fusion_batch(
                validation, response_validation_ids
            ),
        }
    return output


def _fit_anchor(
    *, features, task_data, targets, maneuver_method, response_method, seed
):
    return fit_safe_anchor_predictions(
        maneuver_train_features=features[maneuver_method]["train_maneuver"],
        maneuver_validation_features=features[maneuver_method][
            "validation_maneuver"
        ],
        response_train_features=features[response_method]["train_response"],
        response_validation_features=features[response_method][
            "validation_response"
        ],
        targets=targets,
        seed=seed,
    )


def _combined_targets(task_data):
    maneuver = task_data["maneuver"]["targets"]
    response = task_data["response"]["targets"]
    return FoldTargets(
        train_maneuver=maneuver.train_maneuver,
        validation_maneuver=maneuver.validation_maneuver,
        train_maneuver_score=maneuver.train_maneuver_score,
        validation_maneuver_score=maneuver.validation_maneuver_score,
        train_response=response.train_response,
        validation_response=response.validation_response,
        train_high_response=response.train_high_response,
        validation_high_response=response.validation_high_response,
    )


def _metric_rows(*, split_id, candidate_id, variant, metrics, direct_metrics):
    rows = []
    specifications = {
        "maneuver": ("macro_f1", "higher"),
        "response": ("rmse", "lower"),
        "high_response": ("auprc", "higher"),
    }
    for task, values in metrics.items():
        primary, direction = specifications[task]
        for metric, value in values.items():
            direct_value = (
                None if direct_metrics is None else direct_metrics[task].get(metric)
            )
            delta = None
            if direct_value is not None:
                delta = (
                    float(value - direct_value)
                    if direction == "higher" or metric != primary
                    else float(direct_value - value)
                )
            rows.append(
                {
                    "split_id": split_id,
                    "candidate_id": candidate_id,
                    "variant": variant,
                    "task": task,
                    "metric": metric,
                    "direction": direction if metric == primary else "diagnostic",
                    "value": float(value),
                    "direct_value": direct_value,
                    "direction_normalized_delta": delta,
                    "outer_test_opened": False,
                }
            )
    return rows


def _aggregate_candidates(frame: pd.DataFrame, gate_frame: pd.DataFrame):
    rows = []
    full = frame[frame["variant"] == "full"]
    direct = frame[frame["variant"] == "direct_only"]
    for candidate in CANDIDATES:
        selected = full[full["candidate_id"] == candidate.candidate_id]
        baseline = direct[direct["candidate_id"] == candidate.candidate_id]
        maneuver = selected[
            (selected["task"] == "maneuver") & (selected["metric"] == "macro_f1")
        ]
        response = selected[
            (selected["task"] == "response") & (selected["metric"] == "rmse")
        ]
        response_ratio = selected[
            (selected["task"] == "response") & (selected["metric"] == "rmse_ratio")
        ]
        response_skill = selected[
            (selected["task"] == "response") & (selected["metric"] == "response_skill")
        ]
        high = selected[
            (selected["task"] == "high_response") & (selected["metric"] == "auprc")
        ]
        high_normalized = selected[
            (selected["task"] == "high_response")
            & (selected["metric"] == "normalized_ap")
        ]
        direct_maneuver = baseline[
            (baseline["task"] == "maneuver") & (baseline["metric"] == "macro_f1")
        ]
        direct_response = baseline[
            (baseline["task"] == "response") & (baseline["metric"] == "rmse")
        ]
        direct_high = baseline[
            (baseline["task"] == "high_response") & (baseline["metric"] == "auprc")
        ]
        maneuver_pair = _align_metric_values(maneuver, direct_maneuver)
        response_pair = _align_metric_values(response, direct_response)
        high_pair = _align_metric_values(high, direct_high)
        maneuver_safe = int(
            np.sum(maneuver_pair["full"] >= maneuver_pair["direct"] - 0.005)
        )
        response_safe = int(
            np.sum(response_pair["full"] <= response_pair["direct"] * 1.01)
        )
        high_safe = int(
            np.sum(high_pair["full"] >= high_pair["direct"] - 0.005)
        )
        gates = gate_frame[gate_frame["candidate_id"] == candidate.candidate_id]
        gate_ok = bool(gates["non_degenerate"].all())
        safe = bool(min(maneuver_safe, response_safe, high_safe) >= 4 and gate_ok)
        mean_f1 = float(maneuver["value"].mean())
        median_ratio = float(response_ratio["value"].median())
        mean_normalized = float(high_normalized["value"].mean())
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "maneuver_anchor": candidate.maneuver_anchor,
                "response_anchor": candidate.response_anchor,
                "gate_mode": candidate.gate_mode,
                "maneuver_mean_macro_f1": mean_f1,
                "maneuver_median_macro_f1": float(maneuver["value"].median()),
                "maneuver_worst_macro_f1": float(maneuver["value"].min()),
                "response_median_rmse_ratio": median_ratio,
                "response_mean_skill": float(response_skill["value"].mean()),
                "response_positive_skill_count": int(
                    np.sum(response_skill["value"].to_numpy() > 0)
                ),
                "high_response_mean_normalized_ap": mean_normalized,
                "high_response_median_normalized_ap": float(
                    high_normalized["value"].median()
                ),
                "high_response_positive_split_count": int(
                    np.sum(high_normalized["value"].to_numpy() > 0)
                ),
                "maneuver_safe_support_count": maneuver_safe,
                "response_safe_support_count": response_safe,
                "high_response_safe_support_count": high_safe,
                "gate_non_degenerate": gate_ok,
                "frozen_safety_passed": safe,
                "ranking_score": mean_f1 + (1.0 - median_ratio) + mean_normalized,
                "outer_test_opened": False,
            }
        )
    return rows


def _align_metric_values(full: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    aligned = full[["split_id", "value"]].merge(
        direct[["split_id", "value"]],
        on="split_id",
        suffixes=("_full", "_direct"),
        validate="one_to_one",
    )
    if len(aligned) != len(full) or len(aligned) != len(direct):
        raise ValueError("safe residual metric supports are not aligned")
    return aligned.rename(columns={"value_full": "full", "value_direct": "direct"})


def _select_candidate(rows):
    available = [row for row in rows if row["frozen_safety_passed"]]
    if not available:
        return None
    return max(available, key=lambda row: (row["ranking_score"], row["candidate_id"]))


def _protocol(config, paths, plans):
    return {
        "format": "chronaris.dingxin_task_aware_safe_residual_protocol.v1",
        "config": asdict(config),
        "method_names": list(MATCHED_METHODS),
        "candidate_count": len(CANDIDATES),
        "split_count": len(plans),
        "validation_support_hashes": sorted(
            str(row["validation_support_hash"]) for row in plans.values()
        ),
        "input_sha256": {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.is_file()
        },
        "task_targets_opened": True,
        "outer_test_opened": False,
        "historical_outer_metrics_used_for_selection": False,
        "configuration_locked": False,
    }


def _architecture_manifest():
    return {
        "model_name": "Chronaris",
        "reader_visible_version_suffix": False,
        "safe_anchor_roles": {
            "maneuver": "vehicle_or_direct_observation_anchor",
            "response": "vehicle_or_physiology_inertia_anchor",
            "high_response": "independent_risk_anchor",
        },
        "continuous_path": "dual_stream_continuous_causal_residual",
        "fusion_equation": "task_output = safe_anchor + sigmoid(gate) * chronaris_delta",
        "gate_initialization": "near_zero_with_zero_residual_projection",
        "frozen_stage_backbone_trained": False,
        "partial_stage_backbone_scope": [
            "observation_encoders",
            "observation_updates",
            "causal_fusion_output_projection",
            "lag_scale_gate",
        ],
        "outer_test_opened": False,
    }
