"""Run Dingxin stage 3A residual activation on six inner supports."""

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
from chronaris.evaluation.application_tasks.residual_activation_contracts import (
    ActivationTrainingConfig,
    CANDIDATES,
    build_selective_teacher_targets,
)
from chronaris.evaluation.application_tasks.residual_activation_model import (
    train_activated_residual,
)
from chronaris.evaluation.application_tasks.task_aware_run_utils import (
    resolved_device,
    stable_hash,
    write_json,
)
from chronaris.evaluation.application_tasks.task_aware_safe_residual_run import (
    MATCHED_METHODS,
    _combined_targets,
    _fit_anchor,
    _input_paths,
    _load_split_features,
    _metric_rows,
    _validate_inputs,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class DingxinResidualActivationConfig:
    run_id: str = "2026-07-15_dingxin-residual-activation"
    source_workspace_root: str = "/home/wangminan/projects/chronaris"
    stability_workspace_root: str = (
        "/home/wangminan/projects/chronaris-dingxin-task-stability-20260714"
    )
    matched_workspace_root: str = (
        "/home/wangminan/projects/chronaris-dingxin-target-reconstruction-20260715"
    )
    stage2_workspace_root: str = (
        "/home/wangminan/projects/chronaris-dingxin-task-aware-safe-residual-20260715"
    )
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    seed: int = 17
    max_epochs: int = 80
    patience: int = 12
    device: str = "cuda"
    resume: bool = True

    def __post_init__(self) -> None:
        if min(self.seed, self.max_epochs, self.patience) <= 0:
            raise ValueError("stage-3A seed and epoch configuration must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("stage-3A device is invalid")


@dataclass(frozen=True, slots=True)
class DingxinResidualActivationResult:
    compact_root: str
    heavy_root: str
    selected_candidate_id: str | None
    activation_gate_passed: bool
    safety_gate_passed: bool
    research_gate_passed: bool
    allow_stage_3b: bool
    outer_test_opened: bool


def run_dingxin_residual_activation(
    config: DingxinResidualActivationConfig,
) -> DingxinResidualActivationResult:
    """Execute the bounded stage-3A screen and stop before outer evaluation."""

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
    stage2_compact = (
        Path(config.stage2_workspace_root)
        / "docs/artifacts/runs/2026-07-15_dingxin-task-aware-safe-residual"
    )
    input_paths = _input_paths(source, stability, matched)
    input_paths.update(
        {
            "stage2_allowance": stage2_compact / "allowance.json",
            "stage2_selected": stage2_compact / "selected_candidates.json",
            "stage2_metrics": stage2_compact / "candidate_inner_metrics.csv",
        }
    )
    _validate_inputs(input_paths)
    stage2_allowance = json.loads(
        input_paths["stage2_allowance"].read_text(encoding="utf-8")
    )
    stage2_selection = json.loads(
        input_paths["stage2_selected"].read_text(encoding="utf-8")
    )["selected"]
    _validate_stage2(stage2_allowance, stage2_selection)
    manifest = json.loads(input_paths["split_manifest"].read_text(encoding="utf-8"))
    plans = {
        str(row["fold_id"]): row
        for row in manifest["folds"]
        if bool(row.get("main_selection"))
    }
    _validate_plans(plans)
    target_frame = pd.read_csv(input_paths["nested_targets"])
    maneuver_scores = maneuver_scores_by_fold(
        plans=plans,
        fixed_root=source / "docs/artifacts/runs/2026-07-10_fixed-data-audit",
        e_run_manifest_path=input_paths["e_manifest"],
        f_run_manifest_path=input_paths["f_manifest"],
    )
    protocol = _protocol(config, input_paths, plans, stage2_selection)
    protocol_hash = stable_hash(protocol)
    protocol["protocol_sha256"] = protocol_hash
    write_json(compact / "protocol.json", protocol)
    write_json(compact / "candidate_grid.json", _candidate_grid())
    write_json(compact / "architecture_manifest.json", _architecture_manifest())
    write_json(compact / "teacher_inventory.json", _teacher_inventory())

    metric_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    teacher_rows: list[dict[str, object]] = []
    access_rows: list[dict[str, object]] = []
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
        method_anchors = {
            method: _fit_anchor(
                features=features,
                task_data=task_data,
                targets=targets,
                maneuver_method=method,
                response_method=method,
                seed=config.seed + split_index * 100 + method_index,
            )
            for method_index, method in enumerate(MATCHED_METHODS)
        }
        safe_anchor = _fit_anchor(
            features=features,
            task_data=task_data,
            targets=targets,
            maneuver_method=str(stage2_selection["maneuver_anchor"]),
            response_method=str(stage2_selection["response_anchor"]),
            seed=config.seed + split_index * 100 + 3,
        )
        teachers = build_selective_teacher_targets(
            method_anchors=method_anchors,
            safe_anchor=safe_anchor,
            targets=targets,
        )
        teacher_rows.append(
            {
                "split_id": split_id,
                "teacher_predictions_are_cross_fitted": True,
                "teacher_training_sample_seen_by_predictor": False,
                **teachers.summary(),
                "outer_test_opened": False,
            }
        )
        chronaris = features["chronaris"]
        for candidate_index, candidate in enumerate(CANDIDATES):
            unit_root = heavy / "screen" / candidate.candidate_id / split_id
            state_path = unit_root / "state.json"
            checkpoint_path = unit_root / "best.pt"
            unit_hash = stable_hash(
                {
                    "protocol_sha256": protocol_hash,
                    "candidate": asdict(candidate),
                    "split_id": split_id,
                    "teacher_summary": teachers.summary(),
                }
            )
            if config.resume and state_path.is_file() and checkpoint_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("unit_sha256") != unit_hash:
                    raise ValueError("stage-3A resume unit changed")
                metric_rows.extend(state["metric_rows"])
                training_rows.extend(state["training_rows"])
                activation_rows.append(state["activation_row"])
                gate_rows.extend(state["gate_rows"])
                continue
            result = train_activated_residual(
                chronaris_train_features={
                    "all": chronaris["train_all"],
                    "maneuver": chronaris["train_maneuver"],
                    "response": chronaris["train_response"],
                },
                chronaris_validation_features={
                    "maneuver": chronaris["validation_maneuver"],
                    "response": chronaris["validation_response"],
                },
                anchors=safe_anchor,
                teachers=teachers,
                targets=targets,
                candidate=candidate,
                config=ActivationTrainingConfig(
                    max_epochs=config.max_epochs,
                    patience=config.patience,
                    seed=config.seed + split_index * 100 + candidate_index,
                    device=resolved_device(config.device),
                ),
            )
            unit_metrics = _result_metric_rows(
                split_id=split_id,
                candidate_id=candidate.candidate_id,
                direct_metrics=result.direct_metrics,
                full_metrics=result.full_metrics,
            )
            unit_training = [
                {
                    "split_id": split_id,
                    "candidate_id": candidate.candidate_id,
                    "best_epoch": result.best_epoch,
                    "epoch_count": len(result.epoch_rows),
                    "safety_passed": result.safety_passed,
                    "outer_test_opened": False,
                    **row,
                }
                for row in result.epoch_rows
            ]
            activation_row = {
                "split_id": split_id,
                "candidate_id": candidate.candidate_id,
                "best_epoch": result.best_epoch,
                **{
                    f"{task}_best_epoch": epoch
                    for task, epoch in result.task_best_epochs.items()
                },
                "safety_passed": result.safety_passed,
                **result.prediction_diagnostics,
                "outer_test_opened": False,
            }
            unit_gates = [
                {
                    "split_id": split_id,
                    "candidate_id": candidate.candidate_id,
                    "task": task,
                    **summary,
                    "outer_test_opened": False,
                }
                for task, summary in result.gate_summary.items()
            ]
            unit_root.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": "chronaris.dingxin_residual_activation_checkpoint.v1",
                    "unit_sha256": unit_hash,
                    "candidate": asdict(candidate),
                    "split_id": split_id,
                    "model_state_dict": dict(result.model_state_dict),
                    "feature_mean": result.feature_mean,
                    "feature_scale": result.feature_scale,
                    "best_epoch": result.best_epoch,
                    "task_best_epochs": dict(result.task_best_epochs),
                    "safe_anchor": {
                        "maneuver_method": stage2_selection["maneuver_anchor"],
                        "response_method": stage2_selection["response_anchor"],
                        "integrated_direct_branch": True,
                    },
                    "teacher_runtime_required": False,
                    "teacher_oof_lineage_verified": True,
                    "outer_test_opened": False,
                },
                checkpoint_path,
            )
            write_json(
                state_path,
                {
                    "unit_sha256": unit_hash,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "metric_rows": unit_metrics,
                    "training_rows": unit_training,
                    "activation_row": activation_row,
                    "gate_rows": unit_gates,
                },
            )
            metric_rows.extend(unit_metrics)
            training_rows.extend(unit_training)
            activation_rows.append(activation_row)
            gate_rows.extend(unit_gates)
        access_rows.append(
            {
                "split_id": split_id,
                "outer_observation_request_count": 0,
                "outer_label_request_count": 0,
                "outer_prediction_request_count": 0,
                "outer_metric_request_count": 0,
                "outer_test_opened": False,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    training = pd.DataFrame(training_rows)
    activations = pd.DataFrame(activation_rows)
    gates = pd.DataFrame(gate_rows)
    teachers = pd.DataFrame(teacher_rows)
    access = pd.DataFrame(access_rows)
    aggregate = aggregate_activation_candidates(metrics, activations, gates)
    selected = _select_candidate(aggregate)
    ranked = sorted(
        [row for row in aggregate if row["activation_gate_passed"] and row["safety_gate_passed"]],
        key=lambda row: (row["ranking_score"], row["candidate_id"]),
        reverse=True,
    )
    top_two = [row["candidate_id"] for row in ranked[:2]] if selected else []
    allowance = {
        "decision": "stage_3a_passed" if selected else "stage_3a_gap",
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "top_two_candidate_ids": top_two,
        "activation_gate_passed": any(
            row["activation_gate_passed"] for row in aggregate
        ),
        "safety_gate_passed": any(row["safety_gate_passed"] for row in aggregate),
        "research_gate_passed": selected is not None,
        "allow_stage_3b": selected is not None,
        "configuration_locked": False,
        "outer_test_opened": False,
        "historical_outer_metrics_used_for_selection": False,
    }
    metrics.to_csv(compact / "candidate_support_metrics.csv", index=False)
    training.to_csv(compact / "candidate_training.csv", index=False)
    gradient_columns = [
        "split_id",
        "candidate_id",
        "epoch",
        "gradient_cosine_maneuver_response",
        "gradient_cosine_maneuver_high_response",
        "gradient_cosine_response_high_response",
    ]
    training[gradient_columns].to_csv(
        compact / "gradient_conflict.csv", index=False
    )
    activations.to_csv(compact / "residual_activation_diagnostics.csv", index=False)
    gates.to_csv(compact / "sample_gate_statistics.csv", index=False)
    teachers.to_csv(compact / "teacher_oof_statistics.csv", index=False)
    access.to_csv(compact / "access_audit.csv", index=False)
    pd.DataFrame(aggregate).to_csv(compact / "candidate_summary.csv", index=False)
    write_json(
        compact / "selected_candidates.json",
        {"selected": selected, "top_two_candidate_ids": top_two, **allowance},
    )
    write_json(compact / "allowance.json", allowance)
    write_json(
        compact / "progress.json",
        {
            "status": "residual_activation_complete",
            "completed_split_count": len(plans),
            "completed_candidate_split_count": len(plans) * len(CANDIDATES),
            "elapsed_s": time.perf_counter() - started,
            **allowance,
        },
    )
    (compact / "resume_command.txt").write_text(
        "PYTHONPATH=src /home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_residual_activation.py "
        f"--run-id {config.run_id} --resume\n",
        encoding="utf-8",
    )
    from chronaris.evaluation.application_tasks.residual_activation_reporting import (
        write_residual_activation_figures,
        write_residual_activation_report,
    )

    report_name = "acceptance_report.md" if selected else "gap_report.md"
    obsolete = "gap_report.md" if selected else "acceptance_report.md"
    (compact / obsolete).unlink(missing_ok=True)
    write_residual_activation_report(
        compact / report_name,
        allowance=allowance,
        aggregate=aggregate,
        selected=selected,
    )
    write_residual_activation_figures(compact)
    _write_evidence_manifest(compact, report_name)
    return DingxinResidualActivationResult(
        compact_root=str(compact),
        heavy_root=str(heavy),
        selected_candidate_id=None if selected is None else str(selected["candidate_id"]),
        activation_gate_passed=bool(allowance["activation_gate_passed"]),
        safety_gate_passed=bool(allowance["safety_gate_passed"]),
        research_gate_passed=bool(allowance["research_gate_passed"]),
        allow_stage_3b=bool(allowance["allow_stage_3b"]),
        outer_test_opened=False,
    )


def aggregate_activation_candidates(metrics, activations, gates):
    """Aggregate exact stage-3A activation, safety, and research gates."""

    output = []
    full = metrics[metrics["variant"] == "full"]
    direct = metrics[metrics["variant"] == "direct_only"]
    for candidate in CANDIDATES:
        candidate_id = candidate.candidate_id
        selected = full[full["candidate_id"] == candidate_id]
        baseline = direct[direct["candidate_id"] == candidate_id]

        def values(frame, task, metric):
            return frame[(frame["task"] == task) & (frame["metric"] == metric)][
                ["split_id", "value"]
            ]

        maneuver = _aligned_values(
            values(selected, "maneuver", "macro_f1"),
            values(baseline, "maneuver", "macro_f1"),
        )
        response = _aligned_values(
            values(selected, "response", "rmse"),
            values(baseline, "response", "rmse"),
        )
        high = _aligned_values(
            values(selected, "high_response", "auprc"),
            values(baseline, "high_response", "auprc"),
        )
        response_ratio = values(selected, "response", "rmse_ratio")["value"]
        response_skill = values(selected, "response", "response_skill")["value"]
        normalized_ap = values(
            selected, "high_response", "normalized_ap"
        )["value"]
        candidate_activation = activations[
            activations["candidate_id"] == candidate_id
        ]
        candidate_gates = gates[gates["candidate_id"] == candidate_id]
        best_epoch_positive_count = int(
            np.sum(candidate_activation["best_epoch"].to_numpy() > 0)
        )
        changed_support_count = int(
            np.sum(
                candidate_activation["maximum_prediction_difference"].to_numpy()
                > 1e-6
            )
        )
        corrected_fraction = float(
            candidate_activation["corrected_sample_fraction"].median()
        )
        contribution_ratio = float(
            candidate_activation["median_contribution_ratio"].median()
        )
        gate_not_collapsed = bool(
            (candidate_gates["lower_saturation_fraction"] < 0.95).all()
            and (candidate_gates["upper_saturation_fraction"] < 0.95).all()
        )
        activation_passed = bool(
            best_epoch_positive_count >= 4
            and changed_support_count >= 4
            and corrected_fraction >= 0.20
            and 0.02 <= contribution_ratio <= 0.30
            and gate_not_collapsed
        )
        safety_counts = {
            "maneuver": int(np.sum(maneuver["full"] >= maneuver["direct"] - 0.005)),
            "response": int(np.sum(response["full"] <= response["direct"] * 1.01)),
            "high_response": int(np.sum(high["full"] >= high["direct"] - 0.005)),
        }
        safety_passed = min(safety_counts.values()) >= 4
        maneuver_mean = float(maneuver["full"].mean())
        maneuver_direct_mean = float(maneuver["direct"].mean())
        maneuver_worst = float(maneuver["full"].min())
        maneuver_direct_worst = float(maneuver["direct"].min())
        maneuver_improved = bool(
            maneuver_mean >= maneuver_direct_mean + 0.005
            or maneuver_worst >= maneuver_direct_worst + 0.03
        )
        response_improved = bool(
            float(response_ratio.median()) <= 0.98
            and float(response_skill.mean()) > 0
            and int(np.sum(response_skill.to_numpy() > 0)) >= 4
        )
        high_mean = float(normalized_ap.mean())
        direct_normalized = values(
            baseline, "high_response", "normalized_ap"
        )["value"]
        high_improved = bool(
            (high_mean >= 0.40 or high_mean >= float(direct_normalized.mean()) + 0.03)
            and int(np.sum(normalized_ap.to_numpy() > 0)) >= 5
        )
        improved_count = sum((maneuver_improved, response_improved, high_improved))
        research_passed = bool(
            activation_passed
            and safety_passed
            and improved_count >= 2
            and (maneuver_improved or response_improved)
        )
        output.append(
            {
                "candidate_id": candidate_id,
                "gate_mode": candidate.gate_mode,
                "initial_gate": candidate.initial_gate,
                "objective_mode": candidate.objective_mode,
                "adapter_mode": candidate.adapter_mode,
                "use_distillation": candidate.use_distillation,
                "best_epoch_positive_support_count": best_epoch_positive_count,
                "prediction_changed_support_count": changed_support_count,
                "median_corrected_sample_fraction": corrected_fraction,
                "median_residual_contribution_ratio": contribution_ratio,
                "gate_not_collapsed": gate_not_collapsed,
                "activation_gate_passed": activation_passed,
                "maneuver_safe_support_count": safety_counts["maneuver"],
                "response_safe_support_count": safety_counts["response"],
                "high_response_safe_support_count": safety_counts["high_response"],
                "safety_gate_passed": safety_passed,
                "maneuver_mean_macro_f1": maneuver_mean,
                "maneuver_direct_mean_macro_f1": maneuver_direct_mean,
                "maneuver_worst_macro_f1": maneuver_worst,
                "maneuver_direct_worst_macro_f1": maneuver_direct_worst,
                "response_median_rmse_ratio": float(response_ratio.median()),
                "response_mean_skill": float(response_skill.mean()),
                "response_positive_skill_count": int(
                    np.sum(response_skill.to_numpy() > 0)
                ),
                "high_response_mean_normalized_ap": high_mean,
                "high_response_direct_mean_normalized_ap": float(
                    direct_normalized.mean()
                ),
                "high_response_positive_support_count": int(
                    np.sum(normalized_ap.to_numpy() > 0)
                ),
                "maneuver_improved": maneuver_improved,
                "response_improved": response_improved,
                "high_response_improved": high_improved,
                "improved_task_count": improved_count,
                "research_gate_passed": research_passed,
                "ranking_score": float(
                    maneuver_mean
                    + (1.0 - float(response_ratio.median()))
                    + high_mean
                ),
                "outer_test_opened": False,
            }
        )
    return output


def _result_metric_rows(*, split_id, candidate_id, direct_metrics, full_metrics):
    rows = _metric_rows(
        split_id=split_id,
        candidate_id=candidate_id,
        variant="direct_only",
        metrics=direct_metrics,
        direct_metrics=None,
    )
    rows.extend(
        _metric_rows(
            split_id=split_id,
            candidate_id=candidate_id,
            variant="full",
            metrics=full_metrics,
            direct_metrics=direct_metrics,
        )
    )
    return rows


def _aligned_values(full, direct):
    aligned = full.merge(
        direct,
        on="split_id",
        suffixes=("_full", "_direct"),
        validate="one_to_one",
    )
    if len(aligned) != 6:
        raise ValueError("stage-3A metrics must align on six supports")
    return aligned.rename(
        columns={"value_full": "full", "value_direct": "direct"}
    )


def _select_candidate(rows):
    eligible = [row for row in rows if row["research_gate_passed"]]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["ranking_score"], row["candidate_id"]))


def _validate_stage2(allowance, selection):
    if not bool(allowance.get("frozen_safety_passed")):
        raise PermissionError("stage 3A requires the accepted stage-2 safety anchor")
    if bool(allowance.get("outer_test_opened")):
        raise PermissionError("stage-2 outer test must remain closed")
    if selection is None or selection.get("candidate_id") != "vehicle_observed_scalar":
        raise ValueError("stage-3A source safety anchor changed")


def _validate_plans(plans):
    if len(plans) != 6:
        raise ValueError("stage 3A requires six main-selection supports")
    if len({row["validation_support_hash"] for row in plans.values()}) != 6:
        raise ValueError("stage-3A validation supports are not unique")
    if any(int(row["support_overlap_count"]) != 0 for row in plans.values()):
        raise PermissionError("stage-3A support overlap must be zero")


def _protocol(config, input_paths, plans, selection):
    module_root = Path(__file__).parent
    implementation_paths = {
        "contracts": module_root / "residual_activation_contracts.py",
        "model": module_root / "residual_activation_model.py",
        "optimization": module_root / "residual_activation_optimization.py",
        "runner": Path(__file__),
    }
    return {
        "format": "chronaris.dingxin_residual_activation_protocol.v1",
        "baseline_commit": "e2e769b8a5371083bdbc9aca035d818f968c6c7a",
        "config": asdict(config),
        "stage2_selected_candidate": selection,
        "candidate_count": len(CANDIDATES),
        "maximum_candidate_count": 8,
        "split_count": len(plans),
        "validation_support_hashes": sorted(
            str(row["validation_support_hash"]) for row in plans.values()
        ),
        "input_sha256": {
            name: sha256_file(path)
            for name, path in input_paths.items()
            if path.is_file()
        },
        "implementation_sha256": {
            name: sha256_file(path) for name, path in implementation_paths.items()
        },
        "teacher_training_predictions": "cross_fitted_out_of_fold",
        "teacher_runtime_required": False,
        "stage_3b_conditional": True,
        "configuration_locked": False,
        "outer_test_opened": False,
        "historical_outer_metrics_used_for_selection": False,
    }


def _write_evidence_manifest(compact: Path, report_name: str) -> None:
    expected = (
        "protocol.json",
        "architecture_manifest.json",
        "candidate_grid.json",
        "teacher_inventory.json",
        "candidate_support_metrics.csv",
        "candidate_training.csv",
        "gradient_conflict.csv",
        "residual_activation_diagnostics.csv",
        "sample_gate_statistics.csv",
        "teacher_oof_statistics.csv",
        "access_audit.csv",
        "candidate_summary.csv",
        "selected_candidates.json",
        "allowance.json",
        "progress.json",
        "resume_command.txt",
        "figure_manifest.json",
        "residual_activation_task_metrics.png",
        "residual_activation_usage.png",
        report_name,
    )
    missing = [name for name in expected if not (compact / name).is_file()]
    if missing:
        raise FileNotFoundError(f"stage-3A compact evidence missing: {missing}")
    write_json(
        compact / "evidence_manifest.json",
        {
            "format": "chronaris.dingxin_residual_activation_evidence.v1",
            "files": [
                {
                    "path": name,
                    "sha256": sha256_file(compact / name),
                    "size_bytes": (compact / name).stat().st_size,
                }
                for name in expected
            ],
            "configuration_locked": False,
            "outer_test_opened": False,
        },
    )
def _candidate_grid():
    return {
        "candidate_count": len(CANDIDATES),
        "maximum_candidate_count": 8,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "unbounded_search_allowed": False,
        "outer_test_opened": False,
    }


def _architecture_manifest():
    return {
        "model_name": "Chronaris",
        "reader_visible_version_suffix": False,
        "safe_direct_branch_integrated": True,
        "continuous_path": "dual_stream_continuous_causal_residual",
        "task_specific_adapters": True,
        "gate_inputs": [
            "chronaris_summary",
            "safe_anchor_confidence",
            "safe_anchor_margin",
        ],
        "forbidden_gate_inputs": [
            "absolute_flight_progress",
            "window_index",
            "view_identity",
            "pilot_identity",
            "sortie_identity",
        ],
        "gate_range": [0.0, 0.5],
        "residual_initialization": "small_random",
        "teacher_runtime_required": False,
        "outer_test_opened": False,
    }


def _teacher_inventory():
    return {
        "maneuver_teachers": ["航电单流", "连续时间注意力基线"],
        "response_teachers": [
            "跨模态 Transformer",
            "连续时间注意力基线",
            "朴素时间同步",
        ],
        "high_response_teachers": ["航电单流", "跨模态 Transformer"],
        "training_prediction_protocol": "cross_fitted_out_of_fold",
        "selection": "teacher_consensus_and_oof_advantage_over_safe_anchor",
        "teacher_runtime_required": False,
        "outer_test_opened": False,
    }
