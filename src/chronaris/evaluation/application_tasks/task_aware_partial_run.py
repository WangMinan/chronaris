"""Run the bounded partial-unfreeze screen after frozen residual acceptance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from chronaris.evaluation.application_tasks.clean_input_contract import (
    build_clean_input_contract,
    clean_guarded_provider,
)
from chronaris.evaluation.application_tasks.dingxin_context_data import (
    build_dingxin_lazy_context_index,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
)
from chronaris.evaluation.application_tasks.task_aware_partial_unfreeze import (
    PartialUnfreezeConfig,
    train_partially_unfrozen_chronaris,
)
from chronaris.evaluation.application_tasks.task_aware_safe_residual_model import (
    fit_safe_anchor_predictions,
)
from chronaris.modeling.training import load_common_pretraining_checkpoint
from chronaris.representation import coalesce_observation_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


PARTIAL_RATIOS = (0.02, 0.05)


@dataclass(frozen=True, slots=True)
class PartialScreenResult:
    selected_candidate_id: str | None
    partial_safety_passed: bool
    research_gate_passed: bool
    allow_teacher_distillation: bool
    outer_test_opened: bool


def run_partial_unfreeze_screen(
    *,
    compact_root: Path,
    heavy_root: Path,
    source_root: Path,
    matched_root: Path,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    resume: bool,
    selected_frozen_candidate: Mapping[str, object],
    selected_frozen_candidate_index: int,
    split_payloads: Mapping[str, Mapping[str, object]],
) -> PartialScreenResult:
    """Run two pre-registered backbone learning-rate ratios on six supports."""

    provider_factory, input_contract = _provider_factory(source_root)
    candidate_ids = tuple(f"partial_ratio_{ratio:.2f}" for ratio in PARTIAL_RATIOS)
    _write_json(
        compact_root / "partial_unfreeze_protocol.json",
        {
            "format": "chronaris.dingxin_task_aware_partial_unfreeze.v1",
            "source_frozen_candidate": dict(selected_frozen_candidate),
            "backbone_learning_rate_ratios": list(PARTIAL_RATIOS),
            "partial_unfreeze_scope": [
                "observation_encoders",
                "observation_updates",
                "causal_fusion_output_projection",
                "lag_scale_gate",
            ],
            "input_feature_contract": input_contract,
            "selection_support_count": len(split_payloads),
            "task_targets_opened": True,
            "outer_test_opened": False,
        },
    )
    metric_rows = []
    training_rows = []
    gate_rows = []
    access_rows = []
    trainable_rows = []
    for split_index, (split_id, payload) in enumerate(sorted(split_payloads.items())):
        task_data = payload["task_data"]
        targets = payload["targets"]
        features = payload["features"]
        anchor_seed = seed + split_index * 100 + selected_frozen_candidate_index
        anchors = fit_safe_anchor_predictions(
            maneuver_train_features=features[
                str(selected_frozen_candidate["maneuver_anchor"])
            ]["train_maneuver"],
            maneuver_validation_features=features[
                str(selected_frozen_candidate["maneuver_anchor"])
            ]["validation_maneuver"],
            response_train_features=features[
                str(selected_frozen_candidate["response_anchor"])
            ]["train_response"],
            response_validation_features=features[
                str(selected_frozen_candidate["response_anchor"])
            ]["validation_response"],
            targets=targets,
            seed=anchor_seed,
        )
        all_ids = tuple(
            sorted(
                set(task_data["maneuver"]["train_ids"])
                | set(task_data["maneuver"]["validation_ids"])
                | set(task_data["response"]["train_ids"])
                | set(task_data["response"]["validation_ids"])
            )
        )
        provider, access = provider_factory(all_ids)
        raw_batches = {
            "train_maneuver": provider(task_data["maneuver"]["train_ids"]),
            "validation_maneuver": provider(
                task_data["maneuver"]["validation_ids"]
            ),
            "train_response": provider(task_data["response"]["train_ids"]),
            "validation_response": provider(
                task_data["response"]["validation_ids"]
            ),
        }
        access_rows.append(
            {
                "split_id": split_id,
                **access,
                "outer_test_opened": False,
            }
        )
        frozen_path = (
            heavy_root
            / "frozen"
            / str(selected_frozen_candidate["candidate_id"])
            / split_id
            / "best.pt"
        )
        frozen_payload = torch.load(frozen_path, map_location="cpu", weights_only=False)
        encoder_checkpoint = (
            matched_root
            / "checkpoints"
            / f"seed_{seed}"
            / split_id
            / "chronaris"
            / "best.pt"
        )
        for ratio in PARTIAL_RATIOS:
            candidate_id = f"partial_ratio_{ratio:.2f}"
            root = heavy_root / "partial" / candidate_id / split_id
            state_path = root / "state.json"
            checkpoint_path = root / "best.pt"
            unit_hash = _stable_hash(
                {
                    "frozen_checkpoint_sha256": sha256_file(frozen_path),
                    "encoder_checkpoint_sha256": sha256_file(encoder_checkpoint),
                    "split_id": split_id,
                    "ratio": ratio,
                    "max_epochs": max_epochs,
                    "patience": patience,
                }
            )
            if resume and state_path.is_file() and checkpoint_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("unit_sha256") != unit_hash:
                    raise ValueError("partial-unfreeze resume unit changed")
                metric_rows.extend(state["metric_rows"])
                training_rows.extend(state["training_rows"])
                gate_rows.extend(state["gate_rows"])
                trainable_rows.extend(state["trainable_rows"])
                continue
            encoder, _heads, normalizer, source_payload = (
                load_common_pretraining_checkpoint(
                    encoder_checkpoint,
                    device=device,
                )
            )
            if source_payload.get("method_name") != "chronaris":
                raise ValueError("partial-unfreeze source is not Chronaris")
            result = train_partially_unfrozen_chronaris(
                encoder=encoder,
                normalizer=normalizer,
                train_maneuver_batch=raw_batches["train_maneuver"],
                validation_maneuver_batch=raw_batches["validation_maneuver"],
                train_response_batch=raw_batches["train_response"],
                validation_response_batch=raw_batches["validation_response"],
                anchors=anchors,
                targets=targets,
                residual_state_dict=frozen_payload["model_state_dict"],
                feature_mean=frozen_payload["feature_mean"],
                feature_scale=frozen_payload["feature_scale"],
                gate_mode=str(selected_frozen_candidate["gate_mode"]),
                config=PartialUnfreezeConfig(
                    backbone_learning_rate_ratio=ratio,
                    max_epochs=max_epochs,
                    patience=patience,
                    seed=anchor_seed + int(ratio * 1_000),
                    device=device,
                ),
            )
            unit_metric_rows = _metric_rows(
                split_id=split_id,
                candidate_id=candidate_id,
                frozen=result.frozen_metrics,
                partial=result.partial_metrics,
            )
            unit_training_rows = [
                {
                    "split_id": split_id,
                    "candidate_id": candidate_id,
                    "best_epoch": result.best_epoch,
                    "epoch_count": len(result.epoch_rows),
                    "safety_passed": result.safety_passed,
                    "outer_test_opened": False,
                    **row,
                }
                for row in result.epoch_rows
            ]
            unit_gate_rows = [
                {
                    "split_id": split_id,
                    "candidate_id": candidate_id,
                    "task": task,
                    "gate_index": index,
                    "gate_value": value,
                    "outer_test_opened": False,
                }
                for task, values in result.gate_values.items()
                for index, value in enumerate(values)
            ]
            unit_trainable_rows = [
                {
                    "split_id": split_id,
                    "candidate_id": candidate_id,
                    "parameter_name": name,
                    "outer_test_opened": False,
                }
                for name in result.trainable_encoder_parameter_names
            ]
            root.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": "chronaris.dingxin_task_aware_partial_checkpoint.v1",
                    "unit_sha256": unit_hash,
                    "model_state_dict": dict(result.model_state_dict),
                    "best_epoch": result.best_epoch,
                    "source_encoder_checkpoint_sha256": sha256_file(
                        encoder_checkpoint
                    ),
                    "source_frozen_checkpoint_sha256": sha256_file(frozen_path),
                    "label_used_for_encoder_training": True,
                    "encoder_update_mode": "partial_preregistered_scope",
                    "outer_test_opened": False,
                },
                checkpoint_path,
            )
            _write_json(
                state_path,
                {
                    "unit_sha256": unit_hash,
                    "metric_rows": unit_metric_rows,
                    "training_rows": unit_training_rows,
                    "gate_rows": unit_gate_rows,
                    "trainable_rows": unit_trainable_rows,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                },
            )
            metric_rows.extend(unit_metric_rows)
            training_rows.extend(unit_training_rows)
            gate_rows.extend(unit_gate_rows)
            trainable_rows.extend(unit_trainable_rows)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = pd.DataFrame(metric_rows)
    training = pd.DataFrame(training_rows)
    gates = pd.DataFrame(gate_rows)
    trainable = pd.DataFrame(trainable_rows)
    access_frame = pd.DataFrame(access_rows)
    aggregate = _aggregate(metrics, gates, candidate_ids)
    selected = _select(aggregate)
    selected_id = None if selected is None else str(selected["candidate_id"])
    allowance = {
        "decision": "partial_research_gate_passed" if selected else "gap",
        "selected_partial_candidate_id": selected_id,
        "partial_safety_passed": any(
            bool(row["partial_safety_passed"]) for row in aggregate
        ),
        "research_gate_passed": selected is not None,
        "allow_teacher_distillation": selected is not None,
        "configuration_locked": False,
        "outer_test_opened": False,
    }
    metrics.to_csv(compact_root / "partial_unfreeze_metrics.csv", index=False)
    training.to_csv(compact_root / "partial_training.csv", index=False)
    gates.to_csv(compact_root / "partial_gate_statistics.csv", index=False)
    trainable.to_csv(compact_root / "partial_trainable_parameters.csv", index=False)
    access_frame.to_csv(compact_root / "partial_access_audit.csv", index=False)
    pd.DataFrame(aggregate).to_csv(
        compact_root / "partial_candidate_metrics.csv", index=False
    )
    _write_json(
        compact_root / "partial_selected.json",
        {"selected": selected, **allowance},
    )
    return PartialScreenResult(
        selected_candidate_id=selected_id,
        partial_safety_passed=bool(allowance["partial_safety_passed"]),
        research_gate_passed=bool(allowance["research_gate_passed"]),
        allow_teacher_distillation=bool(allowance["allow_teacher_distillation"]),
        outer_test_opened=False,
    )


def _provider_factory(source_root):
    fixed = source_root / "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    snapshot = (
        source_root
        / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    role_path = fixed / "field_role_manifest.csv"
    index = build_dingxin_lazy_context_index(
        snapshot_root=snapshot,
        field_role_manifest_path=role_path,
        context_manifest_path=fixed / "context_sample_manifest.jsonl",
    )
    removed, contract = build_clean_input_contract(
        role_path=str(role_path),
        vehicle_raw_to_index=index.plan.vehicle_raw_to_index,
        vehicle_channel_count=len(index.plan.schema.vehicle_feature_names),
    )

    def base_provider(sample_ids):
        return coalesce_observation_batch(
            index.load_batch(sample_ids),
            bin_width_s=DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
        )

    def factory(allowed_ids):
        return clean_guarded_provider(
            base_provider,
            allowed_sample_ids=allowed_ids,
            removed_indices=removed,
        )

    return factory, contract


def _metric_rows(*, split_id, candidate_id, frozen, partial):
    rows = []
    primary = {
        "maneuver": ("macro_f1", "higher"),
        "response": ("rmse", "lower"),
        "high_response": ("auprc", "higher"),
    }
    for task, values in partial.items():
        for metric, value in values.items():
            baseline = float(frozen[task][metric])
            direction = primary[task][1] if metric == primary[task][0] else "diagnostic"
            rows.append(
                {
                    "split_id": split_id,
                    "candidate_id": candidate_id,
                    "task": task,
                    "metric": metric,
                    "direction": direction,
                    "frozen_value": baseline,
                    "partial_value": float(value),
                    "direction_normalized_delta": (
                        float(baseline - value)
                        if direction == "lower"
                        else float(value - baseline)
                    ),
                    "outer_test_opened": False,
                }
            )
    return rows


def _aggregate(frame, gates, candidate_ids):
    output = []
    for candidate_id in candidate_ids:
        selected = frame[frame["candidate_id"] == candidate_id]

        def primary(task, metric):
            return selected[
                (selected["task"] == task) & (selected["metric"] == metric)
            ]

        maneuver = primary("maneuver", "macro_f1")
        response = primary("response", "rmse")
        response_ratio = primary("response", "rmse_ratio")
        response_skill = primary("response", "response_skill")
        high = primary("high_response", "auprc")
        high_normalized = primary("high_response", "normalized_ap")
        safety_counts = {
            "maneuver": int(
                np.sum(maneuver["partial_value"] >= maneuver["frozen_value"] - 0.005)
            ),
            "response": int(
                np.sum(response["partial_value"] <= response["frozen_value"] * 1.01)
            ),
            "high_response": int(
                np.sum(high["partial_value"] >= high["frozen_value"] - 0.005)
            ),
        }
        means_improved = {
            "maneuver": bool(
                maneuver["partial_value"].mean() > maneuver["frozen_value"].mean()
            ),
            "response": bool(
                response["partial_value"].mean() < response["frozen_value"].mean()
            ),
            "high_response": bool(
                high["partial_value"].mean() > high["frozen_value"].mean()
            ),
        }
        gate_values = gates[gates["candidate_id"] == candidate_id]["gate_value"]
        gate_ok = bool(
            len(gate_values)
            and (gate_values > 1e-5).all()
            and (gate_values < 0.99999).all()
        )
        safety_passed = min(safety_counts.values()) >= 4 and gate_ok
        improved_count = sum(means_improved.values())
        research_passed = safety_passed and improved_count >= 2
        output.append(
            {
                "candidate_id": candidate_id,
                "maneuver_mean_macro_f1": float(maneuver["partial_value"].mean()),
                "maneuver_median_macro_f1": float(
                    maneuver["partial_value"].median()
                ),
                "maneuver_worst_macro_f1": float(maneuver["partial_value"].min()),
                "response_median_rmse_ratio": float(
                    response_ratio["partial_value"].median()
                ),
                "response_mean_skill": float(
                    response_skill["partial_value"].mean()
                ),
                "response_positive_skill_count": int(
                    np.sum(response_skill["partial_value"] > 0)
                ),
                "high_response_mean_normalized_ap": float(
                    high_normalized["partial_value"].mean()
                ),
                "high_response_median_normalized_ap": float(
                    high_normalized["partial_value"].median()
                ),
                "high_response_positive_split_count": int(
                    np.sum(high_normalized["partial_value"] > 0)
                ),
                "maneuver_safe_support_count": safety_counts["maneuver"],
                "response_safe_support_count": safety_counts["response"],
                "high_response_safe_support_count": safety_counts["high_response"],
                "maneuver_mean_improved": means_improved["maneuver"],
                "response_mean_improved": means_improved["response"],
                "high_response_mean_improved": means_improved["high_response"],
                "improved_task_count": improved_count,
                "gate_non_degenerate": gate_ok,
                "partial_safety_passed": safety_passed,
                "research_gate_passed": research_passed,
                "ranking_score": float(
                    maneuver["partial_value"].mean()
                    + (1.0 - response_ratio["partial_value"].median())
                    + high_normalized["partial_value"].mean()
                ),
                "outer_test_opened": False,
            }
        )
    return output


def _select(rows):
    eligible = [row for row in rows if row["research_gate_passed"]]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["ranking_score"], row["candidate_id"]))


def _stable_hash(payload):
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
