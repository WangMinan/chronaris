"""Three-fold Dingxin task-independent confirmation of the v2 top three."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_candidate_diagnostics import (
    diagnose_chronaris_candidate,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    ensure_dingxin_model_input_contract,
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    _build_guarded_cached_provider,
)
from chronaris.modeling.training import (
    ChronarisV2CandidateConfig,
    ChronarisV2TrainingConfig,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


INNER_VIEW_FOLDS = (
    "leave_one_view_out__fold01",
    "leave_one_view_out__fold02",
    "leave_one_view_out__fold03",
)


@dataclass(frozen=True, slots=True)
class ChronarisV2DingxinInnerConfirmationConfig:
    run_id: str = "2026-07-12_chronaris-v2-dingxin-inner-confirmation"
    hyperparameter_diagnostics_run_id: str = (
        "2026-07-12_chronaris-v2-hyperparameter-diagnostics-seed17"
    )
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    reference_pretraining_run_id: str = (
        "2026-07-12_dingxin-locked-pretraining-coalesced"
    )
    max_epochs: int = 50
    batch_size: int = 32
    device: str = "cuda"


def run_chronaris_v2_dingxin_inner_confirmation(
    config: ChronarisV2DingxinInnerConfirmationConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2DingxinInnerConfirmationConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    ensure_dingxin_model_input_contract(heavy_root)
    hyper_root = (
        Path(resolved.compact_output_root)
        / resolved.hyperparameter_diagnostics_run_id
    )
    hyper_manifest = _read_json(hyper_root / "evidence_manifest.json")
    selected_payload = _read_json(hyper_root / "selected_top_three.json")
    if hyper_manifest.get("status") != "selected_top_three" or len(selected_payload) != 3:
        raise PermissionError("Dingxin inner confirmation requires a locked top three")
    candidates = tuple(
        _candidate_from_row(candidate_id, row)
        for candidate_id, row in selected_payload.items()
    )
    result_rows = []
    diagnostic_rows = []
    access_rows = []
    for fold_id in INNER_VIEW_FOLDS:
        data = load_dingxin_fold_pretraining_data(
            fold_id=fold_id,
            snapshot_root=resolved.snapshot_root,
            fixed_audit_root=resolved.fixed_audit_root,
            inner_split_root=resolved.inner_split_root,
        )
        provider, access = _build_guarded_cached_provider(
            data.load_batch,
            allowed_sample_ids=(
                data.fold.train_sample_ids + data.fold.validation_sample_ids
            ),
            forbidden_sample_ids=data.fold.held_out_sample_ids,
        )
        normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
            provider,
            train_sample_ids=data.fold.train_sample_ids,
            held_out_sample_ids=(
                data.fold.validation_sample_ids + data.fold.held_out_sample_ids
            ),
            batch_size=2,
        )
        train_batch = provider(data.fold.train_sample_ids)
        validation_batch = provider(data.fold.validation_sample_ids)
        physiology_reference = _reference_checkpoint(
            resolved,
            fold_id,
            "physiology_only/A/best.pt",
        )
        vehicle_reference = _reference_checkpoint(
            resolved,
            fold_id,
            "vehicle_only/C/best.pt",
        )
        for candidate in candidates:
            result = train_chronaris_v2_candidate(
                candidate=candidate,
                batch=None,
                batch_provider=provider,
                fold=data.fold,
                physiology_feature_names=(
                    data.index.plan.schema.physiology_feature_names
                ),
                vehicle_feature_names=data.index.plan.schema.vehicle_feature_names,
                vehicle_field_labels=data.vehicle_field_labels,
                normalizer=normalizer,
                output_root=heavy_root / "checkpoints" / fold_id,
                config=ChronarisV2TrainingConfig(
                    max_epochs=resolved.max_epochs,
                    batch_size=resolved.batch_size,
                    patience=min(8, resolved.max_epochs),
                    seed=17,
                    device=resolved.device,
                ),
                resume=True,
            )
            result_rows.append(
                {
                    "fold_id": fold_id,
                    "candidate_id": candidate.candidate_id,
                    "status": result.status,
                    "best_epoch": result.best_epoch,
                    "completed_epochs": result.completed_epochs,
                    "public_self_supervised_validation_loss": (
                        result.best_public_selection_loss
                    ),
                    "checkpoint_path": result.best_checkpoint_path,
                    "checkpoint_sha256": _sha256_file(
                        Path(result.best_checkpoint_path)
                    ),
                }
            )
            diagnostic = diagnose_chronaris_candidate(
                candidate_checkpoint=result.best_checkpoint_path,
                candidate_id=candidate.candidate_id,
                physiology_reference_checkpoint=physiology_reference,
                vehicle_reference_checkpoint=vehicle_reference,
                train_batch=train_batch,
                validation_batch=validation_batch,
                fold_id=fold_id,
                device=resolved.device,
            )
            diagnostic_rows.append(
                {
                    "fold_id": fold_id,
                    **diagnostic.evidence.to_dict(),
                }
            )
        access_rows.append({"fold_id": fold_id, **access})
        if access["forbidden_request_count"]:
            raise ValueError("v2 inner confirmation requested outer-test observations")
    aggregate = _aggregate_candidates(candidates, result_rows, diagnostic_rows)
    eligible = [row for row in aggregate if row["three_fold_gate_passed"]]
    eligible.sort(
        key=lambda row: (
            row["mean_public_self_supervised_validation_loss"],
            -row["worst_fold_fidelity_ratio"],
            row["parameter_count"],
            row["candidate_id"],
        )
    )
    selected = eligible[0] if eligible else None
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
        row["locked"] = rank == 1
    pd.DataFrame(result_rows).to_csv(compact_root / "fold_training.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        compact_root / "fold_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(aggregate).to_csv(compact_root / "candidate_aggregate.csv", index=False)
    pd.DataFrame(access_rows).to_csv(compact_root / "provider_access.csv", index=False)
    status = "configuration_locked" if selected is not None else "gates_failed"
    lock = {
        "format": "chronaris.v2_locked_configuration.v1",
        "configuration_locked": selected is not None,
        "status": status,
        "candidate": (
            asdict(
                next(
                    candidate
                    for candidate in candidates
                    if candidate.candidate_id == selected["candidate_id"]
                )
            )
            if selected is not None
            else None
        ),
        "selection_uses_downstream_labels": False,
        "outer_test_opened": False,
        "sealed_confirmation_opened": False,
        "source_hyperparameter_diagnostics_run_id": (
            resolved.hyperparameter_diagnostics_run_id
        ),
    }
    _write_json(compact_root / "locked_configuration.json", lock)
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_dingxin_inner_confirmation_evidence.v1",
            "run_id": resolved.run_id,
            "status": status,
            "fold_count": len(INNER_VIEW_FOLDS),
            "candidate_count": len(candidates),
            "locked_candidate_id": (
                selected["candidate_id"] if selected is not None else None
            ),
            "outer_test_request_count": sum(
                row["forbidden_request_count"] for row in access_rows
            ),
            "confirmed_metrics_changed": False,
        },
    )
    return compact_root


def _candidate_from_row(candidate_id, row):
    names = {field.name for field in fields(ChronarisV2CandidateConfig)}
    values = {name: row[name] for name in names if name in row}
    values["candidate_id"] = candidate_id
    return ChronarisV2CandidateConfig(**values)


def _aggregate_candidates(candidates, result_rows, diagnostic_rows):
    rows = []
    for candidate in candidates:
        training = [
            row for row in result_rows if row["candidate_id"] == candidate.candidate_id
        ]
        diagnostics = [
            row
            for row in diagnostic_rows
            if row["candidate_id"] == candidate.candidate_id
        ]
        worst_fidelity = min(
            float(row["worst_fold_fidelity_ratio"]) for row in diagnostics
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "fold_count": len(training),
                "mean_public_self_supervised_validation_loss": sum(
                    float(row["public_self_supervised_validation_loss"])
                    for row in training
                )
                / len(training),
                "worst_fold_fidelity_ratio": worst_fidelity,
                "parameter_count": max(
                    int(row["parameter_count"]) for row in diagnostics
                ),
                "three_fold_gate_passed": (
                    len(training) == 3
                    and len(diagnostics) == 3
                    and worst_fidelity >= 0.98
                    and all(
                        row["effective_rank"] >= 2
                        and row["near_zero_variance_fraction"] <= 0.10
                        and row["causal_future_invariance_passed"]
                        and row["invalid_query_pooling_passed"]
                        and row["lag_mask_passed"]
                        for row in diagnostics
                    )
                ),
                "rank": None,
                "locked": False,
            }
        )
    return rows


def _reference_checkpoint(config, fold_id, suffix):
    path = (
        Path(config.heavy_output_root)
        / config.reference_pretraining_run_id
        / "checkpoints"
        / "seed_17"
        / fold_id
        / suffix
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing Dingxin single-stream reference: {path}")
    return path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
