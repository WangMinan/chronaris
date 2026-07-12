"""Locked three-seed, five-fold Dingxin retraining for Chronaris v2 only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
    ensure_dingxin_model_input_contract,
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    DEFAULT_FOLDS,
    _build_guarded_cached_provider,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.training import (
    ChronarisV2CandidateConfig,
    ChronarisV2TrainingConfig,
    train_chronaris_v2_candidate,
)
from chronaris.representation import TrainOnlyRobustNormalizer


@dataclass(frozen=True, slots=True)
class ChronarisV2LockedDingxinConfig:
    run_id: str = "2026-07-13_chronaris-v2-dingxin-locked-pretraining"
    locked_configuration_path: str = (
        "docs/artifacts/runs/"
        "2026-07-12_chronaris-v2-dingxin-inner-confirmation-r3/"
        "locked_configuration.json"
    )
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    fold_ids: tuple[str, ...] = DEFAULT_FOLDS
    seeds: tuple[int, ...] = LOCKED_SEEDS
    max_epochs: int = 50
    batch_size: int = 32
    device: str = "cuda"
    resume: bool = True


def run_chronaris_v2_locked_dingxin_pretraining(
    config: ChronarisV2LockedDingxinConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2LockedDingxinConfig()
    candidate, lock = _load_locked_candidate(resolved.locked_configuration_path)
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    ensure_dingxin_model_input_contract(heavy_root)
    rows: list[dict[str, object]] = []
    access_rows: list[dict[str, object]] = []
    for fold_id in resolved.fold_ids:
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
        for seed in resolved.seeds:
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
                output_root=(
                    heavy_root
                    / "checkpoints"
                    / f"seed_{seed}"
                    / fold_id
                    / "chronaris"
                ),
                config=ChronarisV2TrainingConfig(
                    max_epochs=resolved.max_epochs,
                    batch_size=resolved.batch_size,
                    patience=min(8, resolved.max_epochs),
                    seed=seed,
                    device=resolved.device,
                ),
                resume=resolved.resume,
            )
            checkpoint = Path(result.last_checkpoint_path)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            rows.append(
                {
                    "seed": seed,
                    "fold_id": fold_id,
                    "candidate_id": candidate.candidate_id,
                    "status": result.status,
                    "completed_epochs": result.completed_epochs,
                    "best_epoch": result.best_epoch,
                    "public_self_supervised_validation_loss": (
                        result.best_public_selection_loss
                    ),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "task_labels_opened": bool(
                        payload["label_used_for_encoder_training"]
                    ),
                    "simulation_oracle_opened": bool(
                        payload["simulation_oracle_opened"]
                    ),
                    "outer_test_opened": bool(payload["locked_test_opened"]),
                    "normalizer_sha256": normalizer.to_manifest()[
                        "transform_sha256"
                    ],
                }
            )
        access_rows.append({"fold_id": fold_id, **access})
        if access["forbidden_request_count"]:
            raise ValueError("v2 locked training requested Dingxin outer-test data")
    expected = len(resolved.seeds) * len(resolved.fold_ids)
    acceptance = (
        _check("all_seed_fold_checkpoints", len(rows) == expected),
        _check(
            "all_training_complete",
            all(row["status"] in {"completed", "resumed"} for row in rows),
        ),
        _check(
            "forbidden_sources_closed",
            all(
                not row["task_labels_opened"]
                and not row["simulation_oracle_opened"]
                and not row["outer_test_opened"]
                for row in rows
            ),
        ),
        _check(
            "outer_test_provider_requests_zero",
            all(not row["forbidden_request_count"] for row in access_rows),
        ),
    )
    status = "completed" if all(row["passed"] for row in acceptance) else "partial"
    pd.DataFrame(rows).to_csv(compact_root / "checkpoint_inventory.csv", index=False)
    pd.DataFrame(access_rows).to_csv(compact_root / "provider_access.csv", index=False)
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(
        compact_root / "protocol.json",
        {
            "format": "chronaris.v2_dingxin_locked_pretraining_protocol.v1",
            "config": asdict(resolved),
            "locked_candidate": asdict(candidate),
            "locked_configuration_sha256": _sha256_file(
                Path(resolved.locked_configuration_path)
            ),
            "representation_family": "frozen_task_agnostic_v2",
            "model_input_bin_width_s": DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
            "selection_uses_downstream_labels": False,
            "outer_test_opened": False,
        },
    )
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_dingxin_locked_pretraining_evidence.v1",
            "run_id": resolved.run_id,
            "status": status,
            "checkpoint_count": len(rows),
            "outer_test_request_count": sum(
                int(row["forbidden_request_count"]) for row in access_rows
            ),
            "heavy_run_root": str(heavy_root),
            "confirmed_metrics_changed": False,
        },
    )
    (compact_root / "report.md").write_text(
        "\n".join((
            "# Chronaris v2 鼎新锁定重训",
            "",
            f"状态：{status}；完成 {len(rows)}/{expected} 个随机种子—外层折 checkpoint。",
            "训练只读取 inner-train 与 validation 原始观测，outer-test provider 请求为 0。",
            "任务标签、仿真真值和既有确认指标均未参与训练或早停。",
            "",
        )),
        encoding="utf-8",
    )
    (compact_root / "resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/"
        "run_chronaris_v2_locked_dingxin_pretraining.py "
        f"--run-id {resolved.run_id} --device {resolved.device} "
        f"--max-epochs {resolved.max_epochs}\n",
        encoding="utf-8",
    )
    return compact_root


def _load_locked_candidate(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("format") != "chronaris.v2_locked_configuration.v1"
        or payload.get("configuration_locked") is not True
        or payload.get("selection_uses_downstream_labels") is not False
        or payload.get("outer_test_opened") is not False
    ):
        raise PermissionError("v2 locked retraining requires a clean locked configuration")
    names = {field.name for field in fields(ChronarisV2CandidateConfig)}
    candidate_payload = payload.get("candidate")
    if not isinstance(candidate_payload, dict):
        raise PermissionError("v2 locked configuration has no candidate")
    candidate = ChronarisV2CandidateConfig(
        **{name: candidate_payload[name] for name in names if name in candidate_payload}
    )
    return candidate, payload


def _check(name: str, passed: bool):
    return {"check": name, "passed": bool(passed)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
