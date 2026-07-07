"""Artifact helpers for Dingxin deep baseline E3 representation export."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from chronaris.evaluation.fusion_stream_structure.contracts import ContractError, validate_input_frame
from chronaris.evaluation.fusion_stream_structure.dataset_loader import build_fusion_streams_from_dingxin_artifacts
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_types import (
    FORBIDDEN_REPRESENTATION_COLUMNS,
    REPO_ROOT,
    DeepBaselineRepresentationExportConfig,
    DeepBaselineRepresentationExportResult,
)
from chronaris.modeling.common.gpu_runtime import iter_eval_batches, resolve_amp_runtime


def write_deep_baseline_representation_long_table(rows: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = derive_weak_event_boundaries(pd.DataFrame(list(rows)))
    validate_deep_baseline_representation_frame(frame)
    frame.to_csv(output_path, index=False)
    return output_path


def load_deep_baseline_representation_long_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_deep_baseline_representation_frame(frame)
    return frame


def rows_from_pooled_embeddings(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    config: DeepBaselineRepresentationExportConfig,
    model_name: str,
    fold_index: int,
    fold_group: str,
    checkpoint_path: str | Path,
    train_sample_count: int,
    test_sample_count: int,
) -> list[dict[str, object]]:
    if len(frame) != int(np.asarray(embeddings).shape[0]):
        raise ValueError("embedding row count must match held-out frame row count")
    rows: list[dict[str, object]] = []
    for item, vector in zip(frame.itertuples(index=False), np.asarray(embeddings), strict=True):
        row = {
            "method_name": str(model_name),
            "sortie_id": str(item.sortie_id),
            "view_id": str(item.view_id),
            "window_id": str(item.sample_id),
            "time": float(getattr(item, "start_offset_ms", getattr(item, "window_index", 0))),
            "window_index": int(item.window_index),
            "pilot_id": int(item.pilot_id),
            "sample_partition": getattr(item, "sample_partition", None),
            "representation_family": config.representation_family,
            "source_model_name": str(model_name),
            "source_task_name": config.task_name,
            "source_task_type": config.task_type,
            "source_split_strategy": config.split_strategy,
            "source_seed": int(config.seed),
            "source_fold_index": int(fold_index),
            "source_fold_group": str(fold_group),
            "source_checkpoint_path": repo_rel(checkpoint_path),
            "source_train_sample_count": int(train_sample_count),
            "source_test_sample_count": int(test_sample_count),
            "source_e_run_manifest_path": config.e_run_manifest_path,
            "source_f_run_manifest_path": config.f_run_manifest_path,
            "maneuver_proxy_label": getattr(item, "maneuver_proxy_label", None),
            "physio_fluctuation_interval": bool(getattr(item, "physio_fluctuation_interval", False)),
        }
        for feature_index, value in enumerate(np.asarray(vector, dtype=np.float32).reshape(-1), start=1):
            row[f"fusion_feature_{feature_index}"] = float(value)
        rows.append(row)
    return rows


def extract_pooled_embeddings_for_indices(model, prepared, indices: np.ndarray, *, batch_size: int, amp_mode: str = "off") -> np.ndarray:
    device_name = next(model.parameters()).device.type
    amp = resolve_amp_runtime(requested_mode=amp_mode, device=device_name, grad_scaler=False)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for modality_batch, mask_batch, time_batch, _target_tensor in iter_eval_batches(
            prepared,
            np.asarray(indices, dtype=int),
            batch_size=batch_size,
            device=device_name,
        ):
            with amp.autocast(device=device_name):
                output = model(modality_batch, time_axis=time_batch, modality_masks=mask_batch)
            if getattr(output, "pooled_embedding", None) is None:
                raise ValueError("deep baseline model output has no pooled_embedding")
            chunks.append(output.pooled_embedding.detach().float().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32) if chunks else np.empty((0, 0), dtype=np.float32)


def validate_deep_baseline_representation_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ContractError("deep baseline representation table is empty")
    required = {
        "method_name",
        "sortie_id",
        "view_id",
        "window_id",
        "time",
        "pilot_id",
        "sample_partition",
        "representation_family",
        "source_model_name",
        "source_task_name",
        "source_task_type",
        "source_split_strategy",
        "source_seed",
        "source_fold_index",
        "source_fold_group",
        "source_checkpoint_path",
        "source_train_sample_count",
        "source_test_sample_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError("deep baseline representation table is missing required columns", details={"missing_columns": missing})
    if frame["representation_family"].isna().any() or (frame["representation_family"].astype(str).str.len() == 0).any():
        raise ContractError("representation_family is required for every OOF embedding row")
    lower_columns = {str(column).lower() for column in frame.columns}
    forbidden = sorted(column for column in lower_columns if any(column == token or column.startswith(token + "_") for token in FORBIDDEN_REPRESENTATION_COLUMNS))
    if forbidden:
        raise ContractError("forbidden prediction/diagnostic columns are present in the E3 representation table", details={"forbidden_columns": forbidden})
    validate_input_frame(frame)


def validate_checkpoint_manifest_frame(frame: pd.DataFrame) -> None:
    required = {
        "model_name",
        "task_name",
        "task_type",
        "split_strategy",
        "seed",
        "fold_index",
        "fold_group",
        "checkpoint_path",
        "representation_family",
        "source_e_run_manifest_path",
        "source_f_run_manifest_path",
        "training_invoked",
        "confirmed_metrics_changed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError("checkpoint manifest is missing required columns", details={"missing_columns": missing})


def build_four_method_e3_input_table(
    *,
    deep_representation_table_path: str | Path,
    e_run_manifest_path: str | Path,
    f_run_manifest_path: str | Path,
    output_path: str | Path,
    filter_existing_methods_to_deep_windows: bool = True,
) -> Path:
    deep_frame = load_deep_baseline_representation_long_table(deep_representation_table_path)
    existing = build_fusion_streams_from_dingxin_artifacts(
        e_run_manifest_path=e_run_manifest_path,
        f_run_manifest_path=f_run_manifest_path,
        methods=("chronaris", "naive_time_sync"),
        max_groups=None,
        stage_g_device="cpu",
    )
    base = pd.concat([record.frame.copy() for record in existing.records.values()], ignore_index=True) if existing.records else pd.DataFrame()
    if filter_existing_methods_to_deep_windows and not base.empty:
        base = base[base["window_id"].astype(str).isin(set(deep_frame["window_id"].astype(str)))].copy()
    combined = derive_weak_event_boundaries(pd.concat([base, deep_frame], ignore_index=True, sort=False))
    validate_input_frame(combined)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    output.with_name("combined_four_method_e3_input_manifest.json").write_text(
        json.dumps(
            {
                "source_type": "dingxin_four_method_e3_input",
                "deep_representation_table_path": repo_rel(deep_representation_table_path),
                "e_run_manifest_path": repo_rel(e_run_manifest_path),
                "f_run_manifest_path": repo_rel(f_run_manifest_path),
                "row_count": int(len(combined)),
                "method_counts": {str(k): int(v) for k, v in combined["method_name"].value_counts().sort_index().items()},
                "training_invoked": False,
                "confirmed_metrics_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def write_training_protocol(run_root: Path, config: DeepBaselineRepresentationExportConfig) -> Path:
    path = run_root / "training_protocol.json"
    path.write_text(
        json.dumps(
            {
                **asdict(config),
                "protocol": {
                    "oof": True,
                    "held_out_unit": "view_id",
                    "embedding": "pooled_embedding",
                    "task_label_usage": "train_fold_supervision_only; labels posthoc for E3 only",
                },
                "training_invoked": True,
                "confirmed_metrics_changed": False,
                "source_manifest_paths": source_manifest_paths(config),
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_resume_command(run_root: Path, config: DeepBaselineRepresentationExportConfig) -> Path:
    path = run_root / "resume_command.txt"
    path.write_text(resume_command(config) + "\n", encoding="utf-8")
    return path


def write_export_outputs(
    *,
    config: DeepBaselineRepresentationExportConfig,
    run_root: Path,
    runtime_device: str,
    sequence_frame: pd.DataFrame,
    build_context: Mapping[str, object],
    embedding_rows: Sequence[Mapping[str, object]],
    curve_rows: Sequence[Mapping[str, object]],
    checkpoint_rows: Sequence[Mapping[str, object]],
    fold_status_rows: Sequence[Mapping[str, object]],
    expected_fold_count: int,
) -> DeepBaselineRepresentationExportResult:
    paths = _export_paths(run_root)
    representation_frame = pd.DataFrame()
    if embedding_rows:
        write_deep_baseline_representation_long_table(embedding_rows, paths["representation"])
        representation_frame = pd.read_csv(paths["representation"])
    else:
        representation_frame.to_csv(paths["representation"], index=False)
    checkpoint_frame = pd.DataFrame(list(checkpoint_rows))
    if not checkpoint_frame.empty:
        validate_checkpoint_manifest_frame(checkpoint_frame)
    checkpoint_frame.to_csv(paths["checkpoint_csv"], index=False)
    paths["checkpoint_json"].write_text(json.dumps(list(checkpoint_rows), ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    pd.DataFrame(list(curve_rows)).to_csv(paths["curves"], index=False)
    fold_status_frame = pd.DataFrame(list(fold_status_rows))
    fold_status_frame.to_csv(paths["fold_status"], index=False)
    completed = int((fold_status_frame["status"].isin(["completed", "completed_skipped"])).sum()) if not fold_status_frame.empty else 0
    expected_groups = set(sequence_frame["view_id"].astype(str).unique())
    model_status = {
        str(model): (
            not representation_frame.empty
            and set(representation_frame.loc[representation_frame["method_name"] == model, "source_fold_group"].astype(str)) == expected_groups
            and bool((fold_status_frame["model_name"] == model).any())
            and bool((fold_status_frame.loc[fold_status_frame["model_name"] == model, "status"].isin(["completed", "completed_skipped"])).all())
        )
        for model in config.models
    }
    status = "completed" if completed == int(expected_fold_count) and all(model_status.values()) else "partial"
    summary = _summary_payload(config, status, runtime_device, sequence_frame, representation_frame, fold_status_frame, completed, expected_fold_count, model_status)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    paths["representation_manifest"].write_text(
        json.dumps({**summary, "representation_table_path": repo_rel(paths["representation"]), "checkpoint_manifest_csv": repo_rel(paths["checkpoint_csv"]), "checkpoint_manifest_json": repo_rel(paths["checkpoint_json"]), "oof_protocol": "held-out leave-one-view-out inference only"}, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    blockers = write_blockers(run_root, summary, config) if status != "completed" else None
    manifest = write_evidence_manifest(run_root, config=config, summary=summary, build_context=build_context, output_files=[*paths.values(), run_root / "training_protocol.json", run_root / "resume_command.txt", run_root / "run.log", run_root / "progress.json"], blockers_path=blockers)
    report = write_report(run_root, config=config, summary=summary, blockers_path=blockers)
    return DeepBaselineRepresentationExportResult(config.run_id, str(run_root), status, True, False, str(paths["representation"]), str(paths["checkpoint_csv"]), str(report), str(manifest), completed, int(expected_fold_count), model_status, str(blockers) if blockers else None)


def write_blocked_result(config: DeepBaselineRepresentationExportConfig, run_root: Path, *, reason: str, details: Mapping[str, object] | None = None) -> DeepBaselineRepresentationExportResult:
    summary = {
        "run_id": config.run_id,
        "status": "blocked",
        "blocked_reason": reason,
        "details": dict(details or {}),
        "training_invoked": False if reason == "cuda_unavailable" else True,
        "metrics_changed": False,
        "confirmed_metrics_changed": False,
        "thesis_protocol_snapshot_modified": False,
        "t3_artifact_deleted": False,
        "uses_stage_or_final_naming": False,
        "representation_family": config.representation_family,
        "task_name": config.task_name,
        "split_strategy": config.split_strategy,
        "seed": int(config.seed),
        "epochs": int(config.epochs),
        "resume_command": resume_command(config),
    }
    (run_root / "deep_baseline_oof_embeddings_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    blockers = write_blockers(run_root, summary, config)
    manifest = write_evidence_manifest(run_root, config=config, summary=summary, build_context={}, output_files=[run_root / "deep_baseline_oof_embeddings_summary.json", blockers, run_root / "training_protocol.json", run_root / "resume_command.txt"], blockers_path=blockers)
    report = write_report(run_root, config=config, summary=summary, blockers_path=blockers)
    return DeepBaselineRepresentationExportResult(config.run_id, str(run_root), "blocked", bool(summary["training_invoked"]), False, None, None, str(report), str(manifest), 0, 0, {model: False for model in config.models}, str(blockers))


def write_evidence_manifest(run_root: Path, *, config: DeepBaselineRepresentationExportConfig, summary: Mapping[str, object], build_context: Mapping[str, object], output_files: Sequence[Path], blockers_path: Path | None) -> Path:
    path = run_root / "evidence_manifest.json"
    path.write_text(
        json.dumps(
            {
                "run_id": config.run_id,
                "run_type": "deep_baseline_representation_export",
                "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
                "commit": git_value("rev-parse", "HEAD"),
                "created_at_utc": utc_now(),
                "training_invoked": bool(summary.get("training_invoked", True)),
                "metrics_changed": False,
                "confirmed_metrics_changed": False,
                "thesis_protocol_snapshot_modified": False,
                "t3_artifact_deleted": False,
                "uses_stage_or_final_naming": False,
                "representation_family": config.representation_family,
                "task_name": config.task_name,
                "task_type": config.task_type,
                "split_strategy": config.split_strategy,
                "seed": int(config.seed),
                "models": list(config.models),
                "source_manifest_paths": source_manifest_paths(config),
                "summary": dict(summary),
                "sequence_source_record_count": build_context.get("source_record_count"),
                "output_files": [repo_rel(path) for path in output_files if path.exists()],
                "blockers_path": repo_rel(blockers_path) if blockers_path else None,
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_report(run_root: Path, *, config: DeepBaselineRepresentationExportConfig, summary: Mapping[str, object], blockers_path: Path | None) -> Path:
    path = run_root / "report.md"
    model_lines = [f"- `{model}`: OOF embedding available = `{available}`" for model, available in sorted((summary.get("method_embedding_status") or {}).items())]
    lines = [
        "# Deep Baseline Representation Export for E3",
        "",
        f"- run_id: `{config.run_id}`",
        f"- status: `{summary.get('status')}`",
        "- 评价定位：为 Dingxin E3 结构诊断补齐 MulT / ContiFormer 的 held-out pooled embedding 流。",
        f"- training_invoked: `{str(summary.get('training_invoked')).lower()}`",
        "- confirmed_metrics_changed: `false`",
        "- thesis_protocol_snapshot_modified: `false`",
        f"- representation_family: `{config.representation_family}`",
        f"- task: `{config.task_name}` / `{config.task_type}`",
        f"- split: `{config.split_strategy}`",
        f"- seed: `{config.seed}`",
        f"- epochs: `{config.epochs}`",
        "",
        "## OOF 协议",
        "",
        "每个 `fusion_feature_*` 行来自该样本所属 held-out view 的 inference。训练只使用 train fold 的 T2 标签；导出的 E3 主输入只使用 `pooled_embedding`，不包含 logits、预测值、rank 或诊断标量。",
        "",
        "## 模型状态",
        "",
        *model_lines,
        "",
        "## 输出",
        "",
        "- `deep_baseline_oof_embeddings_long.csv`",
        "- `deep_baseline_oof_embeddings_summary.json`",
        "- `checkpoint_manifest.csv` / `checkpoint_manifest.json`",
        "- `representation_manifest.json`",
        "- `training_curves.csv`",
        "- `fold_status.csv`",
        "",
        "## 边界",
        "",
        "该产物只作为 E3 融合表示流结构评价的输入，不回写分类任务、回归任务或检索任务 confirmed metrics，也不修改论文协议快照。E3 结果是否进入论文仍需后续人工 review。",
    ]
    if blockers_path is not None:
        lines.extend(["", "## Blockers", "", f"详见 `{blockers_path.name}`。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_blockers(run_root: Path, summary: Mapping[str, object], config: DeepBaselineRepresentationExportConfig) -> Path:
    path = run_root / "blockers.md"
    path.write_text(
        "\n".join(
            [
                "# Deep Baseline Representation Export Blockers",
                "",
                f"- status: `{summary.get('status')}`",
                f"- reason: `{summary.get('blocked_reason') or summary.get('fold_status_counts')}`",
                f"- completed folds: `{summary.get('completed_fold_count', 0)}` / `{summary.get('expected_fold_count', 0)}`",
                f"- resume command: `{resume_command(config)}`",
                "",
                "当前没有伪造四方法 E3；只有当 MulT 与 ContiFormer 均完成 OOF pooled embedding 后，才应运行四方法 Dingxin E3 validation。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def derive_weak_event_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "maneuver_proxy_label" not in frame.columns:
        frame = frame.copy()
        frame["weak_event_boundary"] = False
        return frame
    rows = []
    for _key, group in frame.groupby(["method_name", "sortie_id", "view_id"], sort=False):
        ordered = group.sort_values(["time", "window_id"], kind="mergesort").copy()
        labels = ordered["maneuver_proxy_label"].astype(str).to_list()
        boundaries = [False]
        for previous, current in zip(labels[:-1], labels[1:]):
            boundaries.append(previous != current and previous not in {"None", "nan"} and current not in {"None", "nan"})
        ordered["weak_event_boundary"] = boundaries
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True)


def checkpoint_path(run_root: Path, model_name: str, seed: int, fold_index: int) -> Path:
    return run_root / "checkpoints" / model_name / f"seed{seed}" / f"fold{fold_index:03d}" / "checkpoint.pt"


def fold_embedding_path(run_root: Path, model_name: str, seed: int, fold_index: int) -> Path:
    return run_root / "fold_embeddings" / model_name / f"seed{seed}" / f"fold{fold_index:03d}.csv"


def source_manifest_paths(config: DeepBaselineRepresentationExportConfig) -> dict[str, str]:
    return {"e_run_manifest_path": config.e_run_manifest_path, "f_run_manifest_path": config.f_run_manifest_path}


def modality_input_dims(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {name: int(values.shape[-1]) for name, values in arrays.items()}


def checkpoint_manifest_row(config: DeepBaselineRepresentationExportConfig, model_name: str, fold_index: int, fold_group: str, checkpoint: Path, train_idx: np.ndarray, test_idx: np.ndarray, frame: pd.DataFrame, runtime_device: str, dims: Mapping[str, int], status: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "task_name": config.task_name,
        "task_type": config.task_type,
        "split_strategy": config.split_strategy,
        "seed": int(config.seed),
        "fold_index": int(fold_index),
        "fold_group": str(fold_group),
        "checkpoint_path": repo_rel(checkpoint),
        "representation_family": config.representation_family,
        "source_e_run_manifest_path": config.e_run_manifest_path,
        "source_f_run_manifest_path": config.f_run_manifest_path,
        "runtime_device": runtime_device,
        "modality_input_dims": json.dumps(dict(dims), ensure_ascii=False, sort_keys=True),
        "train_sample_count": int(len(train_idx)),
        "test_sample_count": int(len(test_idx)),
        "train_sample_ids": json.dumps(frame.iloc[train_idx]["sample_id"].astype(str).to_list(), ensure_ascii=False),
        "test_sample_ids": json.dumps(frame.iloc[test_idx]["sample_id"].astype(str).to_list(), ensure_ascii=False),
        "training_invoked": True,
        "confirmed_metrics_changed": False,
        "status": status,
    }


def fold_status_row(config: DeepBaselineRepresentationExportConfig, model_name: str, fold_index: int, fold_group: str, train_idx: np.ndarray, test_idx: np.ndarray, status: str, *, checkpoint: str | Path | None = None, embedding_row_count: int = 0, reason: str | None = None, traceback_text: str | None = None, elapsed_s: float | None = None, metrics: Mapping[str, object] | None = None) -> dict[str, object]:
    row = {
        "model_name": model_name,
        "task_name": config.task_name,
        "task_type": config.task_type,
        "split_strategy": config.split_strategy,
        "seed": int(config.seed),
        "fold_index": int(fold_index),
        "fold_group": str(fold_group),
        "train_sample_count": int(len(train_idx)),
        "test_sample_count": int(len(test_idx)),
        "status": status,
        "checkpoint_path": repo_rel(checkpoint) if checkpoint else None,
        "embedding_row_count": int(embedding_row_count),
        "reason": reason,
        "traceback": traceback_text,
        "elapsed_s": elapsed_s,
        "representation_family": config.representation_family,
        "training_invoked": True,
        "confirmed_metrics_changed": False,
    }
    if metrics:
        row.update({f"metric_{key}": value for key, value in metrics.items()})
    return row


def model_hyperparameters(config: DeepBaselineRepresentationExportConfig) -> dict[str, object]:
    return {
        "hidden_dim": int(config.hidden_dim),
        "num_heads": int(config.num_heads),
        "layers": int(config.layers),
        "dropout": float(config.dropout),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "grad_clip_norm": float(config.grad_clip_norm),
        "batch_size": int(config.batch_size),
        "auto_batch_size": bool(config.auto_batch_size),
        "amp": config.amp,
        "torch_compile": config.torch_compile,
    }


def resume_command(config: DeepBaselineRepresentationExportConfig) -> str:
    return f"{sys.executable} scripts/evaluation/fusion_stream_structure/run_deep_baseline_representation_export.py --run-id {config.run_id} --models {' '.join(config.models)} --epochs {config.epochs} --seed {config.seed} --split-strategy {config.split_strategy} --resume --skip-completed"


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def repo_rel(path_like: str | Path | None) -> str | None:
    if path_like is None:
        return None
    path = Path(path_like)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_value(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    except Exception as exc:  # pragma: no cover
        return f"unavailable:{exc!r}"
    return result.stdout.strip()


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def _export_paths(run_root: Path) -> dict[str, Path]:
    return {
        "representation": run_root / "deep_baseline_oof_embeddings_long.csv",
        "summary": run_root / "deep_baseline_oof_embeddings_summary.json",
        "representation_manifest": run_root / "representation_manifest.json",
        "checkpoint_csv": run_root / "checkpoint_manifest.csv",
        "checkpoint_json": run_root / "checkpoint_manifest.json",
        "curves": run_root / "training_curves.csv",
        "fold_status": run_root / "fold_status.csv",
    }


def _summary_payload(config: DeepBaselineRepresentationExportConfig, status: str, runtime_device: str, sequence_frame: pd.DataFrame, representation_frame: pd.DataFrame, fold_status_frame: pd.DataFrame, completed: int, expected_fold_count: int, model_status: Mapping[str, bool]) -> dict[str, object]:
    return {
        "run_id": config.run_id,
        "status": status,
        "training_invoked": True,
        "metrics_changed": False,
        "confirmed_metrics_changed": False,
        "thesis_protocol_snapshot_modified": False,
        "t3_artifact_deleted": False,
        "uses_stage_or_final_naming": False,
        "representation_family": config.representation_family,
        "task_name": config.task_name,
        "task_type": config.task_type,
        "split_strategy": config.split_strategy,
        "seed": int(config.seed),
        "epochs": int(config.epochs),
        "runtime_device": runtime_device,
        "sequence_sample_count": int(len(sequence_frame)),
        "embedding_row_count": int(len(representation_frame)),
        "method_embedding_status": dict(model_status),
        "completed_fold_count": completed,
        "expected_fold_count": int(expected_fold_count),
        "fold_status_counts": dict(fold_status_frame["status"].value_counts().sort_index()) if not fold_status_frame.empty else {},
        "method_counts": dict(representation_frame["method_name"].value_counts().sort_index()) if not representation_frame.empty else {},
        "feature_dimensions": _feature_dimensions(representation_frame),
        "source_manifest_paths": source_manifest_paths(config),
    }


def _feature_dimensions(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    features = [column for column in frame.columns if str(column).startswith("fusion_feature_")]
    return {str(method): int(frame.loc[frame["method_name"] == method, features].dropna(axis=1, how="all").shape[1]) for method in sorted(frame["method_name"].astype(str).unique())}
