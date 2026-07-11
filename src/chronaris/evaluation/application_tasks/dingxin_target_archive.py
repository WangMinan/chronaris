"""Build deterministic weak-supervision targets for the fixed Dingxin data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from chronaris.dataset.application_evaluation.labels import (
    MANEUVER_TASK_ID,
    RESPONSE_TASK_ID,
)
from chronaris.evaluation.application_tasks.dingxin_target_audit import (
    build_dingxin_target_acceptance_rows,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    build_raw_median_response_targets,
    load_dingxin_target_source_data,
)
from chronaris.evaluation.application_tasks.dingxin_target_reporting import (
    write_dingxin_target_outputs,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    sha256_file,
    write_deterministic_npz,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_target_archive")
LOGGER.addHandler(logging.NullHandler())
CLASS_TO_INDEX = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True, slots=True)
class DingxinTargetArchiveConfig:
    run_id: str = "2026-07-11_dingxin-application-targets"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinTargetArchiveResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    archive_count: int
    classification_unique_context_count: int
    response_available_unique_context_count: int
    response_unavailable_unique_context_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_target_archive(
    config: DingxinTargetArchiveConfig,
) -> DingxinTargetArchiveResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_application_target_archives",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "human_expert_labels_available": False,
        },
    ) as progress:
        source = load_dingxin_target_source_data(
            fixed_audit_root=config.fixed_audit_root,
            snapshot_root=config.snapshot_root,
        )
        snapshot_hashes_before = {
            row["path"]: row["sha256"] for row in source.verified_snapshot_files
        }
        classification = _build_classification_rows(source)
        response_result = build_raw_median_response_targets(
            source,
            snapshot_root=config.snapshot_root,
        )
        progress.update(
            "target_rows_built",
            classification_row_count=len(classification),
            response_row_count=len(response_result.label_rows),
            response_available_count=int(
                response_result.label_rows["status"].eq("completed").sum()
            ),
        )
        classification_thresholds = source.audit_thresholds[
            source.audit_thresholds["task_id"] == MANEUVER_TASK_ID
        ].copy()
        classification_thresholds["source_contract"] = (
            "g1_fold_fitted_aligned_window_statistics"
        )
        response_thresholds = response_result.threshold_rows.copy()
        response_thresholds["source_contract"] = "fixed_raw_point_window_median"
        threshold_rows = pd.concat(
            (classification_thresholds, response_thresholds),
            ignore_index=True,
            sort=False,
        )
        archive_rows = _write_target_archives(
            classification_rows=classification,
            response_rows=response_result.label_rows,
            threshold_rows=threshold_rows,
            output_root=heavy_root / "targets",
            resume=config.resume,
        )
        comparison_rows = _build_response_comparison(source, response_result.label_rows)
        snapshot_hashes_after = {
            row["path"]: sha256_file(row["path"])
            for row in source.verified_snapshot_files
        }
        acceptance_rows = build_dingxin_target_acceptance_rows(
            source=source,
            classification_rows=classification,
            response_result=response_result,
            archive_rows=archive_rows,
            snapshot_hashes_unchanged=(
                snapshot_hashes_before == snapshot_hashes_after
            ),
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = write_dingxin_target_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            source_manifest={
                "format": "chronaris.dingxin_target_sources.v1",
                "source_hashes": source.source_hashes,
                "verified_snapshot_files": list(source.verified_snapshot_files),
                "snapshot_hashes_after": snapshot_hashes_after,
                "fold_count": len(source.split_manifest["split_protocols"]),
                "classification_source_contract": (
                    "g1_fold_fitted_aligned_window_statistics"
                ),
                "response_source_contract": "fixed_raw_point_window_median",
                "human_expert_labels_available": False,
            },
            archive_rows=archive_rows,
            availability_rows=response_result.availability_rows.to_dict("records"),
            threshold_rows=threshold_rows.to_dict("records"),
            comparison_rows=comparison_rows,
            acceptance_rows=acceptance_rows,
            heavy_run_root=str(heavy_root),
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        response_available = response_result.label_rows[
            response_result.label_rows["status"] == "completed"
        ]["context_id"].nunique()
        response_unavailable = response_result.label_rows[
            response_result.label_rows["status"] != "completed"
        ]["context_id"].nunique()
        progress.finish(
            status=status,
            archive_count=len(archive_rows),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            response_available_unique_context_count=int(response_available),
        )
        return DingxinTargetArchiveResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            archive_count=len(archive_rows),
            classification_unique_context_count=classification[
                "context_id"
            ].nunique(),
            response_available_unique_context_count=int(response_available),
            response_unavailable_unique_context_count=int(response_unavailable),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _build_classification_rows(source):
    labels = source.audit_labels[
        source.audit_labels["task_id"] == MANEUVER_TASK_ID
    ].copy()
    contexts = source.contexts[
        [
            "context_id",
            "sortie_id",
            "view_id",
            "pilot_id",
            "start_offset_ms",
            "end_offset_ms",
        ]
    ]
    labels = labels.merge(contexts, on="context_id", how="left", validate="many_to_one")
    if labels["sortie_id"].isna().any():
        raise ValueError("classification target lacks context lineage")
    labels["representative_statistic"] = "aligned_window_std_and_delta"
    labels["input_start_offset_ms"] = labels["start_offset_ms"].astype(int)
    labels["input_end_exclusive_ms"] = labels["end_offset_ms"].astype(int)
    labels["target_start_offset_ms"] = labels["end_offset_ms"].astype(int) - 5_000
    labels["target_end_exclusive_ms"] = labels["end_offset_ms"].astype(int)
    labels["weak_supervision"] = True
    labels["label_source_fields_allowed_in_input"] = False
    return labels


def _write_target_archives(
    *, classification_rows, response_rows, threshold_rows, output_root, resume
):
    root = Path(output_root)
    rows = []
    task_frames = {
        "maneuver_intensity_classification": classification_rows,
        "physiology_response_prediction": response_rows,
    }
    for task_slug, frame in task_frames.items():
        for fold_id, fold_frame in frame.groupby("fold_id", sort=True):
            destination = root / fold_id / task_slug
            destination.mkdir(parents=True, exist_ok=True)
            archive_path = destination / "targets.npz"
            threshold_path = destination / "thresholds.json"
            previous_archive_hash = (
                sha256_file(archive_path) if resume and archive_path.is_file() else None
            )
            previous_threshold_hash = (
                sha256_file(threshold_path) if resume and threshold_path.is_file() else None
            )
            ordered = fold_frame.sort_values(
                ["split_role", "context_id"], kind="mergesort"
            ).reset_index(drop=True)
            archive_hash = write_deterministic_npz(
                archive_path,
                _archive_payload(ordered, task_slug=task_slug),
            )
            matching_thresholds = threshold_rows[
                threshold_rows["fold_id"] == fold_id
            ]
            if task_slug == "maneuver_intensity_classification":
                matching_thresholds = matching_thresholds[
                    matching_thresholds["task_id"] == MANEUVER_TASK_ID
                ]
            else:
                matching_thresholds = matching_thresholds[
                    matching_thresholds["task_id"].astype(str).str.endswith(
                        "_raw_median_v1"
                    )
                ]
            threshold_path.write_text(
                json.dumps(
                    {
                        "format": "chronaris.dingxin_target_thresholds.v1",
                        "fold_id": fold_id,
                        "task_slug": task_slug,
                        "rows": _json_records(matching_thresholds),
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            threshold_hash = sha256_file(threshold_path)
            rewrite_hash = write_deterministic_npz(
                archive_path,
                _archive_payload(ordered, task_slug=task_slug),
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "task_slug": task_slug,
                    "status": (
                        "resumed"
                        if previous_archive_hash == archive_hash
                        and previous_threshold_hash == threshold_hash
                        else "completed"
                    ),
                    "row_count": len(ordered),
                    "available_count": int(ordered["status"].eq("completed").sum()),
                    "unavailable_count": int(
                        (~ordered["status"].eq("completed")).sum()
                    ),
                    "train_count": int(ordered["split_role"].eq("train").sum()),
                    "test_count": int(ordered["split_role"].eq("test").sum()),
                    "archive_path": str(archive_path),
                    "archive_sha256": archive_hash,
                    "threshold_path": str(threshold_path),
                    "threshold_sha256": threshold_hash,
                    "hash_stable_on_rewrite": archive_hash == rewrite_hash,
                }
            )
    return rows


def _archive_payload(frame, *, task_slug):
    count = len(frame)
    class_target = np.full(count, -1, dtype=np.int64)
    continuous_target = np.full(count, np.nan, dtype=np.float32)
    binary_target = np.full(count, -1, dtype=np.int64)
    if task_slug == "maneuver_intensity_classification":
        class_target = frame["class_label"].map(CLASS_TO_INDEX).fillna(-1).to_numpy(np.int64)
    else:
        continuous_target = frame["continuous_target"].to_numpy(np.float32)
        binary_target = frame["high_response_label"].fillna(-1).to_numpy(np.int64)
    return {
        "context_ids": np.asarray(frame["context_id"].astype(str).tolist(), dtype=str),
        "split_roles": np.asarray(frame["split_role"].astype(str).tolist(), dtype=str),
        "statuses": np.asarray(frame["status"].astype(str).tolist(), dtype=str),
        "class_target": class_target,
        "continuous_target": continuous_target,
        "binary_target": binary_target,
        "valid_field_count": frame["valid_field_count"].fillna(0).to_numpy(np.int64),
        "fit_sample_hashes": np.asarray(
            frame["fit_sample_hash"].astype(str).tolist(), dtype=str
        ),
        "input_start_offset_ms": frame["input_start_offset_ms"].to_numpy(np.int64),
        "input_end_exclusive_ms": frame["input_end_exclusive_ms"].to_numpy(np.int64),
        "target_start_offset_ms": frame["target_start_offset_ms"].to_numpy(np.int64),
        "target_end_exclusive_ms": frame["target_end_exclusive_ms"].to_numpy(np.int64),
        "representative_statistics": np.asarray(
            frame["representative_statistic"].astype(str).tolist(), dtype=str
        ),
    }


def _build_response_comparison(source, response_rows):
    audit = source.audit_labels[
        source.audit_labels["task_id"] == RESPONSE_TASK_ID
    ][["fold_id", "split_role", "context_id", "continuous_target", "status"]].rename(
        columns={
            "continuous_target": "audit_window_mean_target",
            "status": "audit_status",
        }
    )
    raw = response_rows[
        ["fold_id", "split_role", "context_id", "continuous_target", "status"]
    ].rename(
        columns={
            "continuous_target": "raw_window_median_target",
            "status": "raw_status",
        }
    )
    merged = audit.merge(
        raw,
        on=["fold_id", "split_role", "context_id"],
        validate="one_to_one",
    )
    rows = []
    for fold_id, frame in merged.groupby("fold_id", sort=True):
        paired = frame[
            frame["audit_window_mean_target"].notna()
            & frame["raw_window_median_target"].notna()
        ]
        correlation = spearmanr(
            paired["audit_window_mean_target"],
            paired["raw_window_median_target"],
        ).statistic
        rows.append(
            {
                "fold_id": fold_id,
                "candidate_row_count": len(frame),
                "paired_available_count": len(paired),
                "raw_unavailable_count": int(
                    frame["raw_window_median_target"].isna().sum()
                ),
                "mean_absolute_difference": float(
                    np.mean(
                        np.abs(
                            paired["audit_window_mean_target"]
                            - paired["raw_window_median_target"]
                        )
                    )
                ),
                "spearman": float(correlation),
                "comparison_only": True,
            }
        )
    return rows


def _json_records(frame):
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _json_value(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
