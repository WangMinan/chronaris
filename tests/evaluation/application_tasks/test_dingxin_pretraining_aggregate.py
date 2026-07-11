from __future__ import annotations

from chronaris.evaluation.application_tasks.dingxin_pretraining_aggregate import (
    EXPECTED_FOLD_IDS,
    EXPECTED_METHODS,
    _build_acceptance_rows,
)


def _fixtures():
    fold_rows = [
        {
            "fold_id": fold_id,
            "status": "completed",
            "train_count": 3,
            "validation_count": 2,
            "held_out_count": 1,
            "train_sample_hash": f"hash::{fold_id}",
            "acceptance_pass_count": 12,
            "acceptance_check_count": 12,
            "resume_reused_count": 18,
            "downstream_targets_opened": False,
            "outer_test_metrics_opened": False,
        }
        for fold_id in EXPECTED_FOLD_IDS
    ]
    checkpoint_rows = [
        {
            "fold_id": fold["fold_id"],
            "method_name": method,
            "fit_sample_hash": fold["train_sample_hash"],
            "label_used_for_encoder_training": False,
        }
        for fold in fold_rows
        for method in EXPECTED_METHODS
    ]
    export_rows = [
        {
            "fold_id": fold["fold_id"],
            "method_name": method,
            "export_role": role,
            "sample_count": fold[f"{role}_count"],
            "archive_valid": True,
        }
        for fold in fold_rows
        for method in EXPECTED_METHODS
        for role in ("train", "validation", "held_out")
    ]
    resource_rows = [{"maximum_rss_mb": 2000.0}]
    source_rows = [
        {"schema_sha256": "a" * 64} for _fold_id in EXPECTED_FOLD_IDS
    ]
    return fold_rows, checkpoint_rows, export_rows, resource_rows, source_rows


def test_five_fold_aggregate_accepts_complete_method_role_matrix():
    rows = _build_acceptance_rows(
        fold_rows=_fixtures()[0],
        checkpoint_rows=_fixtures()[1],
        export_rows=_fixtures()[2],
        resource_rows=_fixtures()[3],
        source_rows=_fixtures()[4],
    )

    assert len(rows) == 13
    assert all(row["passed"] for row in rows)


def test_five_fold_aggregate_rejects_bad_checkpoint_fit_hash():
    fold_rows, checkpoints, exports, resources, sources = _fixtures()
    checkpoints[0]["fit_sample_hash"] = "wrong"

    rows = _build_acceptance_rows(
        fold_rows=fold_rows,
        checkpoint_rows=checkpoints,
        export_rows=exports,
        resource_rows=resources,
        source_rows=sources,
    )

    failed = {row["check_id"] for row in rows if not row["passed"]}
    assert failed == {"checkpoint_fit_hashes_match_inner_train"}
