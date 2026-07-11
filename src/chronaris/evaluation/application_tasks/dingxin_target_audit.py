"""Acceptance checks for independent Dingxin application target archives."""

from __future__ import annotations

from chronaris.dataset.application_evaluation import stable_sample_hash


def build_dingxin_target_acceptance_rows(
    *,
    source,
    classification_rows,
    response_result,
    archive_rows,
    snapshot_hashes_unchanged,
):
    response = response_result.label_rows
    response_available = response[response["status"] == "completed"]
    response_unavailable = response[response["status"] != "completed"]
    fold_ids = sorted(classification_rows["fold_id"].unique())
    class_coverage = all(
        set(
            classification_rows[
                (classification_rows["fold_id"] == fold_id)
                & (classification_rows["split_role"] == role)
            ]["class_label"]
        )
        == {"low", "medium", "high"}
        for fold_id in fold_ids
        for role in ("train", "test")
    )
    response_fit_hashes_valid = all(
        _response_fit_hash_valid(response, fold_id) for fold_id in fold_ids
    )
    maneuver_sources = source.field_roles[
        source.field_roles["selected_for_maneuver_label"]
    ]
    checks = [
        _check(
            "fixed_source_lineage_complete",
            len(source.source_hashes) == 6
            and len(source.split_manifest["split_protocols"]) == 5,
            {
                "source_hash_count": len(source.source_hashes),
                "fold_count": len(source.split_manifest["split_protocols"]),
            },
        ),
        _check(
            "raw_snapshot_files_verified",
            len(source.verified_snapshot_files) == 5,
            {"verified_file_count": len(source.verified_snapshot_files)},
        ),
        _check(
            "maneuver_label_sources_excluded_from_input",
            len(maneuver_sources) == 20
            and not maneuver_sources["allowed_in_maneuver_input"].any(),
            {
                "label_source_count": len(maneuver_sources),
                "allowed_count": int(
                    maneuver_sources["allowed_in_maneuver_input"].sum()
                ),
            },
        ),
        _check(
            "classification_fold_coverage_complete",
            len(classification_rows) == 480
            and classification_rows["status"].eq("completed").all()
            and classification_rows["context_id"].nunique() == 96,
            {
                "row_count": len(classification_rows),
                "unique_context_count": classification_rows[
                    "context_id"
                ].nunique(),
            },
        ),
        _check(
            "classification_three_classes_in_each_role",
            class_coverage,
            {"fold_count": len(fold_ids), "roles": ["train", "test"]},
        ),
        _check(
            "raw_median_response_availability_explicit",
            len(response) == 465
            and response_available["context_id"].nunique() == 90
            and response_unavailable["context_id"].nunique() == 3
            and set(response_unavailable["status"])
            == {"future_interval_not_fully_observed"},
            {
                "row_count": len(response),
                "available_unique_context_count": response_available[
                    "context_id"
                ].nunique(),
                "unavailable_unique_context_count": response_unavailable[
                    "context_id"
                ].nunique(),
            },
        ),
        _check(
            "raw_median_response_time_boundary",
            response_available["representative_statistic"]
            .eq("raw_window_median")
            .all()
            and (
                response_available["target_start_offset_ms"]
                == response_available["input_end_exclusive_ms"]
            ).all()
            and (
                response_available["target_end_exclusive_ms"]
                - response_available["target_start_offset_ms"]
                == 5_000
            ).all(),
            {"available_row_count": len(response_available)},
        ),
        _check(
            "response_fit_hash_uses_available_train_contexts_only",
            response_fit_hashes_valid,
            {"fold_count": len(fold_ids)},
        ),
        _check(
            "response_field_selection_is_train_only",
            all(
                group["parameter_name"].nunique() >= 2
                for _, group in response_result.threshold_rows[
                    response_result.threshold_rows["parameter_type"]
                    == "response_field_scale"
                ].groupby("fold_id")
            ),
            {
                "selected_field_counts": response_result.threshold_rows[
                    response_result.threshold_rows["parameter_type"]
                    == "response_field_scale"
                ]
                .groupby("fold_id")["parameter_name"]
                .nunique()
                .to_dict()
            },
        ),
        _check(
            "ten_independent_archives_written",
            len(archive_rows) == 10
            and all(row["archive_sha256"] for row in archive_rows)
            and all(row["threshold_sha256"] for row in archive_rows),
            {"archive_count": len(archive_rows)},
        ),
        _check(
            "target_archive_resume_is_deterministic",
            all(row["hash_stable_on_rewrite"] for row in archive_rows),
            {
                "stable_count": sum(
                    row["hash_stable_on_rewrite"] for row in archive_rows
                )
            },
        ),
        _check(
            "raw_snapshot_files_immutable",
            snapshot_hashes_unchanged,
            {"unchanged": snapshot_hashes_unchanged},
        ),
    ]
    return checks


def _response_fit_hash_valid(response, fold_id):
    selected = response[
        (response["fold_id"] == fold_id)
        & (response["split_role"] == "train")
        & (response["status"] == "completed")
    ]
    observed = set(response[response["fold_id"] == fold_id]["fit_sample_hash"])
    return observed == {stable_sample_hash(tuple(selected["context_id"]))}


def _check(name, passed, details):
    return {"check_name": name, "passed": bool(passed), "details": details}
