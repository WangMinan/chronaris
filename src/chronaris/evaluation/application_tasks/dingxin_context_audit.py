"""Acceptance checks for lazy Dingxin raw-context and target bindings."""

from __future__ import annotations


def build_dingxin_context_acceptance_rows(
    *,
    index,
    stream_rows,
    binding_rows,
    archive_verification,
    schema_payload,
    snapshot_hashes_unchanged,
):
    available_streams = stream_rows[stream_rows["status"] == "completed"]
    unavailable_streams = stream_rows[stream_rows["status"] != "completed"]
    classification = binding_rows[
        binding_rows["task_slug"] == "maneuver_intensity_classification"
    ]
    response = binding_rows[
        binding_rows["task_slug"] == "physiology_response_prediction"
    ]
    excluded_identifiers = set(schema_payload["excluded_feature_names"])
    active_identifiers = {
        f"{sortie_id}:{raw_name}"
        for mapping in (
            index.plan.physiology_raw_to_index,
            index.plan.vehicle_raw_to_index,
        )
        for sortie_id, raw_mapping in mapping.items()
        for raw_name in raw_mapping
    }
    checks = [
        _check(
            "context_catalog_complete",
            len(index.contexts) == 96
            and index.contexts["context_id"].nunique() == 96,
            {"context_count": len(index.contexts)},
        ),
        _check(
            "raw_input_availability_explicit",
            available_streams["context_id"].nunique() == 93
            and unavailable_streams["context_id"].nunique() == 3
            and set(unavailable_streams["reason"])
            == {"input_context_duration_not_30_seconds"},
            {
                "available_count": available_streams["context_id"].nunique(),
                "unavailable_count": unavailable_streams[
                    "context_id"
                ].nunique(),
            },
        ),
        _check(
            "lazy_streams_respect_context_end",
            available_streams["maximum_relative_timestamp_s"].lt(30.0).all()
            and available_streams["physiology_point_count"].gt(0).all()
            and available_streams["vehicle_point_count"].gt(0).all(),
            {
                "maximum_relative_timestamp_s": float(
                    available_streams["maximum_relative_timestamp_s"].max()
                )
            },
        ),
        _check(
            "label_source_fields_absent_from_active_schema",
            schema_payload["excluded_feature_count"] == 20
            and excluded_identifiers.isdisjoint(active_identifiers),
            {
                "excluded_count": schema_payload["excluded_feature_count"],
                "active_raw_mapping_count": len(active_identifiers),
            },
        ),
        _check(
            "common_schema_matches_locked_contract",
            schema_payload["physiology_feature_count"] == 12
            and schema_payload["vehicle_feature_count"] == 955
            and schema_payload["cached_snapshot_file_count"] == 5
            and schema_payload["precomputed_dense_context_bundle"] is False,
            {
                "physiology_feature_count": schema_payload[
                    "physiology_feature_count"
                ],
                "vehicle_feature_count": schema_payload["vehicle_feature_count"],
                "materialization_policy": schema_payload["materialization_policy"],
                "cached_snapshot_file_count": schema_payload[
                    "cached_snapshot_file_count"
                ],
            },
        ),
        _check(
            "classification_binding_uses_complete_inputs_only",
            len(classification) == 480
            and classification[classification["binding_status"] == "available"][
                "context_id"
            ].nunique()
            == 93
            and classification[
                classification["binding_status"]
                == "input_context_duration_not_30_seconds"
            ]["context_id"].nunique()
            == 3,
            {
                "available_unique_context_count": classification[
                    classification["binding_status"] == "available"
                ]["context_id"].nunique()
            },
        ),
        _check(
            "response_binding_requires_complete_future",
            len(response) == 465
            and response[response["binding_status"] == "available"][
                "context_id"
            ].nunique()
            == 90
            and response[response["binding_status"] != "available"][
                "context_id"
            ].nunique()
            == 3,
            {
                "available_unique_context_count": response[
                    response["binding_status"] == "available"
                ]["context_id"].nunique()
            },
        ),
        _check(
            "response_target_starts_after_input",
            (
                response["target_start_offset_ms"]
                == response["input_end_exclusive_ms"]
            ).all(),
            {"row_count": len(response)},
        ),
        _check(
            "outer_fold_groups_are_disjoint",
            _fold_groups_are_disjoint(binding_rows),
            {"fold_count": binding_rows["fold_id"].nunique()},
        ),
        _check(
            "target_archives_verified",
            len(archive_verification) == 10,
            {"verified_archive_count": len(archive_verification)},
        ),
        _check(
            "lazy_context_hashes_complete",
            available_streams["source_sample_hash"].notna().all()
            and available_streams["source_sample_hash"].nunique() == 93,
            {
                "unique_source_hash_count": available_streams[
                    "source_sample_hash"
                ].nunique()
            },
        ),
        _check(
            "snapshot_files_immutable",
            snapshot_hashes_unchanged,
            {"unchanged": snapshot_hashes_unchanged},
        ),
    ]
    return checks


def _fold_groups_are_disjoint(bindings):
    for (fold_id, task_slug), frame in bindings.groupby(
        ["fold_id", "task_slug"]
    ):
        group_column = (
            "view_id" if str(fold_id).startswith("leave_one_view_out") else "sortie_id"
        )
        train = set(frame[frame["split_role"] == "train"][group_column])
        test = set(frame[frame["split_role"] == "test"][group_column])
        if not train or not test or train & test:
            return False
    return True


def _check(name, passed, details):
    return {"check_name": name, "passed": bool(passed), "details": details}
