from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation import (
    stable_sample_hash,
    write_raw_point_snapshot,
)
from chronaris.dataset.application_evaluation.labels import MANEUVER_TASK_ID
from chronaris.evaluation.application_tasks.dingxin_target_archive import (
    _write_target_archives,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    DingxinTargetSourceData,
    build_raw_median_response_targets,
)
from chronaris.schema.models import RawPoint, StreamKind


def _source(tmp_path: Path, *, perturb_test: bool = False):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    raw_path = tmp_path / "views" / "view_a" / "physiology_points.jsonl.gz"
    points = []
    for second in range(61):
        multiplier = 100.0 if perturb_test and second >= 50 else 1.0
        points.append(
            RawPoint(
                stream_kind=StreamKind.PHYSIOLOGY,
                measurement="eeg",
                timestamp=start + timedelta(seconds=second),
                values={
                    "x": multiplier * second**2,
                    "y": multiplier * second**3,
                },
            )
        )
    file_record = write_raw_point_snapshot(
        raw_path,
        points,
        snapshot_root=tmp_path,
        sortie_id="sortie_a",
        view_id="view_a",
        pilot_id=1,
        expected_stream_kind=StreamKind.PHYSIOLOGY,
    ).to_dict()
    ends = (30_000, 35_000, 40_000, 45_000, 50_000, 55_000, 60_000)
    context_ids = tuple(f"context_{value}" for value in ends)
    contexts = pd.DataFrame(
        [
            {
                "context_id": context_id,
                "sortie_id": "sortie_a",
                "view_id": "view_a",
                "pilot_id": 1,
                "start_offset_ms": end - 30_000,
                "end_offset_ms": end,
                "response_eligible": True,
            }
            for context_id, end in zip(context_ids, ends, strict=True)
        ]
    )
    roles = pd.DataFrame(
        [
            {
                "feature_name": f"eeg.{field}",
                "measurement": "eeg",
                "source_field": field,
                "selected_for_response_target": True,
            }
            for field in ("x", "y")
        ]
    )
    source = DingxinTargetSourceData(
        contexts=contexts,
        audit_labels=pd.DataFrame(),
        audit_thresholds=pd.DataFrame(),
        field_roles=roles,
        split_manifest={
            "split_protocols": [
                {
                    "fold_id": "fold_a",
                    "split_strategy": "leave_one_view_out",
                    "held_out_group": "view_a",
                    "response_train_context_ids": list(context_ids[:4]),
                    "response_test_context_ids": list(context_ids[4:]),
                }
            ]
        },
        snapshot_manifest={
            "plans": [
                {
                    "view_id": "view_a",
                    "start_utc": start.isoformat(),
                    "stop_utc": (start + timedelta(seconds=61)).isoformat(),
                }
            ],
            "files": [file_record],
        },
        source_hashes={},
        verified_snapshot_files=(),
    )
    return source, context_ids


def test_raw_median_targets_are_train_only_and_mark_incomplete_future(tmp_path) -> None:
    source, context_ids = _source(tmp_path / "baseline")
    result = build_raw_median_response_targets(source, snapshot_root=tmp_path / "baseline")
    perturbed_source, _ = _source(tmp_path / "perturbed", perturb_test=True)
    perturbed = build_raw_median_response_targets(
        perturbed_source,
        snapshot_root=tmp_path / "perturbed",
    )

    labels = result.label_rows.set_index("context_id")
    assert labels.loc[context_ids[-1], "status"] == "future_interval_not_fully_observed"
    assert labels.loc[list(context_ids[:-1]), "status"].eq("completed").all()
    assert labels["representative_statistic"].eq("raw_window_median").all()
    assert set(
        result.threshold_rows[
            result.threshold_rows["parameter_type"] == "response_field_scale"
        ]["parameter_name"]
    ) == {"eeg.x", "eeg.y"}
    baseline_thresholds = result.threshold_rows.sort_values(
        ["parameter_type", "parameter_name"]
    ).reset_index(drop=True)
    perturbed_thresholds = perturbed.threshold_rows.sort_values(
        ["parameter_type", "parameter_name"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_thresholds, perturbed_thresholds)
    assert set(result.label_rows["fit_sample_hash"]) == {
        stable_sample_hash(context_ids[:4])
    }


def test_target_archives_are_pickle_free_and_resumable(tmp_path) -> None:
    source, context_ids = _source(tmp_path / "raw")
    response = build_raw_median_response_targets(source, snapshot_root=tmp_path / "raw")
    classification = pd.DataFrame(
        [
            {
                "fold_id": "fold_a",
                "split_role": "train" if index < 4 else "test",
                "context_id": context_id,
                "class_label": ("low", "medium", "high")[index % 3],
                "status": "completed",
                "valid_field_count": 4,
                "fit_sample_hash": stable_sample_hash(context_ids[:4]),
                "input_start_offset_ms": index * 5_000,
                "input_end_exclusive_ms": index * 5_000 + 30_000,
                "target_start_offset_ms": index * 5_000 + 25_000,
                "target_end_exclusive_ms": index * 5_000 + 30_000,
                "representative_statistic": "aligned_window_std_and_delta",
            }
            for index, context_id in enumerate(context_ids)
        ]
    )
    classification_threshold = pd.DataFrame(
        [
            {
                "fold_id": "fold_a",
                "task_id": MANEUVER_TASK_ID,
                "parameter_type": "class_bounds",
                "parameter_name": "low_medium_high",
            }
        ]
    )
    thresholds = pd.concat(
        (classification_threshold, response.threshold_rows),
        ignore_index=True,
        sort=False,
    )
    first = _write_target_archives(
        classification_rows=classification,
        response_rows=response.label_rows,
        threshold_rows=thresholds,
        output_root=tmp_path / "archives",
        resume=True,
    )
    second = _write_target_archives(
        classification_rows=classification,
        response_rows=response.label_rows,
        threshold_rows=thresholds,
        output_root=tmp_path / "archives",
        resume=True,
    )

    assert len(first) == 2
    assert {row["status"] for row in second} == {"resumed"}
    assert all(row["hash_stable_on_rewrite"] for row in second)
    for row in second:
        with np.load(row["archive_path"], allow_pickle=False) as archive:
            assert archive["context_ids"].dtype.kind == "U"
            assert archive["statuses"].dtype.kind == "U"
