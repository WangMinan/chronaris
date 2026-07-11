from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation import write_raw_point_snapshot
from chronaris.evaluation.application_tasks.dingxin_context_data import (
    audit_lazy_dingxin_contexts,
    build_dingxin_lazy_context_index,
    build_fold_task_context_bindings,
)
from chronaris.representation import load_dingxin_observed_context
from chronaris.schema.models import RawPoint, StreamKind
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    sha256_file,
    write_deterministic_npz,
)


def _fixture(tmp_path: Path):
    root = tmp_path / "snapshot"
    sortie_id = "sortie_a"
    view_id = "view_a"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    physiology_path = root / "physiology.jsonl.gz"
    vehicle_path = root / "vehicle.jsonl.gz"
    physiology_record = write_raw_point_snapshot(
        physiology_path,
        [
            RawPoint(
                StreamKind.PHYSIOLOGY,
                "eeg",
                start + timedelta(seconds=second),
                {"f3": second},
            )
            for second in range(31)
        ],
        snapshot_root=root,
        sortie_id=sortie_id,
        view_id=view_id,
        pilot_id=1,
        expected_stream_kind=StreamKind.PHYSIOLOGY,
    )
    vehicle_record = write_raw_point_snapshot(
        vehicle_path,
        [
            RawPoint(
                StreamKind.VEHICLE,
                "bus",
                start + timedelta(seconds=second),
                {"safe": second, "label": second * 10},
            )
            for second in range(31)
        ],
        snapshot_root=root,
        sortie_id=sortie_id,
        view_id=None,
        pilot_id=None,
        expected_stream_kind=StreamKind.VEHICLE,
    )
    (root / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "plans": [
                    {
                        "sortie_id": sortie_id,
                        "view_id": view_id,
                        "pilot_id": 1,
                        "start_utc": start.isoformat(),
                        "stop_utc": (start + timedelta(seconds=31)).isoformat(),
                        "physiology_measurements": ["eeg"],
                        "vehicle_measurements": ["bus"],
                    }
                ],
                "files": [physiology_record.to_dict(), vehicle_record.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    roles_path = tmp_path / "roles.csv"
    pd.DataFrame(
        [
            {
                "sortie_id": sortie_id,
                "stream_kind": "physiology",
                "measurement": "eeg",
                "source_field": "f3",
                "semantic_category": "eeg",
                "allowed_in_maneuver_input": True,
            },
            {
                "sortie_id": sortie_id,
                "stream_kind": "vehicle",
                "measurement": "bus",
                "source_field": "safe",
                "semantic_category": "other_own_aircraft",
                "allowed_in_maneuver_input": True,
            },
            {
                "sortie_id": sortie_id,
                "stream_kind": "vehicle",
                "measurement": "bus",
                "source_field": "label",
                "semantic_category": "pitch",
                "allowed_in_maneuver_input": False,
            },
        ]
    ).to_csv(roles_path, index=False)
    context_path = tmp_path / "contexts.jsonl"
    contexts = [
        {
            "context_id": "full",
            "sortie_id": sortie_id,
            "view_id": view_id,
            "pilot_id": 1,
            "start_window_index": 0,
            "end_window_index": 5,
            "start_offset_ms": 0,
            "end_offset_ms": 30_000,
        },
        {
            "context_id": "partial",
            "sortie_id": sortie_id,
            "view_id": view_id,
            "pilot_id": 1,
            "start_window_index": 1,
            "end_window_index": 6,
            "start_offset_ms": 5_000,
            "end_offset_ms": 30_991,
        },
    ]
    context_path.write_text(
        "".join(json.dumps(row) + "\n" for row in contexts),
        encoding="utf-8",
    )
    return root, roles_path, context_path


def test_lazy_context_index_rejects_partial_window_and_excludes_label_source(
    tmp_path: Path,
) -> None:
    root, roles_path, context_path = _fixture(tmp_path)
    index = build_dingxin_lazy_context_index(
        snapshot_root=root,
        field_role_manifest_path=roles_path,
        context_manifest_path=context_path,
    )
    audit = audit_lazy_dingxin_contexts(index).set_index("context_id")
    cached = index.load_sample("full")
    uncached = load_dingxin_observed_context(
        index.plan,
        index.context_by_id["full"],
    )
    batch = index.load_batch(("full",))

    assert audit.loc["full", "status"] == "completed"
    assert audit.loc["partial", "reason"] == "input_context_duration_not_30_seconds"
    assert batch.sample_ids == ("full",)
    assert index.plan.schema.vehicle_feature_names == ("vehicle.channel_00.safe",)
    assert any("label" in value for value in index.plan.schema.excluded_feature_names)
    assert np.array_equal(cached.physiology_values, uncached.physiology_values)
    assert np.array_equal(cached.vehicle_values, uncached.vehicle_values)
    assert np.array_equal(cached.physiology_feature_mask, uncached.physiology_feature_mask)
    assert np.array_equal(cached.vehicle_feature_mask, uncached.vehicle_feature_mask)


def test_target_binding_uses_archive_hash_and_input_availability(tmp_path: Path) -> None:
    root, roles_path, context_path = _fixture(tmp_path)
    index = build_dingxin_lazy_context_index(
        snapshot_root=root,
        field_role_manifest_path=roles_path,
        context_manifest_path=context_path,
    )
    archive_path = tmp_path / "targets.npz"
    archive_hash = write_deterministic_npz(
        archive_path,
        {
            "context_ids": np.asarray(["full", "partial"]),
            "split_roles": np.asarray(["train", "test"]),
            "statuses": np.asarray(["completed", "completed"]),
            "fit_sample_hashes": np.asarray(["a" * 64, "a" * 64]),
            "input_start_offset_ms": np.asarray([0, 5_000], dtype=np.int64),
            "input_end_exclusive_ms": np.asarray([30_000, 30_991], dtype=np.int64),
            "target_start_offset_ms": np.asarray([25_000, 25_991], dtype=np.int64),
            "target_end_exclusive_ms": np.asarray([30_000, 30_991], dtype=np.int64),
        },
    )
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "fold_id": "leave_one_view_out__fold01",
                "task_slug": "maneuver_intensity_classification",
                "archive_path": str(archive_path),
                "archive_sha256": archive_hash,
                "threshold_path": str(threshold_path),
                "threshold_sha256": sha256_file(threshold_path),
            }
        ]
    ).to_csv(manifest_path, index=False)
    bindings, verified = build_fold_task_context_bindings(
        index=index,
        target_archive_manifest_path=manifest_path,
    )
    status = bindings.set_index("context_id")["binding_status"]

    assert status["full"] == "available"
    assert status["partial"] == "input_context_duration_not_30_seconds"
    assert len(verified) == 1
