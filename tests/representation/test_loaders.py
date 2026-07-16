from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from chronaris.dataset.application_evaluation.snapshot_io import write_raw_point_snapshot
from chronaris.representation import (
    DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
    build_dingxin_observation_schema_plan,
    load_dingxin_observed_context,
    load_simulation_observed_context,
)
from chronaris.representation.contracts import RepresentationContractError
from chronaris.schema.models import RawPoint, StreamKind
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    write_deterministic_npz,
)


def _simulation_arrays() -> dict[str, np.ndarray]:
    return {
        "vehicle_observed_time_s": np.asarray([0.0, 5.0, 30.0]),
        "vehicle_values": np.asarray([[1.0], [2.0], [3.0]]),
        "vehicle_feature_names": np.asarray(["speed"]),
        "physiology_observed_time_s": np.asarray([0.5, 10.0, 29.5]),
        "physiology_values": np.asarray([[4.0], [5.0], [6.0]]),
        "physiology_feature_names": np.asarray(["eeg"]),
    }


def test_simulation_loader_accepts_only_observed_archive_and_slices_context(tmp_path):
    path = tmp_path / "raw_dual_stream.npz"
    write_deterministic_npz(path, _simulation_arrays())

    sample = load_simulation_observed_context(
        path,
        context_start_s=0.0,
        sample_id="sim_a",
        group_id="profile_a",
    )

    assert sample.vehicle_values[:, 0].tolist() == [1.0, 2.0]
    assert sample.physiology_values[:, 0].tolist() == [4.0, 5.0, 6.0]
    assert sample.schema.vehicle_feature_names == ("vehicle.speed",)
    assert sample.source_sample_hash


def test_simulation_loader_fails_closed_on_oracle_injection(tmp_path):
    path = tmp_path / "raw_dual_stream.npz"
    arrays = _simulation_arrays()
    arrays["workload"] = np.asarray([0.5])
    write_deterministic_npz(path, arrays)

    with pytest.raises(RepresentationContractError, match="extra=.*workload"):
        load_simulation_observed_context(path, context_start_s=0.0)


def test_dingxin_loader_builds_common_schema_and_excludes_label_source(tmp_path):
    root = tmp_path / "snapshot"
    sortie_id = "sortie_a"
    view_id = "sortie_a__pilot_1"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vehicle_path = root / "sorties" / sortie_id / "vehicle_points.jsonl.gz"
    physiology_path = (
        root / "sorties" / sortie_id / "views" / view_id / "physiology_points.jsonl.gz"
    )
    vehicle_record = write_raw_point_snapshot(
        vehicle_path,
        [
            RawPoint(
                StreamKind.VEHICLE,
                "bus_a",
                base + timedelta(seconds=offset),
                {"safe": str(10 + offset), "label_source": str(100 + offset)},
            )
            for offset in (0, 1, 29)
        ],
        snapshot_root=root,
        sortie_id=sortie_id,
        view_id=None,
        pilot_id=None,
        expected_stream_kind=StreamKind.VEHICLE,
    )
    physiology_record = write_raw_point_snapshot(
        physiology_path,
        [
            RawPoint(
                StreamKind.PHYSIOLOGY,
                "eeg",
                base + timedelta(seconds=offset),
                {"f3": str(1 + offset), "date_time": "not_numeric"},
            )
            for offset in (0, 2, 28)
        ],
        snapshot_root=root,
        sortie_id=sortie_id,
        view_id=view_id,
        pilot_id=1,
        expected_stream_kind=StreamKind.PHYSIOLOGY,
    )
    manifest = {
        "plans": [
            {
                "sortie_id": sortie_id,
                "view_id": view_id,
                "pilot_id": 1,
                "start_utc": base.isoformat(),
                "stop_utc": (base + timedelta(seconds=31)).isoformat(),
                "physiology_measurements": ["eeg"],
                "vehicle_measurements": ["bus_a"],
            }
        ],
        "files": [vehicle_record.to_dict(), physiology_record.to_dict()],
    }
    (root / "snapshot_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    roles_path = tmp_path / "field_roles.csv"
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
                "measurement": "bus_a",
                "source_field": "safe",
                "semantic_category": "other_own_aircraft",
                "allowed_in_maneuver_input": True,
            },
            {
                "sortie_id": sortie_id,
                "stream_kind": "vehicle",
                "measurement": "bus_a",
                "source_field": "label_source",
                "semantic_category": "pitch",
                "allowed_in_maneuver_input": False,
            },
        ]
    ).to_csv(roles_path, index=False)
    plan = build_dingxin_observation_schema_plan(
        snapshot_root=root,
        field_role_manifest_path=roles_path,
    )

    sample = load_dingxin_observed_context(
        plan,
        {
            "context_id": "context_a",
            "sortie_id": sortie_id,
            "view_id": view_id,
            "start_offset_ms": 0,
            "end_offset_ms": 30_000,
        },
    )

    assert plan.schema.physiology_feature_names == ("physiology.eeg.f3",)
    assert plan.schema.vehicle_feature_names == ("vehicle.channel_00.safe",)
    assert any("label_source" in value for value in plan.schema.excluded_feature_names)
    assert sample.vehicle_values[:, 0].tolist() == [10.0, 11.0, 39.0]
    assert sample.physiology_values[:, 0].tolist() == [1.0, 3.0, 29.0]

    future_roles = pd.read_csv(roles_path)
    future_roles["selected_for_maneuver_label"] = (
        future_roles["source_field"] == "label_source"
    )
    future_roles.to_csv(roles_path, index=False)
    future_plan = build_dingxin_observation_schema_plan(
        snapshot_root=root,
        field_role_manifest_path=roles_path,
        maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
    )

    assert future_plan.schema.schema_id == (
        "dingxin_future_prediction_common_observed.v1"
    )
    assert future_plan.schema.vehicle_feature_names == (
        "vehicle.channel_00.label_source",
        "vehicle.channel_00.safe",
    )
    assert not future_plan.schema.excluded_feature_names
