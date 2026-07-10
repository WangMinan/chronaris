"""G2a snapshot orchestration and resume tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from chronaris.dataset.application_evaluation.snapshot_contracts import SnapshotViewPlan
from chronaris.evaluation.application_tasks.fixed_data_snapshot import (
    FixedDataSnapshotConfig,
    run_fixed_data_snapshot,
)
from chronaris.schema.models import RawPoint, StreamKind


@dataclass(frozen=True)
class _View:
    view_id: str
    pilot_id: int


@dataclass(frozen=True)
class _Profile:
    sortie_id: str
    views: tuple[_View, ...]
    model_physiology_measurements: tuple[str, ...] = ("eeg", "spo2")
    vehicle_measurements: tuple[str, ...] = ("BUS.demo",)


class _ProfileResolver:
    def __init__(self, profile: _Profile):
        self.profile = profile

    def resolve_many(self, _sortie_ids):
        return (self.profile,)


class _PointSource:
    def __init__(self):
        self.call_count = 0

    def fetch_vehicle(self, profile, *, start_utc, stop_utc):
        del profile, stop_utc
        self.call_count += 1
        return tuple(
            RawPoint(
                stream_kind=StreamKind.VEHICLE,
                measurement="BUS.demo",
                timestamp=start_utc + timedelta(seconds=index),
                values={"code1": str(index)},
                source="influx",
            )
            for index in range(3)
        )

    def fetch_physiology(self, profile, view, *, start_utc, stop_utc):
        del profile, view, stop_utc
        self.call_count += 1
        return tuple(
            RawPoint(
                stream_kind=StreamKind.PHYSIOLOGY,
                measurement="eeg",
                timestamp=start_utc + timedelta(seconds=index),
                values={"af3": str(index)},
                source="influx",
            )
            for index in range(2)
        )


def test_snapshot_writes_shared_vehicle_file_and_resumes(monkeypatch, tmp_path) -> None:
    start = datetime(2025, 10, 2, 8, 35, tzinfo=timezone.utc)
    plan = SnapshotViewPlan(
        sortie_id="sortie-a",
        view_id="view-a",
        pilot_id=10033,
        start_utc=start.isoformat(),
        stop_utc=(start + timedelta(seconds=5)).isoformat(),
        physiology_measurements=("eeg", "spo2"),
        vehicle_measurements=("BUS.demo",),
        expected_physiology_point_count=2,
        expected_vehicle_point_count=3,
    )
    monkeypatch.setattr(
        "chronaris.evaluation.application_tasks.fixed_data_snapshot.build_fixed_snapshot_plan",
        lambda **_: (plan,),
    )
    monkeypatch.setattr(
        "chronaris.evaluation.application_tasks.fixed_data_snapshot.source_manifest_hashes",
        lambda *_: {"e_run_manifest_sha256": "e", "f_run_manifest_sha256": "f"},
    )
    label_manifest = tmp_path / "label-fields.json"
    label_manifest.write_text(
        json.dumps(
            {
                "maneuver_label_fields": [
                    {
                        "sortie_id": "sortie-a",
                        "feature_name": "BUS.demo.code1",
                        "display_label": "载机演示字段",
                        "semantic_key": "demo",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = FixedDataSnapshotConfig(
        run_id="snapshot-test",
        snapshot_output_root=str(tmp_path / "raw"),
        compact_output_root=str(tmp_path / "compact"),
        e_run_manifest_path="unused-e.json",
        f_run_manifest_path="unused-f.json",
        label_field_manifest_path=str(label_manifest),
        allowed_sortie_ids=("sortie-a",),
        resume=True,
    )
    source = _PointSource()
    resolver = _ProfileResolver(_Profile("sortie-a", (_View("view-a", 10033),)))

    first = run_fixed_data_snapshot(config, profile_resolver=resolver, point_source=source)
    assert first.status == "completed"
    assert first.file_count == 2
    assert first.vehicle_point_count == 3
    assert first.physiology_point_count == 2
    assert source.call_count == 2
    assert (
        tmp_path / "raw/snapshot-test/sorties/sortie-a/vehicle_points.jsonl.gz"
    ).exists()

    resumed = run_fixed_data_snapshot(config, profile_resolver=resolver, point_source=source)
    assert resumed.resumed
    assert source.call_count == 2
