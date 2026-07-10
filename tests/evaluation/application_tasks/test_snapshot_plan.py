"""Reference-manifest snapshot plan tests."""

from __future__ import annotations

from chronaris.evaluation.application_tasks.fixed_data_snapshot_plan import (
    build_fixed_snapshot_plan,
)


def test_fixed_snapshot_plan_matches_committed_feature_export_contract() -> None:
    plans = build_fixed_snapshot_plan(
        e_run_manifest_path=(
            "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
        ),
        f_run_manifest_path=(
            "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
        ),
    )

    assert len(plans) == 3
    assert {plan.pilot_id for plan in plans} == {10033, 10035}
    assert all(plan.physiology_measurements == ("eeg", "spo2") for plan in plans)
    assert all(len(plan.vehicle_measurements) == 6 for plan in plans)
    assert all(plan.expected_physiology_point_count > 0 for plan in plans)
    assert all(plan.expected_vehicle_point_count > 0 for plan in plans)
    assert {plan.stop_utc for plan in plans} == {
        "2025-10-05T01:38:01+00:00",
        "2025-10-02T08:38:01+00:00",
    }
