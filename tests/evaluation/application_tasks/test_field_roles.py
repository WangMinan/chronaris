"""Vehicle metadata semantics and leakage-exclusion tests."""

from __future__ import annotations

from chronaris.dataset.application_evaluation import (
    build_field_role_manifest,
    selected_maneuver_roles,
)

from tests.evaluation.application_tasks.helpers import make_records, vehicle_labels_by_sortie


def test_maneuver_fields_require_resolved_own_aircraft_semantics() -> None:
    records = make_records()
    roles = build_field_role_manifest(
        records,
        vehicle_labels_by_sortie=vehicle_labels_by_sortie(records),
    )
    selected = selected_maneuver_roles(roles)

    assert len(selected) == 8
    assert {role.semantic_key for role in selected} == {
        "acceleration_north",
        "acceleration_west",
        "acceleration_up",
        "roll",
    }
    assert all(role.metadata_status == "resolved" for role in selected)
    assert all(not role.allowed_in_maneuver_input for role in selected)
    assert not any("target" in role.feature_name for role in selected)
    assert not any("quality" in role.feature_name for role in selected)


def test_unresolved_vehicle_fields_never_become_label_sources() -> None:
    records = make_records()
    roles = build_field_role_manifest(records, vehicle_labels_by_sortie={})

    assert selected_maneuver_roles(roles) == ()
    vehicle_roles = [role for role in roles if role.stream_kind == "vehicle"]
    assert vehicle_roles
    assert all(role.allowed_in_maneuver_input for role in vehicle_roles)
