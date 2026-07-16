from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chronaris.evaluation.dingxin.representation_compatibility import (
    audit_representation_compatibility,
)


def test_compatibility_requires_historical_maneuver_fields(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.jsonl"
    protocol_path = tmp_path / "protocol.json"
    roles_path = tmp_path / "roles.csv"
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    rows = []
    contexts_by_role = {
        "train": ["context_1"],
        "validation": ["context_2"],
        "held_out": ["context_3"],
    }
    for fold_id in (
        "leave_one_sortie_out__fold01",
        "leave_one_sortie_out__fold02",
    ):
        for role, sample_ids in contexts_by_role.items():
            manifest_path = manifest_root / f"{fold_id}_{role}.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "label_used_for_encoder_training": False,
                        "output_dim": 64,
                        "query_point_count": 96,
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                {
                    "seed": 17,
                    "method_name": "chronaris",
                    "fold_id": fold_id,
                    "export_role": role,
                    "sample_ids": sample_ids,
                    "manifest_path": str(manifest_path.relative_to(tmp_path)),
                }
            )
    pd.DataFrame(rows).to_json(inventory_path, orient="records", lines=True)
    protocol_path.write_text(
        json.dumps(
            {
                "task_targets_opened": False,
                "outer_test_metrics_opened": False,
                "representation_family": "frozen_task_agnostic_v1",
            }
        ),
        encoding="utf-8",
    )
    roles = pd.DataFrame(
        [
            {
                "selected_for_maneuver_label": True,
                "allowed_in_maneuver_input": True,
            },
            {
                "selected_for_maneuver_label": False,
                "allowed_in_maneuver_input": True,
            },
        ]
    )
    roles.to_csv(roles_path, index=False)

    reusable = audit_representation_compatibility(
        expected_context_ids=contexts_by_role["train"]
        + contexts_by_role["validation"]
        + contexts_by_role["held_out"],
        inventory_path=inventory_path,
        protocol_path=protocol_path,
        field_role_manifest_path=roles_path,
        required_methods=("chronaris",),
        required_seeds=(17,),
        repository_root=tmp_path,
    )
    assert reusable["status"] == "reusable_as_primary"

    roles.loc[0, "allowed_in_maneuver_input"] = False
    roles.to_csv(roles_path, index=False)
    incompatible = audit_representation_compatibility(
        expected_context_ids=("context_1", "context_2", "context_3"),
        inventory_path=inventory_path,
        protocol_path=protocol_path,
        field_role_manifest_path=roles_path,
        required_methods=("chronaris",),
        required_seeds=(17,),
        repository_root=tmp_path,
    )
    assert incompatible["status"] == "retraining_required"
    assert incompatible["excluded_historical_maneuver_source_count"] == 1
