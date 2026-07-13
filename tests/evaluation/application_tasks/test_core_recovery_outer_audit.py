import json

import pandas as pd

from chronaris.evaluation.application_tasks.core_recovery_outer_audit import (
    inspect_outer_support_isolation,
)


def test_outer_support_audit_detects_cross_view_time_overlap(tmp_path):
    context_path = tmp_path / "contexts.jsonl"
    split_path = tmp_path / "splits.json"
    pd.DataFrame(
        [
            {
                "context_id": "train",
                "sortie_id": "sortie-a",
                "view_id": "pilot-a",
                "start_offset_ms": 0,
                "end_offset_ms": 30_000,
            },
            {
                "context_id": "held",
                "sortie_id": "sortie-a",
                "view_id": "pilot-b",
                "start_offset_ms": 0,
                "end_offset_ms": 30_000,
            },
        ]
    ).to_json(context_path, orient="records", lines=True)
    split_path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold_id": "fold01",
                        "train_sample_ids": ["train"],
                        "held_out_sample_ids": ["held"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = inspect_outer_support_isolation(
        context_manifest_path=context_path,
        split_manifest_path=split_path,
        fold_ids=("fold01",),
    )

    assert not result["valid"]
    assert result["folds"][0]["overlapping_train_context_count"] == 1
    assert result["folds"][0]["exact_sortie_anchor_pair_count"] == 1
