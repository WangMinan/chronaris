from __future__ import annotations

import pandas as pd
import pytest

from chronaris.evaluation.fusion_stream_structure.contracts import (
    ContractError,
    normalize_method_name,
    validate_input_frame,
)
from chronaris.evaluation.fusion_stream_structure.dataset_loader import (
    method_unavailable_entry,
    records_from_validated_frame,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_name": "chronaris_full",
                "sortie_id": "s1",
                "view_id": "v1",
                "window_id": "w2",
                "time": 2,
                "fusion_feature_1": 2.0,
            },
            {
                "method_name": "chronaris",
                "sortie_id": "s1",
                "view_id": "v1",
                "window_id": "w1",
                "time": 1,
                "fusion_feature_1": 1.0,
            },
        ]
    )


def test_required_columns_missing_raises() -> None:
    frame = _frame().drop(columns=["sortie_id"])
    with pytest.raises(ContractError) as excinfo:
        validate_input_frame(frame)
    assert "sortie_id" in excinfo.value.details["missing_columns"]


def test_method_name_normalization_aliases() -> None:
    assert normalize_method_name("chronaris_full") == "chronaris"
    assert normalize_method_name("naive_sync") == "naive_time_sync"


def test_illegal_method_name_raises() -> None:
    frame = _frame()
    frame.loc[0, "method_name"] = "not_a_method"
    with pytest.raises(ContractError):
        validate_input_frame(frame)


def test_forbidden_sidecar_selected_as_feature_raises() -> None:
    frame = _frame()
    frame["alignment_score"] = [0.1, 0.2]
    with pytest.raises(ContractError) as excinfo:
        validate_input_frame(frame, explicit_feature_columns=("alignment_score",))
    assert "alignment_score" in excinfo.value.details["forbidden_columns"]


def test_all_empty_feature_columns_raise() -> None:
    frame = _frame()
    frame["fusion_feature_1"] = None
    with pytest.raises(ContractError):
        validate_input_frame(frame)


def test_time_sorting_and_window_mapping() -> None:
    validation = validate_input_frame(_frame())
    records = records_from_validated_frame(validation.frame, validation.feature_columns)
    record = records[("chronaris", "s1", "v1")]
    assert record.window_ids == ("w1", "w2")
    assert record.times == (1.0, 2.0)


def test_method_unavailable_structure() -> None:
    entry = method_unavailable_entry("mult", "no_reusable_fusion_feature_frame")
    assert entry["method_name"] == "mult"
    assert entry["status"] == "method_unavailable"
    assert entry["reason"] == "no_reusable_fusion_feature_frame"
