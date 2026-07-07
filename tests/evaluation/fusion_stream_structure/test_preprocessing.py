from __future__ import annotations

import numpy as np
import pandas as pd

from chronaris.evaluation.fusion_stream_structure.contracts import FusionStreamRunConfig, validate_input_frame
from chronaris.evaluation.fusion_stream_structure.dataset_loader import records_from_validated_frame
from chronaris.evaluation.fusion_stream_structure.preprocessing import preprocess_stream, preprocess_streams


def _record(frame: pd.DataFrame):
    validation = validate_input_frame(frame)
    return next(iter(records_from_validated_frame(validation.frame, validation.feature_columns).values()))


def test_missing_values_and_low_variance_filtering() -> None:
    frame = pd.DataFrame(
        {
            "method_name": ["chronaris"] * 35,
            "sortie_id": ["s1"] * 35,
            "view_id": ["v1"] * 35,
            "window_id": [f"w{i}" for i in range(35)],
            "time": list(range(35)),
            "fusion_feature_1": [np.nan if i == 3 else float(i) for i in range(35)],
            "fusion_feature_2": [1.0] * 35,
            "fusion_feature_3": [np.nan] * 35,
        }
    )
    stream = preprocess_stream(_record(frame), FusionStreamRunConfig(pca_explained_variance=1.0))
    assert not np.isnan(stream.matrix).any()
    assert "fusion_feature_3" in stream.manifest["missing_values"]["dropped_all_missing_columns"]
    assert "fusion_feature_2" in stream.manifest["low_variance_filter"]["dropped_columns"]


def test_groupwise_zscore_does_not_cross_method_groups() -> None:
    rows = []
    for method, offset in (("chronaris", 0.0), ("naive_time_sync", 100.0)):
        for index in range(35):
            rows.append(
                {
                    "method_name": method,
                    "sortie_id": "s1",
                    "view_id": "v1",
                    "window_id": f"{method}_{index}",
                    "time": index,
                    "fusion_feature_1": offset + index,
                }
            )
    validation = validate_input_frame(pd.DataFrame(rows))
    records = records_from_validated_frame(validation.frame, validation.feature_columns)
    streams = preprocess_streams(records, FusionStreamRunConfig(pca_explained_variance=1.0))
    for stream in streams.values():
        assert abs(float(stream.matrix[:, 0].mean())) < 1e-9
        assert abs(float(stream.matrix[:, 0].std()) - 1.0) < 1e-9


def test_pca_manifest_records_retained_components() -> None:
    x = np.linspace(0, 1, 40)
    frame = pd.DataFrame(
        {
            "method_name": ["chronaris"] * 40,
            "sortie_id": ["s1"] * 40,
            "view_id": ["v1"] * 40,
            "window_id": [f"w{i}" for i in range(40)],
            "time": list(range(40)),
            "fusion_feature_1": x,
            "fusion_feature_2": x * 2.0,
            "fusion_feature_3": x * -1.0,
        }
    )
    stream = preprocess_stream(_record(frame), FusionStreamRunConfig(pca_explained_variance=0.90))
    assert stream.manifest["pca"]["status"] == "applied"
    assert stream.manifest["pca"]["retained_components"] == 1


def test_min_t_marks_too_short() -> None:
    frame = pd.DataFrame(
        {
            "method_name": ["chronaris"] * 8,
            "sortie_id": ["s1"] * 8,
            "view_id": ["v1"] * 8,
            "window_id": [f"w{i}" for i in range(8)],
            "time": list(range(8)),
            "fusion_feature_1": list(range(8)),
        }
    )
    stream = preprocess_stream(_record(frame), FusionStreamRunConfig(min_T=30))
    assert stream.status == "too_short"
    assert stream.manifest["status"] == "too_short"
