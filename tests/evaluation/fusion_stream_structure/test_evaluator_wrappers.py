from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from chronaris.evaluation.fusion_stream_structure.clasp_segmentation import (
    ClaspImportState,
    run_clasp_segmentation,
)
from chronaris.evaluation.fusion_stream_structure.contracts import FusionStreamRunConfig, validate_input_frame
from chronaris.evaluation.fusion_stream_structure.dataset_loader import records_from_validated_frame
from chronaris.evaluation.fusion_stream_structure.preprocessing import preprocess_stream
from chronaris.evaluation.fusion_stream_structure.stumpy_motif_discord import (
    StumpyImportState,
    run_stumpy_motif_discord,
)
import chronaris.evaluation.fusion_stream_structure.clasp_segmentation as clasp_module
import chronaris.evaluation.fusion_stream_structure.stumpy_motif_discord as stumpy_module


def _stream():
    T = 72
    x = np.linspace(0, 6 * np.pi, T)
    rows = []
    for index in range(T):
        rows.append({
            "method_name": "chronaris",
            "sortie_id": "synthetic_sortie",
            "view_id": "view_alpha",
            "window_id": f"w{index:03d}",
            "time": index,
            "maneuver_proxy_label": "high" if index >= 48 else "low",
            "weak_event_boundary": index in {24, 48},
            "physio_fluctuation_interval": 55 <= index < 61,
            "fusion_feature_1": float(np.sin(x[index])),
            "fusion_feature_2": float(np.cos(x[index])),
            "fusion_feature_3": float(np.sin(x[index] * 0.5)),
        })
    validation = validate_input_frame(pd.DataFrame(rows))
    record = next(iter(records_from_validated_frame(validation.frame, validation.feature_columns).values()))
    return preprocess_stream(record, FusionStreamRunConfig(min_T=30, pca_explained_variance=1.0))


def test_clasp_fallback_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clasp_module,
        "_load_claspy",
        lambda: ClaspImportState(available=False, error="missing test dependency"),
    )
    result = run_clasp_segmentation(_stream())
    assert result["status"] == "claspy_unavailable"
    assert result["import_available"] is False
    assert result["import_error"] == "missing test dependency"


def test_stumpy_fallback_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stumpy_module,
        "_load_stumpy",
        lambda: StumpyImportState(available=False, error="missing test dependency"),
    )
    result = run_stumpy_motif_discord(_stream(), m_grid=(8,))
    assert result["status"] == "stumpy_unavailable"
    assert result["import_available"] is False
    assert result["import_error"] == "missing test dependency"


def test_optional_evaluators_complete_when_dependencies_are_installed() -> None:
    if importlib.util.find_spec("claspy") is None or importlib.util.find_spec("stumpy") is None:
        pytest.skip("optional E3 evaluator dependencies are not installed")
    stream = _stream()
    clasp_result = run_clasp_segmentation(stream)
    stumpy_result = run_stumpy_motif_discord(stream, m_grid=(8,))
    assert clasp_result["status"] == "completed"
    assert clasp_result["import_available"] is True
    assert clasp_result["clap_status"] in {"completed", "clap_unavailable"}
    assert stumpy_result["status"] == "completed"
    assert stumpy_result["import_available"] is True
    assert stumpy_result["mode"] in {"mstump", "stump_pc1"}
