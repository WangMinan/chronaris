"""ClaSP/CLaP evaluator wrapper for E3."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from chronaris.evaluation.fusion_stream_structure.preprocessing import PreprocessedFusionStream


@dataclass(frozen=True, slots=True)
class ClaspImportState:
    available: bool
    error: str | None = None
    version: str = "unknown"
    binary_segmentation: object | None = None
    clap_detection: object | None = None


def run_clasp_segmentation(stream: PreprocessedFusionStream) -> dict[str, object]:
    base = _base_result(stream)
    if stream.status == "too_short":
        return {**base, "status": "too_short", "reason": "T_below_min_T"}
    if stream.d == 0:
        return {**base, "status": "no_feature_columns"}
    import_state = _load_claspy()
    if not import_state.available:
        return {
            **base,
            "status": "claspy_unavailable",
            "import_available": False,
            "import_error": import_state.error,
            "algorithm_version": import_state.version,
        }
    try:
        data = np.asarray(stream.matrix, dtype=np.float64).T
        change_points = _run_binary_clasp(import_state.binary_segmentation, data)
        state_sequence, transition_graph = _run_clap(import_state.clap_detection, data, stream.T)
        return {
            **base,
            "status": "completed",
            "import_available": True,
            "algorithm_version": import_state.version,
            "change_points": change_points,
            "change_point_windows": [_map_point(point, stream) for point in change_points],
            "state_sequence": state_sequence,
            "state_transition_graph": transition_graph,
        }
    except Exception as exc:  # pragma: no cover - depends on optional third-party APIs.
        return {
            **base,
            "status": "clasp_error",
            "import_available": True,
            "algorithm_version": import_state.version,
            "error": repr(exc),
        }


def run_clasp_for_streams(
    streams: Mapping[tuple[str, str, str], PreprocessedFusionStream],
) -> dict[tuple[str, str, str], dict[str, object]]:
    return {key: run_clasp_segmentation(stream) for key, stream in streams.items()}


def _load_claspy() -> ClaspImportState:
    try:
        import claspy  # type: ignore
        from claspy.segmentation import BinaryClaSPSegmentation  # type: ignore
        from claspy.state_detection import AgglomerativeCLaPDetection  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when dependency absent.
        return ClaspImportState(available=False, error=repr(exc))
    return ClaspImportState(
        available=True,
        version=str(getattr(claspy, "__version__", "unknown")),
        binary_segmentation=BinaryClaSPSegmentation,
        clap_detection=AgglomerativeCLaPDetection,
    )


def _run_binary_clasp(binary_segmentation: object, data: np.ndarray) -> list[int]:
    model = binary_segmentation()
    prediction = None
    if hasattr(model, "fit_predict"):
        prediction = model.fit_predict(data)
    elif hasattr(model, "fit"):
        fitted = model.fit(data)
        if hasattr(fitted, "predict"):
            prediction = fitted.predict()
    if prediction is None and hasattr(model, "change_points"):
        prediction = getattr(model, "change_points")
    if prediction is None:
        return []
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    values = np.asarray(prediction).reshape(-1)
    return sorted({int(value) for value in values if np.isfinite(value) and int(value) > 0})


def _run_clap(clap_detection: object, data: np.ndarray, T: int) -> tuple[list[int], dict[str, object]]:
    model = clap_detection()
    states = None
    transitions = None
    if hasattr(model, "fit_predict"):
        prediction = model.fit_predict(data)
        if isinstance(prediction, tuple):
            states, transitions = prediction[0], prediction[1] if len(prediction) > 1 else None
        else:
            states = prediction
    elif hasattr(model, "fit"):
        fitted = model.fit(data)
        if hasattr(fitted, "predict"):
            try:
                prediction = fitted.predict(sparse=True)
            except TypeError:
                prediction = fitted.predict()
            if isinstance(prediction, tuple):
                states, transitions = prediction[0], prediction[1] if len(prediction) > 1 else None
            else:
                states = prediction
    state_sequence = _coerce_state_sequence(states, T)
    graph = _transition_graph(state_sequence)
    if transitions is not None:
        graph["raw_transitions"] = _json_safe(transitions)
    return state_sequence, graph


def _coerce_state_sequence(states: object, T: int) -> list[int]:
    if states is None:
        return [0] * T
    values = np.asarray(states).reshape(-1)
    if values.size == 0:
        return [0] * T
    if values.size < T:
        values = np.pad(values, (0, T - values.size), mode="edge")
    return [int(value) for value in values[:T]]


def _transition_graph(state_sequence: list[int]) -> dict[str, object]:
    counts = Counter(zip(state_sequence[:-1], state_sequence[1:]))
    return {
        "state_count": len(set(state_sequence)),
        "transitions": [
            {"from": int(left), "to": int(right), "count": int(count)}
            for (left, right), count in sorted(counts.items())
        ],
    }


def _map_point(point: int, stream: PreprocessedFusionStream) -> dict[str, object]:
    index = min(max(int(point), 0), stream.T - 1)
    return {
        "row_index": index,
        "window_id": stream.window_ids[index],
        "time": float(stream.times[index]),
    }


def _base_result(stream: PreprocessedFusionStream) -> dict[str, object]:
    return {
        "method_name": stream.record.method_name,
        "sortie_id": stream.record.sortie_id,
        "view_id": stream.record.view_id,
        "T": stream.T,
        "d": stream.d,
        "change_points": [],
        "change_point_windows": [],
        "state_sequence": [],
        "state_transition_graph": {"state_count": 0, "transitions": []},
    }


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value
