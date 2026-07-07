"""STUMPY Matrix Profile evaluator wrapper for E3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from chronaris.evaluation.fusion_stream_structure.preprocessing import PreprocessedFusionStream


@dataclass(frozen=True, slots=True)
class StumpyImportState:
    available: bool
    error: str | None = None
    version: str = "unknown"
    module: object | None = None


def default_m_grid(T: int, *, slack: int = 2) -> tuple[int, ...]:
    candidates = (
        max(3, int(round(0.05 * T))),
        int(round(0.10 * T)),
        int(round(0.20 * T)),
    )
    values = sorted({value for value in candidates if value >= 3 and value < T - slack})
    return tuple(values)


def run_stumpy_motif_discord(
    stream: PreprocessedFusionStream,
    *,
    m_grid: str | Sequence[int] = "auto",
) -> dict[str, object]:
    base = _base_result(stream)
    if stream.d == 0:
        return {**base, "status": "no_feature_columns"}
    import_state = _load_stumpy()
    if not import_state.available:
        return {
            **base,
            "status": "stumpy_unavailable",
            "import_available": False,
            "import_error": import_state.error,
            "algorithm_version": import_state.version,
            "m_grid": list(_resolve_m_grid(stream.T, m_grid)),
        }
    valid_m = _resolve_m_grid(stream.T, m_grid)
    if not valid_m:
        return {
            **base,
            "status": "invalid_m_grid",
            "import_available": True,
            "algorithm_version": import_state.version,
            "m_grid": [],
        }
    try:
        best_result = None
        errors = []
        for m in valid_m:
            try:
                result = _run_one_m(import_state.module, stream, m)
                best_result = result
                break
            except Exception as exc:  # pragma: no cover - depends on optional third-party APIs.
                errors.append({"m": m, "error": repr(exc)})
        if best_result is None:
            return {
                **base,
                "status": "stumpy_error",
                "import_available": True,
                "algorithm_version": import_state.version,
                "m_grid": list(valid_m),
                "errors": errors,
            }
        return {
            **base,
            **best_result,
            "status": "completed",
            "import_available": True,
            "algorithm_version": import_state.version,
            "m_grid": list(valid_m),
            "errors": errors,
        }
    except Exception as exc:  # pragma: no cover
        return {
            **base,
            "status": "stumpy_error",
            "import_available": True,
            "algorithm_version": import_state.version,
            "error": repr(exc),
            "m_grid": list(valid_m),
        }


def run_stumpy_for_streams(
    streams: Mapping[tuple[str, str, str], PreprocessedFusionStream],
    *,
    m_grid: str | Sequence[int] = "auto",
) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        key: run_stumpy_motif_discord(stream, m_grid=m_grid)
        for key, stream in streams.items()
    }


def _load_stumpy() -> StumpyImportState:
    try:
        import stumpy  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when dependency absent.
        return StumpyImportState(available=False, error=repr(exc))
    return StumpyImportState(
        available=True,
        version=str(getattr(stumpy, "__version__", "unknown")),
        module=stumpy,
    )


def _resolve_m_grid(T: int, m_grid: str | Sequence[int]) -> tuple[int, ...]:
    if m_grid == "auto":
        return default_m_grid(T)
    values = sorted({int(value) for value in m_grid if int(value) >= 3 and int(value) < T - 2})
    return tuple(values)


def _run_one_m(stumpy_module: object, stream: PreprocessedFusionStream, m: int) -> dict[str, object]:
    matrix = np.asarray(stream.matrix, dtype=np.float64)
    matrix_t = matrix.T
    try:
        matrix_profile, matrix_profile_indices = stumpy_module.mstump(matrix_t, m=m)
        profile_vector = _profile_vector(matrix_profile)
        index_vector = _index_vector(matrix_profile_indices)
        mode = "mstump"
    except Exception:
        pc1 = _first_component(matrix)
        stump_output = stumpy_module.stump(pc1, m=m)
        profile_vector = np.asarray(stump_output[:, 0], dtype=np.float64)
        index_vector = np.asarray(stump_output[:, 1], dtype=np.float64)
        mode = "stump_pc1"
    finite_profile = np.where(np.isfinite(profile_vector), profile_vector, np.nan)
    if np.all(np.isnan(finite_profile)):
        raise ValueError("matrix profile contains no finite distances")
    motif_index = int(np.nanargmin(finite_profile))
    discord_index = int(np.nanargmax(finite_profile))
    nearest_index = int(index_vector[motif_index]) if np.isfinite(index_vector[motif_index]) else None
    result = {
        "m": int(m),
        "mode": mode,
        "matrix_profile_summary": _profile_summary(finite_profile),
        "motif_pair": _segment_pair(motif_index, nearest_index, float(finite_profile[motif_index]), m, stream),
        "discord_segment": _segment(discord_index, m, stream, distance=float(finite_profile[discord_index])),
        "nearest_neighbor_segment": _segment_pair(motif_index, nearest_index, float(finite_profile[motif_index]), m, stream),
        "mp_discord_isolation_value": _discord_isolation(finite_profile, discord_index),
        "fluss_regimes": _try_fluss(stumpy_module, finite_profile, m),
        "snippets": _try_snippets(stumpy_module, matrix, m),
    }
    return result


def _profile_vector(matrix_profile: object) -> np.ndarray:
    values = np.asarray(matrix_profile, dtype=np.float64)
    if values.ndim == 1:
        return values
    return np.nanmin(values, axis=0)


def _index_vector(indices: object) -> np.ndarray:
    values = np.asarray(indices, dtype=np.float64)
    if values.ndim == 1:
        return values
    return values[0, :]


def _first_component(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 1:
        return matrix[:, 0]
    _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    return matrix @ vt[0, :].T


def _segment_pair(
    left_index: int,
    right_index: int | None,
    distance: float,
    m: int,
    stream: PreprocessedFusionStream,
) -> dict[str, object]:
    return {
        "left": _segment(left_index, m, stream, distance=distance),
        "right": None if right_index is None or right_index < 0 else _segment(right_index, m, stream, distance=distance),
        "distance": float(distance),
    }


def _segment(start_index: int, m: int, stream: PreprocessedFusionStream, *, distance: float | None = None) -> dict[str, object]:
    start = min(max(int(start_index), 0), max(stream.T - 1, 0))
    end = min(start + int(m), stream.T)
    payload = {
        "start_index": start,
        "end_index": end,
        "start_window_id": stream.window_ids[start],
        "end_window_id": stream.window_ids[end - 1] if end > start else stream.window_ids[start],
        "start_time": float(stream.times[start]),
        "end_time": float(stream.times[end - 1] if end > start else stream.times[start]),
    }
    if distance is not None:
        payload["distance"] = float(distance)
    return payload


def _profile_summary(profile: np.ndarray) -> dict[str, object]:
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return {"finite_count": 0}
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _discord_isolation(profile: np.ndarray, discord_index: int) -> float:
    finite = profile[np.isfinite(profile)]
    if finite.size < 2:
        return 0.0
    std = float(np.std(finite))
    if std < 1e-12:
        return 0.0
    return float((profile[discord_index] - np.mean(finite)) / std)


def _try_fluss(stumpy_module: object, profile: np.ndarray, m: int) -> list[int]:
    if not hasattr(stumpy_module, "fluss") or profile.size < max(6, m + 3):
        return []
    try:
        _cac, regimes = stumpy_module.fluss(profile, L=m, n_regimes=2, excl_factor=1)
        return [int(value) for value in np.asarray(regimes).reshape(-1) if np.isfinite(value)]
    except Exception:
        return []


def _try_snippets(stumpy_module: object, matrix: np.ndarray, m: int) -> list[dict[str, object]]:
    if not hasattr(stumpy_module, "snippets") or matrix.shape[0] < max(6, m + 3):
        return []
    try:
        snippet_output = stumpy_module.snippets(_first_component(matrix), m=m, k=2)
    except Exception:
        return []
    values = np.asarray(snippet_output[0] if isinstance(snippet_output, tuple) else snippet_output)
    return [{"index": int(index), "value": float(value)} for index, value in enumerate(values.reshape(-1)[:4])]


def _base_result(stream: PreprocessedFusionStream) -> dict[str, object]:
    return {
        "method_name": stream.record.method_name,
        "sortie_id": stream.record.sortie_id,
        "view_id": stream.record.view_id,
        "T": stream.T,
        "d": stream.d,
        "matrix_profile_summary": {},
        "motif_pair": {},
        "discord_segment": {},
        "nearest_neighbor_segment": {},
        "fluss_regimes": [],
        "snippets": [],
    }
