"""Stage I Phase 2 helpers for consuming frozen Stage H assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.features.stage_h_bundle import StageHFeatureRun, StageHFeatureView, load_stage_h_feature_run


@dataclass(frozen=True, slots=True)
class StageICaseStudyWindowRow:
    """One window-manifest row associated with a Stage H view."""

    sample_id: str
    sortie_id: str
    window_index: int
    start_offset_ms: int
    end_offset_ms: int
    physiology_point_count: int
    vehicle_point_count: int
    selected_for_model: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "sortie_id": self.sortie_id,
            "window_index": self.window_index,
            "start_offset_ms": self.start_offset_ms,
            "end_offset_ms": self.end_offset_ms,
            "physiology_point_count": self.physiology_point_count,
            "vehicle_point_count": self.vehicle_point_count,
            "selected_for_model": self.selected_for_model,
        }


@dataclass(frozen=True, slots=True)
class StageICaseStudyViewInput:
    """One Phase 2 view assembled from a Stage H bundle plus sidecars."""

    view_id: str
    sortie_id: str
    pilot_id: int
    projection_diagnostics_verdict: str
    stage_h_view: StageHFeatureView
    sample_ids: tuple[str, ...]
    all_window_rows: tuple[StageICaseStudyWindowRow, ...]
    case_window_rows: tuple[StageICaseStudyWindowRow, ...]
    projection_summary: Mapping[str, object]
    threshold_evaluation: Mapping[str, object]
    causal_summary: Mapping[str, object]
    intermediate_summary: Mapping[str, object]

    @property
    def window_count(self) -> int:
        return len(self.all_window_rows)

    @property
    def selected_window_count(self) -> int:
        return sum(1 for row in self.all_window_rows if row.selected_for_model)

    @property
    def case_partition_sample_count(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class StageICaseStudyRunInput:
    """All Phase 2 views derived from one Stage H run."""

    run_manifest_path: str
    stage_h_run: StageHFeatureRun
    views: tuple[StageICaseStudyViewInput, ...]


def load_stage_i_case_study_run(run_manifest_path: str | Path) -> StageICaseStudyRunInput:
    """Load all Phase 2-consumable inputs from one Stage H run manifest."""

    stage_h_run = load_stage_h_feature_run(run_manifest_path)
    views = tuple(_load_case_view(view) for view in stage_h_run.views)
    return StageICaseStudyRunInput(
        run_manifest_path=str(Path(run_manifest_path)),
        stage_h_run=stage_h_run,
        views=views,
    )


def _load_case_view(view: StageHFeatureView) -> StageICaseStudyViewInput:
    manifest_dir = Path(view.manifest_path).parent
    artifact_paths = view.view_manifest.get("artifact_paths", {})
    if not isinstance(artifact_paths, Mapping):
        raise ValueError("view manifest artifact_paths must be a mapping.")

    projection_summary = _read_json(artifact_paths.get("projection_diagnostics_summary_json"))
    causal_summary = _read_json(artifact_paths.get("causal_fusion_summary_json"))
    intermediate_summary = _read_json(artifact_paths.get("intermediate_summary_json"))
    threshold_evaluation = projection_summary.get("threshold_evaluation", {})
    sample_ids = _resolve_case_sample_ids(projection_summary, causal_summary)
    if sample_ids and view.fused_representation.shape[0] != len(sample_ids):
        raise ValueError(
            f"view {view.view_id} has {view.fused_representation.shape[0]} fused samples but "
            f"{len(sample_ids)} sample ids in sidecars."
        )

    all_window_rows = tuple(
        _window_row_from_dict(row)
        for row in _read_jsonl(
            _resolve_path(
                artifact_paths.get("window_manifest_jsonl"),
                base_dir=manifest_dir,
            )
        )
    )
    row_by_sample_id = {row.sample_id: row for row in all_window_rows}
    case_window_rows = tuple(row_by_sample_id[sample_id] for sample_id in sample_ids if sample_id in row_by_sample_id)
    if sample_ids and len(case_window_rows) != len(sample_ids):
        missing = tuple(sample_id for sample_id in sample_ids if sample_id not in row_by_sample_id)
        raise ValueError(f"window manifest is missing case-study sample ids: {', '.join(missing)}")

    return StageICaseStudyViewInput(
        view_id=view.view_id,
        sortie_id=view.sortie_id,
        pilot_id=view.pilot_id,
        projection_diagnostics_verdict=view.projection_diagnostics_verdict,
        stage_h_view=view,
        sample_ids=sample_ids,
        all_window_rows=all_window_rows,
        case_window_rows=case_window_rows,
        projection_summary=projection_summary,
        threshold_evaluation=threshold_evaluation,
        causal_summary=causal_summary,
        intermediate_summary=intermediate_summary,
    )


def _resolve_case_sample_ids(
    projection_summary: Mapping[str, object],
    causal_summary: Mapping[str, object],
) -> tuple[str, ...]:
    projection_samples = projection_summary.get("summary", {}).get("samples", ())
    causal_samples = causal_summary.get("samples", ())
    projection_ids = tuple(
        str(sample.get("sample_id"))
        for sample in projection_samples
        if isinstance(sample, Mapping) and sample.get("sample_id")
    )
    causal_ids = tuple(
        str(sample.get("sample_id"))
        for sample in causal_samples
        if isinstance(sample, Mapping) and sample.get("sample_id")
    )
    if projection_ids and causal_ids and projection_ids != causal_ids:
        raise ValueError("projection diagnostics and causal fusion sample orders do not match.")
    if projection_ids:
        return projection_ids
    return causal_ids


def _window_row_from_dict(row: Mapping[str, object]) -> StageICaseStudyWindowRow:
    return StageICaseStudyWindowRow(
        sample_id=str(row["sample_id"]),
        sortie_id=str(row["sortie_id"]),
        window_index=int(row["window_index"]),
        start_offset_ms=int(row["start_offset_ms"]),
        end_offset_ms=int(row["end_offset_ms"]),
        physiology_point_count=int(row["physiology_point_count"]),
        vehicle_point_count=int(row["vehicle_point_count"]),
        selected_for_model=bool(row["selected_for_model"]),
    )


def _read_json(path_like: object) -> Mapping[str, object]:
    path = _resolve_path(path_like)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _resolve_path(path_like: object, *, base_dir: Path | None = None) -> Path:
    if not isinstance(path_like, (str, Path)) or not path_like:
        raise ValueError("artifact path must be a non-empty string or Path.")
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        sibling = base_dir / candidate.name
        if sibling.exists():
            return sibling
    return Path.cwd() / candidate
