"""Readers for Stage H exported feature bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

STAGE_H_FEATURE_KEYS = (
    "physiology_reference_projection",
    "vehicle_reference_projection",
    "fused_representation",
    "reference_offsets_s",
    "attention_weights",
    "vehicle_event_scores",
)


@dataclass(frozen=True, slots=True)
class StageHFeatureView:
    """One view-level Stage H feature bundle plus manifest metadata."""

    view_id: str
    sortie_id: str
    pilot_id: int
    manifest_path: str
    feature_bundle_path: str
    projection_diagnostics_verdict: str
    physiology_reference_projection: np.ndarray
    vehicle_reference_projection: np.ndarray
    fused_representation: np.ndarray
    reference_offsets_s: np.ndarray
    attention_weights: np.ndarray
    vehicle_event_scores: np.ndarray
    view_manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StageHFeatureRun:
    """A loaded Stage H run with all view-level feature bundles."""

    run_manifest_path: str
    run_manifest: Mapping[str, object]
    views: tuple[StageHFeatureView, ...]

    @property
    def generated_view_count(self) -> int:
        return len(self.views)


def load_stage_h_feature_run(run_manifest_path: str | Path) -> StageHFeatureRun:
    """Load all view feature bundles referenced by a Stage H run manifest."""

    manifest_path = Path(run_manifest_path)
    run_manifest = _read_json(manifest_path)
    run_root = manifest_path.parent
    output_root = run_manifest.get("output_root")
    views: list[StageHFeatureView] = []
    for sortie_manifest_path in run_manifest.get("sortie_manifest_paths", {}).values():
        resolved_sortie_manifest_path = _resolve_artifact_path(
            sortie_manifest_path,
            output_root=output_root,
            run_root=run_root,
        )
        sortie_manifest = _read_json(resolved_sortie_manifest_path)
        for view_manifest_path in sortie_manifest.get("view_manifest_paths", {}).values():
            resolved_view_manifest_path = _resolve_artifact_path(
                view_manifest_path,
                output_root=output_root,
                run_root=run_root,
            )
            view_manifest = _read_json(resolved_view_manifest_path)
            views.append(
                _load_view(
                    view_manifest_path=str(resolved_view_manifest_path),
                    view_manifest=view_manifest,
                    output_root=output_root,
                    run_root=run_root,
                )
            )
    return StageHFeatureRun(
        run_manifest_path=str(manifest_path),
        run_manifest=run_manifest,
        views=tuple(views),
    )


def load_stage_h_feature_view(view_manifest_path: str | Path) -> StageHFeatureView:
    """Load one Stage H view feature bundle by its view manifest."""

    path = Path(view_manifest_path)
    return _load_view(
        view_manifest_path=str(path),
        view_manifest=_read_json(path),
        output_root=None,
        run_root=None,
    )


def _load_view(
    *,
    view_manifest_path: str,
    view_manifest: Mapping[str, object],
    output_root: object,
    run_root: Path | None,
) -> StageHFeatureView:
    artifact_paths = view_manifest.get("artifact_paths", {})
    if not isinstance(artifact_paths, Mapping):
        raise ValueError("view_manifest.artifact_paths must be a mapping.")
    feature_bundle_path = artifact_paths.get("feature_bundle_npz")
    if not isinstance(feature_bundle_path, str) or not feature_bundle_path:
        raise ValueError("view manifest is missing artifact_paths.feature_bundle_npz.")
    bundle = np.load(
        _resolve_artifact_path(
            feature_bundle_path,
            output_root=output_root,
            run_root=run_root,
            colocated_with=Path(view_manifest_path).parent,
        )
    )
    missing = tuple(key for key in STAGE_H_FEATURE_KEYS if key not in bundle.files)
    if missing:
        raise ValueError(f"Stage H feature bundle is missing keys: {', '.join(missing)}")
    return StageHFeatureView(
        view_id=str(view_manifest["view_id"]),
        sortie_id=str(view_manifest["sortie_id"]),
        pilot_id=int(view_manifest["pilot_id"]),
        manifest_path=view_manifest_path,
        feature_bundle_path=feature_bundle_path,
        projection_diagnostics_verdict=str(view_manifest["projection_diagnostics_verdict"]),
        physiology_reference_projection=bundle["physiology_reference_projection"],
        vehicle_reference_projection=bundle["vehicle_reference_projection"],
        fused_representation=bundle["fused_representation"],
        reference_offsets_s=bundle["reference_offsets_s"],
        attention_weights=bundle["attention_weights"],
        vehicle_event_scores=bundle["vehicle_event_scores"],
        view_manifest=view_manifest,
    )


def _read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(
    path: object,
    *,
    output_root: object,
    run_root: Path | None,
    colocated_with: Path | None = None,
) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValueError("artifact path must be a string or Path.")
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    if isinstance(output_root, str) and run_root is not None:
        try:
            return run_root / candidate.relative_to(Path(output_root))
        except ValueError:
            pass
    if colocated_with is not None:
        sibling = colocated_with / candidate.name
        if sibling.exists():
            return sibling
    return cwd_candidate
