"""Build a fixed snapshot plan from the paired E/F feature-export manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.dataset.application_evaluation.snapshot_contracts import (
    FIXED_DINGXIN_SORTIE_IDS,
    SnapshotViewPlan,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import load_aligned_private_records


def build_fixed_snapshot_plan(
    *,
    e_run_manifest_path: str | Path,
    f_run_manifest_path: str | Path,
    allowed_sortie_ids: Sequence[str] = FIXED_DINGXIN_SORTIE_IDS,
) -> tuple[SnapshotViewPlan, ...]:
    """Resolve exact view, scope, measurement and expected-count contracts."""

    e_path = Path(e_run_manifest_path)
    f_path = Path(f_run_manifest_path)
    e_manifest = _read_json(e_path)
    f_manifest = _read_json(f_path)
    _validate_paired_run_manifests(e_manifest, f_manifest, allowed_sortie_ids)
    records = load_aligned_private_records(
        e_run_manifest_path=e_path,
        f_run_manifest_path=f_path,
    )
    plans: list[SnapshotViewPlan] = []
    scope_by_sortie = _scope_by_sortie(e_manifest)
    sortie_paths = dict(e_manifest["sortie_manifest_paths"])
    for sortie_id in e_manifest["sortie_ids"]:
        sortie_manifest = _read_json(_resolve_repo_path(str(sortie_paths[sortie_id])))
        start_utc, stop_utc = scope_by_sortie[str(sortie_id)]
        for view_id in sortie_manifest["exported_view_ids"]:
            view_frame = records.loc[records["view_id"].astype(str) == str(view_id)]
            if view_frame.empty:
                raise ValueError(f"reference records missing view {view_id}")
            view_manifest_path = sortie_manifest["view_manifest_paths"][view_id]
            view_manifest = _read_json(_resolve_repo_path(str(view_manifest_path)))
            if (
                _utc(view_manifest["export_start_utc"]) != start_utc
                or _utc(view_manifest["export_stop_utc"]) != stop_utc
            ):
                raise ValueError(f"view scope differs from run scope: {view_id}")
            plans.append(
                SnapshotViewPlan(
                    sortie_id=str(sortie_id),
                    view_id=str(view_id),
                    pilot_id=int(view_manifest["pilot_id"]),
                    start_utc=start_utc.isoformat(),
                    stop_utc=stop_utc.isoformat(),
                    physiology_measurements=tuple(view_manifest["physiology_measurements"]),
                    vehicle_measurements=tuple(view_manifest["vehicle_measurements"]),
                    expected_physiology_point_count=int(
                        view_frame["physiology_point_count"].sum()
                    ),
                    expected_vehicle_point_count=int(view_frame["vehicle_point_count"].sum()),
                )
            )
    return tuple(plans)


def source_manifest_hashes(
    e_run_manifest_path: str | Path,
    f_run_manifest_path: str | Path,
) -> dict[str, str]:
    return {
        "e_run_manifest_sha256": _sha256(Path(e_run_manifest_path)),
        "f_run_manifest_sha256": _sha256(Path(f_run_manifest_path)),
    }


def _validate_paired_run_manifests(
    e_manifest: Mapping[str, object],
    f_manifest: Mapping[str, object],
    allowed_sortie_ids: Sequence[str],
) -> None:
    e_sorties = tuple(str(value) for value in e_manifest["sortie_ids"])
    f_sorties = tuple(str(value) for value in f_manifest["sortie_ids"])
    if e_sorties != f_sorties:
        raise ValueError("E/F run manifests contain different sortie order")
    if set(e_sorties) != set(allowed_sortie_ids):
        raise ValueError("run manifests do not contain exactly the fixed Dingxin sortie whitelist")
    e_config = dict(e_manifest["config"])
    f_config = dict(f_manifest["config"])
    for key in (
        "window_duration_ms",
        "window_stride_ms",
        "export_scope_overrides_utc",
    ):
        if e_config.get(key) != f_config.get(key):
            raise ValueError(f"E/F run manifests differ on {key}")


def _scope_by_sortie(manifest: Mapping[str, object]) -> dict[str, tuple[datetime, datetime]]:
    raw_scopes = dict(dict(manifest["config"])["export_scope_overrides_utc"])
    return {
        str(sortie_id): (_utc(bounds[0]), _utc(bounds[1]))
        for sortie_id, bounds in raw_scopes.items()
    }


def _utc(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
