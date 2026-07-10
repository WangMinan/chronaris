"""Freeze the exact existing Dingxin raw inputs without expanding data scope."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.dataset.application_evaluation import (
    FIXED_DINGXIN_SORTIE_IDS,
    SnapshotFileRecord,
    SnapshotRunResult,
    SnapshotViewPlan,
    sha256_file,
    write_raw_point_snapshot,
)
from chronaris.evaluation.application_tasks.fixed_data_audit import (
    DEFAULT_E_MANIFEST,
    DEFAULT_F_MANIFEST,
)
from chronaris.evaluation.application_tasks.fixed_data_snapshot_reporting import (
    write_snapshot_compact_outputs,
)
from chronaris.evaluation.application_tasks.fixed_data_snapshot_plan import (
    build_fixed_snapshot_plan,
    source_manifest_hashes,
)
from chronaris.evaluation.application_tasks.snapshot_live_source import SnapshotPointSource
from chronaris.feature_export.profile import StageHSortieProfile
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.schema.models import RawPoint, StreamKind


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.fixed_data_snapshot")
LOGGER.addHandler(logging.NullHandler())

DEFAULT_SNAPSHOT_RUN_ID = "2026-07-10_dingxin-input-snapshot"
DEFAULT_LABEL_FIELD_MANIFEST = (
    "docs/artifacts/runs/2026-07-10_fixed-data-audit/label_field_manifest.json"
)


@dataclass(frozen=True, slots=True)
class FixedDataSnapshotConfig:
    run_id: str = DEFAULT_SNAPSHOT_RUN_ID
    snapshot_output_root: str = "artifacts/application_evaluation"
    compact_output_root: str = "docs/artifacts/runs"
    e_run_manifest_path: str = DEFAULT_E_MANIFEST
    f_run_manifest_path: str = DEFAULT_F_MANIFEST
    label_field_manifest_path: str = DEFAULT_LABEL_FIELD_MANIFEST
    allowed_sortie_ids: tuple[str, ...] = FIXED_DINGXIN_SORTIE_IDS
    resume: bool = True


def run_fixed_data_snapshot(
    config: FixedDataSnapshotConfig,
    *,
    profile_resolver,
    point_source: SnapshotPointSource,
) -> SnapshotRunResult:
    """Read, serialize and audit the fixed raw inputs for both existing sorties."""

    snapshot_root = Path(config.snapshot_output_root) / config.run_id
    compact_root = Path(config.compact_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    source_hashes = source_manifest_hashes(
        config.e_run_manifest_path,
        config.f_run_manifest_path,
    )
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="fixed_data_raw_snapshot",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "snapshot_root": str(snapshot_root),
        },
    ) as progress:
        if config.resume:
            resumed_manifest = _validated_existing_manifest(
                snapshot_root,
                run_id=config.run_id,
                source_hashes=source_hashes,
            )
            if resumed_manifest is not None:
                consistency_rows = tuple(resumed_manifest["consistency_rows"])
                exclusion_rows = tuple(resumed_manifest["field_exclusion_rows"])
                output_paths = write_snapshot_compact_outputs(
                    config=config,
                    compact_run_root=compact_root,
                    snapshot_manifest=resumed_manifest,
                    consistency_rows=consistency_rows,
                    exclusion_rows=exclusion_rows,
                    status=str(resumed_manifest["status"]),
                    resumed=True,
                )
                progress.finish(status=resumed_manifest["status"], resumed=True)
                return _result_from_manifest(
                    config,
                    snapshot_root,
                    compact_root,
                    resumed_manifest,
                    output_paths,
                    resumed=True,
                )

        plans = build_fixed_snapshot_plan(
            e_run_manifest_path=config.e_run_manifest_path,
            f_run_manifest_path=config.f_run_manifest_path,
            allowed_sortie_ids=config.allowed_sortie_ids,
        )
        profiles = tuple(profile_resolver.resolve_many(config.allowed_sortie_ids))
        _validate_profiles(profiles, plans)
        progress.update(
            "snapshot_plan_resolved",
            sortie_count=len(profiles),
            view_count=len(plans),
        )

        file_records: list[SnapshotFileRecord] = []
        consistency_rows: list[dict[str, object]] = []
        observed_vehicle_fields: dict[str, set[str]] = {}
        plans_by_sortie = _plans_by_sortie(plans)
        for profile in profiles:
            sortie_plans = plans_by_sortie[profile.sortie_id]
            start_utc = _utc(sortie_plans[0].start_utc)
            stop_utc = _utc(sortie_plans[0].stop_utc)
            vehicle_points = tuple(
                point_source.fetch_vehicle(
                    profile,
                    start_utc=start_utc,
                    stop_utc=stop_utc,
                )
            )
            vehicle_path = snapshot_root / "sorties" / profile.sortie_id / "vehicle_points.jsonl.gz"
            vehicle_record = write_raw_point_snapshot(
                vehicle_path,
                vehicle_points,
                snapshot_root=snapshot_root,
                sortie_id=profile.sortie_id,
                view_id=None,
                pilot_id=None,
                expected_stream_kind=StreamKind.VEHICLE,
            )
            file_records.append(vehicle_record)
            observed_vehicle_fields[profile.sortie_id] = _observed_fields(vehicle_points)
            for plan in sortie_plans:
                consistency_rows.append(
                    _consistency_row(
                        plan=plan,
                        stream_kind="vehicle",
                        actual_point_count=vehicle_record.point_count,
                    )
                )
            view_by_id = {view.view_id: view for view in profile.views}
            for plan in sortie_plans:
                view = view_by_id[plan.view_id]
                physiology_points = tuple(
                    point_source.fetch_physiology(
                        profile,
                        view,
                        start_utc=start_utc,
                        stop_utc=stop_utc,
                    )
                )
                physiology_path = (
                    snapshot_root
                    / "sorties"
                    / profile.sortie_id
                    / "views"
                    / plan.view_id
                    / "physiology_points.jsonl.gz"
                )
                physiology_record = write_raw_point_snapshot(
                    physiology_path,
                    physiology_points,
                    snapshot_root=snapshot_root,
                    sortie_id=profile.sortie_id,
                    view_id=plan.view_id,
                    pilot_id=plan.pilot_id,
                    expected_stream_kind=StreamKind.PHYSIOLOGY,
                )
                file_records.append(physiology_record)
                consistency_rows.append(
                    _consistency_row(
                        plan=plan,
                        stream_kind="physiology",
                        actual_point_count=physiology_record.point_count,
                    )
                )
            progress.update(
                "sortie_snapshot_written",
                sortie_id=profile.sortie_id,
                vehicle_point_count=vehicle_record.point_count,
                completed_file_count=len(file_records),
            )

        exclusion_rows = _build_exclusion_rows(
            config.label_field_manifest_path,
            observed_vehicle_fields,
        )
        count_matches = all(bool(row["count_matches_reference"]) for row in consistency_rows)
        exclusions_observed = all(bool(row["observed_in_snapshot"]) for row in exclusion_rows)
        status = "completed" if count_matches and exclusions_observed else "partial"
        manifest = {
            "run_id": config.run_id,
            "status": status,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "snapshot_root": str(snapshot_root),
            "raw_values_committed": False,
            "allowed_sortie_ids": list(config.allowed_sortie_ids),
            "source_manifest_hashes": source_hashes,
            "source_run_manifests": [
                config.e_run_manifest_path,
                config.f_run_manifest_path,
            ],
            "plans": [plan.to_dict() for plan in plans],
            "files": [record.to_dict() for record in file_records],
            "consistency_rows": consistency_rows,
            "field_exclusion_rows": exclusion_rows,
        }
        snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_manifest_path = snapshot_root / "snapshot_manifest.json"
        snapshot_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths = write_snapshot_compact_outputs(
            config=config,
            compact_run_root=compact_root,
            snapshot_manifest=manifest,
            consistency_rows=consistency_rows,
            exclusion_rows=exclusion_rows,
            status=status,
            resumed=False,
        )
        progress.finish(
            status=status,
            resumed=False,
            file_count=len(file_records),
            point_count=sum(record.point_count for record in file_records),
        )
        return _result_from_manifest(
            config,
            snapshot_root,
            compact_root,
            manifest,
            output_paths,
            resumed=False,
        )


def _validate_profiles(
    profiles: Sequence[StageHSortieProfile],
    plans: Sequence[SnapshotViewPlan],
) -> None:
    plan_by_view = {plan.view_id: plan for plan in plans}
    if {profile.sortie_id for profile in profiles} != {plan.sortie_id for plan in plans}:
        raise ValueError("live profiles differ from the fixed snapshot sortie plan")
    for profile in profiles:
        for view in profile.views:
            plan = plan_by_view.get(view.view_id)
            if plan is None or plan.pilot_id != view.pilot_id:
                raise ValueError(f"live profile view differs from reference: {view.view_id}")
            if tuple(profile.model_physiology_measurements) != plan.physiology_measurements:
                raise ValueError(f"physiology measurements differ from reference: {view.view_id}")
            if tuple(profile.vehicle_measurements) != plan.vehicle_measurements:
                raise ValueError(f"vehicle measurements differ from reference: {view.view_id}")


def _plans_by_sortie(
    plans: Sequence[SnapshotViewPlan],
) -> dict[str, tuple[SnapshotViewPlan, ...]]:
    result = {}
    for sortie_id in sorted({plan.sortie_id for plan in plans}):
        result[sortie_id] = tuple(plan for plan in plans if plan.sortie_id == sortie_id)
    return result


def _consistency_row(
    *,
    plan: SnapshotViewPlan,
    stream_kind: str,
    actual_point_count: int,
) -> dict[str, object]:
    expected = (
        plan.expected_physiology_point_count
        if stream_kind == "physiology"
        else plan.expected_vehicle_point_count
    )
    return {
        "sortie_id": plan.sortie_id,
        "view_id": plan.view_id,
        "pilot_id": plan.pilot_id,
        "stream_kind": stream_kind,
        "start_utc": plan.start_utc,
        "stop_utc": plan.stop_utc,
        "expected_point_count": expected,
        "actual_point_count": actual_point_count,
        "point_count_delta": actual_point_count - expected,
        "count_matches_reference": actual_point_count == expected,
    }


def _build_exclusion_rows(
    label_field_manifest_path: str | Path,
    observed_vehicle_fields: Mapping[str, set[str]],
) -> list[dict[str, object]]:
    payload = json.loads(Path(label_field_manifest_path).read_text(encoding="utf-8"))
    rows = []
    for role in payload["maneuver_label_fields"]:
        sortie_id = str(role["sortie_id"])
        feature_name = str(role["feature_name"])
        rows.append(
            {
                "sortie_id": sortie_id,
                "feature_name": feature_name,
                "display_label": role.get("display_label"),
                "semantic_key": role.get("semantic_key"),
                "observed_in_snapshot": feature_name in observed_vehicle_fields.get(sortie_id, set()),
                "allowed_in_maneuver_input": False,
                "derived_feature_policy": "exclude_raw_statistics_differences_rates_and_normalized_copies",
                "exclusion_reason": "maneuver_label_source",
            }
        )
    return rows


def _observed_fields(points: Sequence[RawPoint]) -> set[str]:
    return {
        f"{point.measurement}.{field_name}"
        for point in points
        for field_name in point.values
    }


def _validated_existing_manifest(
    snapshot_root: Path,
    *,
    run_id: str,
    source_hashes: Mapping[str, str],
) -> Mapping[str, object] | None:
    path = snapshot_root / "snapshot_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or manifest.get("source_manifest_hashes") != dict(source_hashes):
        return None
    for file_record in manifest.get("files", []):
        file_path = snapshot_root / str(file_record["relative_path"])
        if not file_path.exists() or sha256_file(file_path) != file_record["sha256"]:
            return None
    return manifest


def _result_from_manifest(
    config: FixedDataSnapshotConfig,
    snapshot_root: Path,
    compact_root: Path,
    manifest: Mapping[str, object],
    output_paths: Mapping[str, str],
    *,
    resumed: bool,
) -> SnapshotRunResult:
    files = list(manifest["files"])
    return SnapshotRunResult(
        run_id=config.run_id,
        status=str(manifest["status"]),
        snapshot_root=str(snapshot_root),
        compact_run_root=str(compact_root),
        snapshot_manifest_path=str(snapshot_root / "snapshot_manifest.json"),
        compact_manifest_path=output_paths["compact_manifest_path"],
        report_path=output_paths["report_path"],
        evidence_manifest_path=output_paths["evidence_manifest_path"],
        file_count=len(files),
        physiology_point_count=sum(
            int(record["point_count"]) for record in files if record["stream_kind"] == "physiology"
        ),
        vehicle_point_count=sum(
            int(record["point_count"]) for record in files if record["stream_kind"] == "vehicle"
        ),
        resumed=resumed,
    )


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
