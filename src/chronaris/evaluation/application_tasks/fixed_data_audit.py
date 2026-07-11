"""Leakage-safe fixed-data audit for Dingxin application tasks."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chronaris.access.mysql_cli import SQLQueryRunner
from chronaris.access.mysql_metadata import (
    MySQLFlightTaskReader,
    MySQLRealBusContextReader,
    MySQLStorageAnalysisReader,
)
from chronaris.dataset.application_evaluation import (
    build_application_contexts,
    build_field_role_manifest,
    build_fold_task_labels,
    build_outer_folds,
    contexts_to_frame,
    selected_maneuver_roles,
    selected_response_roles,
)
from chronaris.evaluation.application_tasks.fixed_data_reporting import write_fixed_data_audit_outputs
from chronaris.evaluation.dingxin.pipelines.benchmark_data import load_aligned_private_records
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.schema.models import SortieLocator


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.fixed_data_audit")
LOGGER.addHandler(logging.NullHandler())

DEFAULT_E_MANIFEST = "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
DEFAULT_F_MANIFEST = "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
DEFAULT_RUN_ID = "2026-07-10_fixed-data-audit"


@dataclass(frozen=True, slots=True)
class FixedDataAuditConfig:
    """Configuration for the no-training G1 fixed-data audit."""

    run_id: str = DEFAULT_RUN_ID
    output_root: str = "docs/artifacts/runs"
    e_run_manifest_path: str = DEFAULT_E_MANIFEST
    f_run_manifest_path: str = DEFAULT_F_MANIFEST
    bus_access_rule_id: int = 6000019510066
    strict_mysql_field_labels: bool = True


@dataclass(frozen=True, slots=True)
class FixedDataAuditResult:
    run_id: str
    run_root: str
    status: str
    report_path: str
    evidence_manifest_path: str
    classification_context_count: int
    response_context_count: int
    selected_maneuver_field_count: int
    selected_response_field_count: int
    fold_count: int


def run_fixed_data_audit(
    config: FixedDataAuditConfig,
    *,
    mysql_runner: SQLQueryRunner | None,
) -> FixedDataAuditResult:
    """Run G1 without model training or confirmed-metric mutation."""

    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="fixed_data_application_audit",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "e_run_manifest_path": config.e_run_manifest_path,
            "f_run_manifest_path": config.f_run_manifest_path,
        },
    ) as progress:
        records = load_aligned_private_records(
            e_run_manifest_path=config.e_run_manifest_path,
            f_run_manifest_path=config.f_run_manifest_path,
        )
        progress.update(
            "records_loaded",
            record_count=int(len(records)),
            sortie_count=int(records["sortie_id"].nunique()),
            view_count=int(records["view_id"].nunique()),
        )
        labels_by_sortie, metadata_rows, metadata_errors = _load_vehicle_field_metadata(
            records,
            mysql_runner=mysql_runner,
            bus_access_rule_id=config.bus_access_rule_id,
        )
        if config.strict_mysql_field_labels:
            missing_sorties = sorted(
                sortie_id
                for sortie_id in records["sortie_id"].astype(str).unique()
                if not labels_by_sortie.get(sortie_id)
            )
            if missing_sorties:
                raise RuntimeError(
                    "vehicle field metadata unavailable for sorties: " + ", ".join(missing_sorties)
                )
            if metadata_errors:
                raise RuntimeError(
                    f"vehicle field metadata has {len(metadata_errors)} analysis errors in strict mode"
                )
        field_roles = build_field_role_manifest(
            records,
            vehicle_labels_by_sortie=labels_by_sortie,
        )
        maneuver_roles = selected_maneuver_roles(field_roles)
        response_roles = selected_response_roles(field_roles)
        response_fields = {role.feature_name for role in response_roles}
        progress.update(
            "field_roles_resolved",
            metadata_error_count=len(metadata_errors),
            field_role_count=len(field_roles),
            selected_maneuver_field_count=len(maneuver_roles),
            selected_response_role_count=len(response_roles),
            selected_response_field_count=len(response_fields),
        )

        contexts = build_application_contexts(records)
        context_frame = contexts_to_frame(contexts)
        classification_count = int(context_frame["classification_eligible"].sum())
        response_count = int(context_frame["response_eligible"].sum())
        folds = tuple(
            fold
            for strategy in ("leave_one_view_out", "leave_one_sortie_out")
            for fold in build_outer_folds(contexts, split_strategy=strategy)
        )
        progress.update(
            "contexts_and_splits_built",
            context_candidate_count=len(contexts),
            classification_context_count=classification_count,
            response_context_count=response_count,
            fold_count=len(folds),
        )

        fold_results = tuple(
            build_fold_task_labels(records, contexts, fold, field_roles)
            for fold in folds
        )
        status = "completed" if all(result.status == "completed" for result in fold_results) else "partial"
        output_paths = write_fixed_data_audit_outputs(
            config=config,
            run_root=run_root,
            records=records,
            contexts=contexts,
            field_roles=field_roles,
            folds=folds,
            fold_results=fold_results,
            metadata_rows=metadata_rows,
            metadata_errors=metadata_errors,
            source_hashes={
                "e_run_manifest_sha256": _sha256(Path(config.e_run_manifest_path)),
                "f_run_manifest_sha256": _sha256(Path(config.f_run_manifest_path)),
            },
            status=status,
        )
        progress.finish(
            status=status,
            classification_context_count=classification_count,
            response_context_count=response_count,
            selected_maneuver_field_count=len(maneuver_roles),
            selected_response_role_count=len(response_roles),
            selected_response_field_count=len(response_fields),
            fold_count=len(folds),
            report_path=output_paths["report_path"],
            evidence_manifest_path=output_paths["evidence_manifest_path"],
        )
        return FixedDataAuditResult(
            run_id=config.run_id,
            run_root=str(run_root),
            status=status,
            report_path=output_paths["report_path"],
            evidence_manifest_path=output_paths["evidence_manifest_path"],
            classification_context_count=classification_count,
            response_context_count=response_count,
            selected_maneuver_field_count=len(maneuver_roles),
            selected_response_field_count=len(response_fields),
            fold_count=len(folds),
        )


def _load_vehicle_field_metadata(
    records: pd.DataFrame,
    *,
    mysql_runner: SQLQueryRunner | None,
    bus_access_rule_id: int,
) -> tuple[dict[str, dict[str, str]], list[dict[str, object]], list[dict[str, object]]]:
    if mysql_runner is None:
        return {}, [], [{"status": "unavailable", "reason": "mysql_runner_not_provided"}]
    storage_reader = MySQLStorageAnalysisReader(mysql_runner)
    context_reader = MySQLRealBusContextReader(
        runner=mysql_runner,
        flight_task_reader=MySQLFlightTaskReader(mysql_runner),
    )
    labels_by_sortie: dict[str, dict[str, str]] = {}
    metadata_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for sortie_id in sorted(records["sortie_id"].astype(str).unique()):
        labels: dict[str, str] = {}
        analyses = storage_reader.list_for_sortie(
            SortieLocator(sortie_id=sortie_id),
            category="BUS",
        )
        for analysis in analyses:
            try:
                context = context_reader.fetch_context(
                    locator=SortieLocator(sortie_id=sortie_id),
                    access_rule_id=bus_access_rule_id,
                    analysis_id=analysis.analysis_id,
                )
            except Exception as exc:  # pragma: no cover - live error path.
                errors.append(
                    {
                        "sortie_id": sortie_id,
                        "analysis_id": analysis.analysis_id,
                        "measurement": analysis.measurement,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue
            measurement = context.analysis.measurement or analysis.measurement or ""
            source_rows = []
            source_rows.extend((detail.col_field, detail.col_name, "analysis_detail") for detail in context.detail_list)
            source_rows.extend((detail.col_field, detail.col_name, "storage_structure") for detail in context.structure_list)
            source_rows.extend(
                (detail.col_field, detail.col_name, "access_rule_detail")
                for detail in context.access_rule_details
                if detail.col_name
            )
            for source_field, display_label, metadata_source in source_rows:
                feature_name = f"{measurement}.{source_field}"
                labels.setdefault(feature_name, str(display_label))
                metadata_rows.append(
                    {
                        "sortie_id": sortie_id,
                        "analysis_id": analysis.analysis_id,
                        "measurement": measurement,
                        "source_field": source_field,
                        "feature_name": feature_name,
                        "display_label": display_label,
                        "metadata_source": metadata_source,
                    }
                )
        labels_by_sortie[sortie_id] = labels
    return labels_by_sortie, metadata_rows, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
