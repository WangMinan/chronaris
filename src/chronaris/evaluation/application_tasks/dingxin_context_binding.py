"""Audit lazy Dingxin raw contexts against independent target archives."""

from __future__ import annotations

import json
import logging
import resource
import time
from dataclasses import dataclass
from pathlib import Path

from chronaris.evaluation.application_tasks.dingxin_context_audit import (
    build_dingxin_context_acceptance_rows,
)
from chronaris.evaluation.application_tasks.dingxin_context_data import (
    audit_lazy_dingxin_contexts,
    build_dingxin_lazy_context_index,
    build_fold_task_context_bindings,
    schema_manifest,
)
from chronaris.evaluation.application_tasks.dingxin_context_reporting import (
    write_dingxin_context_outputs,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_context_binding")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class DingxinContextBindingConfig:
    run_id: str = "2026-07-11_dingxin-context-bindings"
    output_root: str = "docs/artifacts/runs"
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    target_run_root: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-application-targets"
    )


@dataclass(frozen=True, slots=True)
class DingxinContextBindingResult:
    run_id: str
    status: str
    run_root: str
    context_count: int
    input_available_context_count: int
    classification_available_context_count: int
    response_available_context_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_context_binding_audit(
    config: DingxinContextBindingConfig,
) -> DingxinContextBindingResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    fixed_root = Path(config.fixed_audit_root)
    target_root = Path(config.target_run_root)
    snapshot_root = Path(config.snapshot_root)
    source_paths = {
        "field_role_manifest": fixed_root / "field_role_manifest.csv",
        "context_manifest": fixed_root / "context_sample_manifest.jsonl",
        "split_manifest": fixed_root / "split_manifest.json",
        "target_archive_manifest": target_root / "target_archive_manifest.csv",
        "snapshot_manifest": snapshot_root / "snapshot_manifest.json",
    }
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="dingxin_lazy_context_bindings",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "precomputed_dense_context_bundle": False,
        },
    ) as progress:
        started = time.perf_counter()
        missing = sorted(name for name, path in source_paths.items() if not path.is_file())
        if missing:
            raise FileNotFoundError(f"Dingxin context binding sources missing: {missing}")
        snapshot_manifest = json.loads(
            source_paths["snapshot_manifest"].read_text(encoding="utf-8")
        )
        snapshot_hashes_before = {
            item["relative_path"]: sha256_file(
                snapshot_root / item["relative_path"]
            )
            for item in snapshot_manifest["files"]
        }
        index = build_dingxin_lazy_context_index(
            snapshot_root=snapshot_root,
            field_role_manifest_path=source_paths["field_role_manifest"],
            context_manifest_path=source_paths["context_manifest"],
        )
        stream_frame = audit_lazy_dingxin_contexts(index)
        progress.update(
            "lazy_context_streams_audited",
            context_count=len(index.contexts),
            available_count=int(stream_frame["status"].eq("completed").sum()),
        )
        binding_frame, archive_verification = build_fold_task_context_bindings(
            index=index,
            target_archive_manifest_path=source_paths[
                "target_archive_manifest"
            ],
        )
        schema_payload = schema_manifest(index)
        snapshot_hashes_after = {
            item["relative_path"]: sha256_file(
                snapshot_root / item["relative_path"]
            )
            for item in snapshot_manifest["files"]
        }
        acceptance_rows = build_dingxin_context_acceptance_rows(
            index=index,
            stream_rows=stream_frame,
            binding_rows=binding_frame,
            archive_verification=archive_verification,
            schema_payload=schema_payload,
            snapshot_hashes_unchanged=(
                snapshot_hashes_before == snapshot_hashes_after
            ),
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        resource_rows = [
            {
                "stage": "lazy_context_binding_audit",
                "elapsed_s": time.perf_counter() - started,
                "maximum_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
                "context_count": len(index.contexts),
                "cached_sparse_array_bytes": schema_payload[
                    "cached_sparse_array_bytes"
                ],
                "precomputed_dense_context_bundle": False,
            }
        ]
        paths = write_dingxin_context_outputs(
            run_root=run_root,
            run_id=config.run_id,
            status=status,
            source_manifest={
                "format": "chronaris.dingxin_context_binding_sources.v1",
                "source_hashes": {
                    name: sha256_file(path) for name, path in source_paths.items()
                },
                "snapshot_file_hashes_before": snapshot_hashes_before,
                "snapshot_file_hashes_after": snapshot_hashes_after,
                "target_archive_count": len(archive_verification),
                "raw_values_committed": False,
            },
            schema_payload=schema_payload,
            context_rows=index.contexts.to_dict("records"),
            stream_rows=stream_frame.to_dict("records"),
            binding_rows=binding_frame.to_dict("records"),
            archive_verification=archive_verification,
            resource_rows=resource_rows,
            acceptance_rows=acceptance_rows,
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        classification_available = binding_frame[
            (binding_frame["task_slug"] == "maneuver_intensity_classification")
            & (binding_frame["binding_status"] == "available")
        ]["context_id"].nunique()
        response_available = binding_frame[
            (binding_frame["task_slug"] == "physiology_response_prediction")
            & (binding_frame["binding_status"] == "available")
        ]["context_id"].nunique()
        input_available = stream_frame[stream_frame["status"] == "completed"][
            "context_id"
        ].nunique()
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            input_available_context_count=int(input_available),
        )
        return DingxinContextBindingResult(
            run_id=config.run_id,
            status=status,
            run_root=str(run_root),
            context_count=len(index.contexts),
            input_available_context_count=int(input_available),
            classification_available_context_count=int(classification_available),
            response_available_context_count=int(response_available),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )
