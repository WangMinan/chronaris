"""Preparation pipeline for Stage I deep-baseline sequence assets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from chronaris.dataset import (
    dump_stage_i_sequence_entries,
    dump_stage_i_sequence_summary,
    save_stage_i_sequence_bundle,
)
from chronaris.features import (
    prepare_nasa_sequences,
    prepare_stage_h_case_sequences,
    prepare_uab_sequences,
)
from chronaris.features.stage_i_sequences import (
    REAL_SORTIE_V1,
    STAGE_H_CASE_DATASET_ID,
    WINDOW_V2,
    StageISequencePreparationPayload,
)
from chronaris.pipelines.stage_i.common.run_observer import open_stage_i_run_observer

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageISequencePreparationConfig:
    dataset_id: str
    artifact_root: str
    dataset_root: str | None = None
    stage_h_run_manifest_path: str | None = None
    profile: str = WINDOW_V2
    target_steps: int = 64


@dataclass(frozen=True, slots=True)
class StageISequencePreparationRunResult:
    dataset_id: str
    artifact_root: str
    manifest_path: str
    bundle_path: str
    schema_path: str
    summary_path: str
    processing_diagnostics_path: str
    summary: dict[str, object]


def run_stage_i_sequence_preparation(
    config: StageISequencePreparationConfig,
) -> StageISequencePreparationRunResult:
    artifact_root = Path(config.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=artifact_root,
        run_id=artifact_root.name,
        stage_name="stage_i_sequence_preparation",
        logger=LOGGER,
        initial_progress={
            "dataset_id": config.dataset_id,
            "dataset_root": config.dataset_root,
            "artifact_root": str(artifact_root),
            "profile": config.profile,
            "target_steps": config.target_steps,
        },
    ) as progress:
        return _run_stage_i_sequence_preparation_observed(
            config=config,
            artifact_root=artifact_root,
            progress=progress,
        )


def _run_stage_i_sequence_preparation_observed(
    *,
    config: StageISequencePreparationConfig,
    artifact_root: Path,
    progress,
) -> StageISequencePreparationRunResult:
    LOGGER.info(
        "stage_i_sequence_preparation start dataset_id=%s artifact_root=%s dataset_root=%s",
        config.dataset_id,
        artifact_root,
        config.dataset_root,
    )
    payload = _prepare_payload(
        config,
        progress_callback=lambda event, fields: _record_sequence_preparation_progress(
            progress=progress,
            event=event,
            **fields,
        ),
    )
    manifest_path = artifact_root / "task_manifest.jsonl"
    bundle_path = artifact_root / "sequence_bundle.npz"
    schema_path = artifact_root / "sequence_schema.json"
    summary_path = artifact_root / "dataset_summary.json"
    processing_diagnostics_path = artifact_root / "processing_diagnostics.json"
    processing_diagnostics = _build_processing_diagnostics(payload)
    progress.update(
        "payload_ready",
        dataset_id=payload.summary.dataset_id,
        entry_count=len(payload.entries),
        modalities=list(payload.bundle.modality_arrays),
        label_distribution=payload.summary.to_dict()["label_distribution"],
    )
    entries = tuple(
        replace(entry, sequence_bundle_path=str(bundle_path))
        for entry in payload.entries
    )
    dump_stage_i_sequence_entries(entries, path=manifest_path)
    save_stage_i_sequence_bundle(payload.bundle, path=bundle_path)
    dump_stage_i_sequence_summary(payload.summary, path=summary_path)
    schema_path.write_text(
        json.dumps(payload.sequence_schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    processing_diagnostics_path.write_text(
        json.dumps(processing_diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.finish(
        manifest_path=str(manifest_path),
        bundle_path=str(bundle_path),
        schema_path=str(schema_path),
        summary_path=str(summary_path),
        processing_diagnostics_path=str(processing_diagnostics_path),
    )
    LOGGER.info(
        "stage_i_sequence_preparation finished dataset_id=%s summary_path=%s diagnostics_path=%s",
        payload.summary.dataset_id,
        summary_path,
        processing_diagnostics_path,
    )
    return StageISequencePreparationRunResult(
        dataset_id=config.dataset_id,
        artifact_root=str(artifact_root),
        manifest_path=str(manifest_path),
        bundle_path=str(bundle_path),
        schema_path=str(schema_path),
        summary_path=str(summary_path),
        processing_diagnostics_path=str(processing_diagnostics_path),
        summary=payload.summary.to_dict(),
    )


def _build_processing_diagnostics(
    payload: StageISequencePreparationPayload,
) -> dict[str, object]:
    schema = dict(payload.sequence_schema)
    modalities = tuple(payload.bundle.modality_arrays)
    context_feature_names: list[str] = []
    guard = schema.get("label_leakage_guard")
    if isinstance(guard, dict):
        context_feature_names = [str(value) for value in guard.get("context_feature_names", [])]
    summary = payload.summary.to_dict()
    extra = summary.get("extra_summary", {})
    processing = dict(extra.get("processing_diagnostics") or {})
    metadata_refs = []
    for entry in payload.entries:
        for key in ("csv_path", "source_csv", "relative_path"):
            value = entry.context_payload.get(key)
            if value:
                metadata_refs.append(str(value))
    return {
        "dataset_id": payload.summary.dataset_id,
        "profile": payload.summary.profile,
        "entry_count": payload.summary.entry_count,
        "csv_file_count": int(processing.get("csv_file_count", len(set(metadata_refs)))),
        "chunk_size": int(processing.get("chunk_size", payload.summary.sequence_length)),
        "window_count": int(processing.get("window_count", payload.summary.entry_count)),
        "primary_count": int(processing.get("primary_count", payload.summary.training_role_counts.get("primary", 0))),
        "background_count": int(processing.get(
            "background_count",
            payload.summary.training_role_counts.get("inventory_only", 0)
        )),
        "label_distribution": summary["label_distribution"],
        "modalities": list(processing.get("modalities", list(modalities))),
        "context_feature_names": list(
            processing.get("context_feature_names", context_feature_names)
        ),
        "adapter_contract": schema.get("adapter_contract")
        or summary.get("extra_summary", {}).get("adapter_contract"),
        "label_leakage_guard": schema.get("label_leakage_guard", {}),
    }


def _prepare_payload(
    config: StageISequencePreparationConfig,
    *,
    progress_callback=None,
) -> StageISequencePreparationPayload:
    normalized = config.dataset_id.strip().lower()
    if normalized == STAGE_H_CASE_DATASET_ID:
        if not config.stage_h_run_manifest_path:
            raise ValueError("stage_h_case preparation requires stage_h_run_manifest_path.")
        return prepare_stage_h_case_sequences(
            config.stage_h_run_manifest_path,
            profile=config.profile or REAL_SORTIE_V1,
        )
    if normalized == "uab":
        normalized = "uab_workload_dataset"
    if normalized == "nasa":
        normalized = "nasa_csm"
    if normalized == "uab_workload_dataset":
        if not config.dataset_root:
            raise ValueError("UAB sequence preparation requires dataset_root.")
        return prepare_uab_sequences(
            config.dataset_root,
            profile=config.profile or WINDOW_V2,
            target_steps=config.target_steps,
            progress_callback=progress_callback,
        )
    if normalized == "nasa_csm":
        if not config.dataset_root:
            raise ValueError("NASA sequence preparation requires dataset_root.")
        return prepare_nasa_sequences(
            config.dataset_root,
            profile=config.profile or WINDOW_V2,
            target_steps=config.target_steps,
            progress_callback=progress_callback,
        )
    raise ValueError(f"unsupported Stage I deep sequence dataset: {config.dataset_id}")


def _record_sequence_preparation_progress(
    *,
    progress,
    event: str,
    **fields: object,
) -> None:
    LOGGER.info(
        "stage_i_sequence_preparation %s %s",
        event,
        " ".join(f"{key}={value}" for key, value in sorted(fields.items())),
    )
    progress.update(event, **fields)
