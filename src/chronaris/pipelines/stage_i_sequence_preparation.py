"""Preparation pipeline for Stage I deep-baseline sequence assets."""

from __future__ import annotations

import json
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
    summary: dict[str, object]


def run_stage_i_sequence_preparation(
    config: StageISequencePreparationConfig,
) -> StageISequencePreparationRunResult:
    artifact_root = Path(config.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = _prepare_payload(config)
    manifest_path = artifact_root / "task_manifest.jsonl"
    bundle_path = artifact_root / "sequence_bundle.npz"
    schema_path = artifact_root / "sequence_schema.json"
    summary_path = artifact_root / "dataset_summary.json"
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
    return StageISequencePreparationRunResult(
        dataset_id=config.dataset_id,
        artifact_root=str(artifact_root),
        manifest_path=str(manifest_path),
        bundle_path=str(bundle_path),
        schema_path=str(schema_path),
        summary_path=str(summary_path),
        summary=payload.summary.to_dict(),
    )


def _prepare_payload(
    config: StageISequencePreparationConfig,
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
        )
    if normalized == "nasa_csm":
        if not config.dataset_root:
            raise ValueError("NASA sequence preparation requires dataset_root.")
        return prepare_nasa_sequences(
            config.dataset_root,
            profile=config.profile or WINDOW_V2,
            target_steps=config.target_steps,
        )
    raise ValueError(f"unsupported Stage I deep sequence dataset: {config.dataset_id}")
