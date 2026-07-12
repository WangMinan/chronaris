"""Merge device-partitioned structure-screen rows without moving checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.modeling.training import chronaris_v2_structure_candidates


@dataclass(frozen=True, slots=True)
class ChronarisV2StructureMergeConfig:
    run_id: str = "2026-07-12_chronaris-v2-structure-screen-seed17-combined"
    source_run_ids: tuple[str, ...] = (
        "2026-07-12_chronaris-v2-structure-screen-seed17-r4",
        "2026-07-12_chronaris-v2-structure-screen-seed17-r5-cpu-recovery",
    )
    compact_output_root: str = "docs/artifacts/runs"


def merge_chronaris_v2_structure_screens(
    config: ChronarisV2StructureMergeConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2StructureMergeConfig()
    if len(resolved.source_run_ids) < 2:
        raise ValueError("structure merge requires at least two source runs")
    rows = []
    for source_run_id in resolved.source_run_ids:
        path = (
            Path(resolved.compact_output_root)
            / source_run_id
            / "candidate_training.csv"
        )
        source_rows = pd.read_csv(path).to_dict(orient="records")
        rows.extend({**row, "source_run_id": source_run_id} for row in source_rows)
    expected_ids = {
        candidate.candidate_id for candidate in chronaris_v2_structure_candidates()
    }
    by_id = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in by_id:
            if candidate_id == "structure_01_v1":
                if row["checkpoint_sha256"] != by_id[candidate_id]["checkpoint_sha256"]:
                    raise ValueError("device runs disagree on the immutable v1 reference")
                continue
            by_id[candidate_id] = row
            continue
        by_id[candidate_id] = row
    if set(by_id) != expected_ids:
        raise ValueError(
            f"combined structure candidates differ: {sorted(set(by_id) ^ expected_ids)}"
        )
    combined = [by_id[candidate.candidate_id] for candidate in chronaris_v2_structure_candidates()]
    if any(
        row["status"] not in {"immutable_reference", "completed", "resumed"}
        for row in combined
    ):
        raise ValueError("combined structure screen contains incomplete checkpoints")
    if any(
        bool(row["task_labels_opened"])
        or bool(row["simulation_oracle_opened"])
        or bool(row["locked_test_opened"])
        for row in combined
    ):
        raise ValueError("combined structure screen contains forbidden evidence")
    output_root = Path(resolved.compact_output_root) / resolved.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(combined).to_csv(output_root / "candidate_training.csv", index=False)
    _write_json(
        output_root / "merge_protocol.json",
        {
            "format": "chronaris.v2_structure_screen_merge.v1",
            "config": asdict(resolved),
            "candidate_source_map": {
                row["candidate_id"]: row["source_run_id"] for row in combined
            },
            "duplicate_candidate_policy": (
                "later source_run_ids entry supersedes earlier development recovery"
            ),
            "checkpoint_files_moved": False,
            "task_labels_opened": False,
            "simulation_oracle_opened": False,
            "locked_test_opened": False,
        },
    )
    _write_json(
        output_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_structure_screen_merge_evidence.v1",
            "run_id": resolved.run_id,
            "status": "completed",
            "candidate_count": len(combined),
            "source_run_ids": list(resolved.source_run_ids),
            "confirmed_metrics_changed": False,
        },
    )
    return output_root


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
