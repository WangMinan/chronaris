"""Observed-only G1 data split for equal-budget encoder candidate screening."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chronaris.representation import (
    FoldLineage,
    ObservationSchema,
    collate_observation_samples,
    load_simulation_observed_context,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class CandidateScreenData:
    batch: object
    fold: FoldLineage
    schema: ObservationSchema
    data_manifest_rows: tuple[Mapping[str, object], ...]
    sealed_locked_test_file_count: int


def load_candidate_screen_data(simulation_root: str | Path) -> CandidateScreenData:
    root = Path(simulation_root)
    train_paths = _g1_paths(root / "train", prefix="train_profile_")
    validation_paths = _g1_paths(root / "validation", prefix="validation_profile_")
    if len(train_paths) != 96 or len(validation_paths) != 24:
        raise ValueError(
            "formal candidate screen requires 96 G1 train and 24 G1 validation profiles"
        )
    ranking_paths = validation_paths[:-1]
    confirmation_paths = validation_paths[-1:]
    selected = (*train_paths, *ranking_paths, *confirmation_paths)
    samples = tuple(
        load_simulation_observed_context(path, context_start_s=0.0) for path in selected
    )
    roles = (
        *("train" for _ in train_paths),
        *("validation" for _ in ranking_paths),
        *("held_out" for _ in confirmation_paths),
    )
    fold = FoldLineage(
        fold_id="simulation_g1_seed17_candidate_screen",
        train_sample_ids=tuple(
            sample.sample_id for sample, role in zip(samples, roles, strict=True) if role == "train"
        ),
        validation_sample_ids=tuple(
            sample.sample_id
            for sample, role in zip(samples, roles, strict=True)
            if role == "validation"
        ),
        held_out_sample_ids=tuple(
            sample.sample_id
            for sample, role in zip(samples, roles, strict=True)
            if role == "held_out"
        ),
    )
    rows = tuple(
        {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "role": role,
            "profile_id": next(part for part in path.parts if "_profile_" in part),
            "trajectory_id": path.parents[1].name,
            "observed_path": str(path),
            "observed_sha256": sha256_file(path),
            "source_kind": "raw_dual_stream_only",
            "ground_truth_opened": False,
            "locked_test_member": False,
        }
        for sample, role, path in zip(samples, roles, selected, strict=True)
    )
    sealed_count = len(tuple((root / "locked_test").glob("**/raw_dual_stream.npz")))
    return CandidateScreenData(
        batch=collate_observation_samples(samples),
        fold=fold,
        schema=samples[0].schema,
        data_manifest_rows=rows,
        sealed_locked_test_file_count=sealed_count,
    )


def _g1_paths(root: Path, *, prefix: str) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            root.glob(
                f"{prefix}*/g1_state_space__*/clean_asynchronous/raw_dual_stream.npz"
            )
        )
    )
    if any("ground_truth" in path.name for path in paths):
        raise ValueError("candidate screen observed paths unexpectedly contain truth files")
    return paths
