"""Observed-only G1 train/validation data for multi-seed locked retraining."""

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
class SimulationLockedPretrainingData:
    batch: object
    fold: FoldLineage
    schema: ObservationSchema
    data_manifest_rows: tuple[Mapping[str, object], ...]
    sealed_locked_test_file_count: int


def load_simulation_locked_pretraining_data(
    simulation_root: str | Path,
) -> SimulationLockedPretrainingData:
    """Load all G1 clean train/validation observations without opening G2."""

    root = Path(simulation_root)
    train_paths = _clean_paths(root / "train", "train_profile_")
    validation_paths = _clean_paths(root / "validation", "validation_profile_")
    if len(train_paths) != 96 or len(validation_paths) != 24:
        raise ValueError("locked retraining requires 96 G1 train and 24 validation sorties")
    ranking_validation_paths = validation_paths[:-1]
    reserved_confirmation_paths = validation_paths[-1:]
    paths = train_paths + ranking_validation_paths + reserved_confirmation_paths
    roles = (
        ("train",) * len(train_paths)
        + ("validation",) * len(ranking_validation_paths)
        + ("held_out",) * len(reserved_confirmation_paths)
    )
    samples = tuple(
        load_simulation_observed_context(path, context_start_s=0.0)
        for path in paths
    )
    fold = FoldLineage(
        fold_id="simulation_g1_locked_retraining",
        train_sample_ids=tuple(
            sample.sample_id
            for sample, role in zip(samples, roles, strict=True)
            if role == "train"
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
        for sample, role, path in zip(samples, roles, paths, strict=True)
    )
    return SimulationLockedPretrainingData(
        batch=collate_observation_samples(samples),
        fold=fold,
        schema=samples[0].schema,
        data_manifest_rows=rows,
        sealed_locked_test_file_count=len(
            tuple((root / "locked_test").glob("**/raw_dual_stream.npz"))
        ),
    )


def _clean_paths(root: Path, profile_prefix: str) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            root.glob(
                f"{profile_prefix}*/g1_state_space__*/clean_asynchronous/"
                "raw_dual_stream.npz"
            )
        )
    )
    if any("locked_test" in path.parts or path.name != "raw_dual_stream.npz" for path in paths):
        raise ValueError("locked retraining path crossed the observed G1 contract")
    return paths
