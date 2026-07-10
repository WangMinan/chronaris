"""Observed-only G1 simulation split for the common pretraining loop smoke."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.representation import (
    FoldLineage,
    ObservationSchema,
    collate_observation_samples,
    load_simulation_observed_context,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class PretrainingSmokeData:
    batch: object
    fold: FoldLineage
    schema: ObservationSchema
    data_manifest_rows: tuple[Mapping[str, object], ...]
    observed_paths: tuple[str, ...]


def load_pretraining_smoke_data(
    simulation_root: str | Path,
) -> PretrainingSmokeData:
    root = Path(simulation_root)
    candidates = sorted(
        (root / "train").glob(
            "train_profile_*/g1_state_space__*/clean_asynchronous/raw_dual_stream.npz"
        )
    )
    selected = select_one_g1_observation_per_profile(candidates, sample_count=16)
    samples = tuple(
        load_simulation_observed_context(path, context_start_s=0.0)
        for path in selected
    )
    batch = collate_observation_samples(samples)
    train_ids = tuple(sample.sample_id for sample in samples[:8])
    validation_ids = tuple(sample.sample_id for sample in samples[8:12])
    held_out_ids = tuple(sample.sample_id for sample in samples[12:16])
    fold = FoldLineage(
        fold_id="simulation_g1_common_pretraining_smoke",
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        held_out_sample_ids=held_out_ids,
    )
    roles = {
        sample_id: role
        for role, values in (
            ("train", train_ids),
            ("validation", validation_ids),
            ("held_out", held_out_ids),
        )
        for sample_id in values
    }
    rows = tuple(
        {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "role": roles[sample.sample_id],
            "profile_id": path.parents[2].name,
            "trajectory_id": path.parents[1].name,
            "observed_path": str(path),
            "observed_sha256": sha256_file(path),
            "source_kind": "raw_dual_stream_only",
            "oracle_opened_for_pretraining": False,
        }
        for sample, path in zip(samples, selected, strict=True)
    )
    return PretrainingSmokeData(
        batch=batch,
        fold=fold,
        schema=samples[0].schema,
        data_manifest_rows=rows,
        observed_paths=tuple(str(path) for path in selected),
    )


def select_one_g1_observation_per_profile(
    paths: Sequence[Path],
    *,
    sample_count: int,
) -> tuple[Path, ...]:
    if sample_count <= 0:
        raise ValueError("pretraining smoke sample count must be positive")
    by_profile: dict[str, Path] = {}
    for raw_path in sorted(Path(path) for path in paths):
        normalized = raw_path.as_posix()
        if "/locked_test/" in normalized or "/validation/" in normalized:
            raise ValueError("pretraining smoke paths must come only from train split")
        if "g1_state_space__" not in normalized:
            continue
        profile_parts = [part for part in raw_path.parts if part.startswith("train_profile_")]
        if len(profile_parts) != 1:
            raise ValueError(f"cannot resolve profile from {raw_path}")
        by_profile.setdefault(profile_parts[0], raw_path)
    if len(by_profile) < sample_count:
        raise ValueError(
            f"pretraining smoke requires {sample_count} distinct profiles, "
            f"found {len(by_profile)}"
        )
    return tuple(by_profile[key] for key in sorted(by_profile)[:sample_count])
