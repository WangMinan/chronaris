"""Fixed application contexts spanning G1 development and clean G2 test sorties."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    APPLICATION_CONTEXT_STARTS_S,
)
from chronaris.representation import (
    DualStreamObservationBatch,
    FoldLineage,
    ObservationSchema,
    collate_observation_samples,
    load_simulation_observed_context,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class SimulationLockedContextData:
    batch: DualStreamObservationBatch
    schema: ObservationSchema
    fold: FoldLineage
    role_sample_ids: Mapping[str, tuple[str, ...]]
    sample_manifest_rows: tuple[Mapping[str, object], ...]


def load_simulation_locked_context_data(
    simulation_root: str | Path,
) -> SimulationLockedContextData:
    """Open G2 observations only after the caller has verified all checkpoints."""

    root = Path(simulation_root)
    role_paths = {
        "train": _paths(root / "train", "train_profile_", "g1_state_space__"),
        "validation": _paths(
            root / "validation", "validation_profile_", "g1_state_space__"
        ),
        "held_out": _paths(
            root / "locked_test", "locked_test_profile_", "g2_event_spline__"
        ),
    }
    expected = {"train": 96, "validation": 24, "held_out": 48}
    actual = {role: len(paths) for role, paths in role_paths.items()}
    if actual != expected:
        raise ValueError(f"locked application context trajectory counts changed: {actual}")
    samples = []
    rows = []
    role_ids = {role: [] for role in role_paths}
    for role, paths in role_paths.items():
        for path in paths:
            for context_start_s in APPLICATION_CONTEXT_STARTS_S:
                sample = load_simulation_observed_context(
                    path,
                    context_start_s=context_start_s,
                )
                samples.append(sample)
                role_ids[role].append(sample.sample_id)
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "role": role,
                        "split_id": (
                            "locked_test" if role == "held_out" else role
                        ),
                        "profile_id": next(
                            part for part in path.parts if "_profile_" in part
                        ),
                        "trajectory_id": path.parents[1].name,
                        "scenario_id": path.parent.name,
                        "context_start_s": context_start_s,
                        "context_end_s": context_start_s + 30.0,
                        "observed_path": str(path),
                        "observed_sha256": sha256_file(path),
                        "oracle_opened_for_representation": False,
                    }
                )
    role_sample_ids = {role: tuple(values) for role, values in role_ids.items()}
    fold = FoldLineage(
        fold_id="simulation_g1_to_g2_clean_locked",
        train_sample_ids=role_sample_ids["train"],
        validation_sample_ids=role_sample_ids["validation"],
        held_out_sample_ids=role_sample_ids["held_out"],
    )
    return SimulationLockedContextData(
        batch=collate_observation_samples(samples),
        schema=samples[0].schema,
        fold=fold,
        role_sample_ids=role_sample_ids,
        sample_manifest_rows=tuple(rows),
    )


def _paths(root: Path, profile_prefix: str, family_prefix: str) -> tuple[Path, ...]:
    return tuple(
        sorted(
            root.glob(
                f"{profile_prefix}*/{family_prefix}*/clean_asynchronous/"
                "raw_dual_stream.npz"
            )
        )
    )
