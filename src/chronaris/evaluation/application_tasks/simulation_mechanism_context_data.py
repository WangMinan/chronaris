"""Observed-only G1 contexts for clock-offset and response-lag recovery probes."""

from __future__ import annotations

from pathlib import Path

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    APPLICATION_CONTEXT_STARTS_S,
    ApplicationConsumerSmokeData,
)
from chronaris.representation import (
    collate_observation_samples,
    load_simulation_observed_context,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def load_simulation_mechanism_context_data(
    simulation_root: str | Path,
    *,
    role: str,
    scenario_id: str,
) -> ApplicationConsumerSmokeData:
    if role not in {"train", "validation"}:
        raise ValueError("mechanism context role must be train or validation")
    expected = 96 if role == "train" else 24
    root = Path(simulation_root) / role
    paths = tuple(
        sorted(
            root.glob(
                f"{role}_profile_*/g1_state_space__*/{scenario_id}/"
                "raw_dual_stream.npz"
            )
        )
    )
    if len(paths) != expected:
        raise ValueError(
            f"mechanism {role}/{scenario_id} requires {expected} trajectories, "
            f"found {len(paths)}"
        )
    samples = []
    rows = []
    for path in paths:
        observed_hash = sha256_file(path)
        for context_start_s in APPLICATION_CONTEXT_STARTS_S:
            sample = load_simulation_observed_context(
                path,
                context_start_s=context_start_s,
            )
            samples.append(sample)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "profile_id": next(
                        part for part in path.parts if part.startswith(f"{role}_profile_")
                    ),
                    "trajectory_id": path.parents[1].name,
                    "role": role,
                    "scenario_id": scenario_id,
                    "context_start_s": context_start_s,
                    "context_end_s": context_start_s + 30.0,
                    "observed_path": str(path),
                    "observed_sha256": observed_hash,
                    "oracle_opened_for_representation": False,
                }
            )
    batch = collate_observation_samples(samples)
    return ApplicationConsumerSmokeData(
        batch=batch,
        schema=samples[0].schema,
        role_sample_ids={
            "train": batch.sample_ids if role == "train" else (),
            "validation": batch.sample_ids if role == "validation" else (),
            "held_out": (),
        },
        sample_manifest_rows=tuple(rows),
    )
