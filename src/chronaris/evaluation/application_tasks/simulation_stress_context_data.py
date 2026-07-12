"""Observed-only four-context slices for one frozen G2 stress scenario."""

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


def load_simulation_stress_context_data(
    stress_root: str | Path,
    *,
    scenario_id: str,
    split_id: str = "locked_test",
    profile_prefix: str = "locked_test_profile_",
) -> ApplicationConsumerSmokeData:
    root = Path(stress_root)
    paths = tuple(
        sorted(
            (root / split_id).glob(
                f"{profile_prefix}*/g2_event_spline__*/{scenario_id}/"
                "raw_dual_stream.npz"
            )
        )
    )
    if len(paths) != 48:
        raise ValueError(
            f"stress scenario {scenario_id} requires 48 G2 trajectories, found {len(paths)}"
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
                        part for part in path.parts if part.startswith(profile_prefix)
                    ),
                    "trajectory_id": path.parents[1].name,
                    "role": "held_out",
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
            "train": (),
            "validation": (),
            "held_out": batch.sample_ids,
        },
        sample_manifest_rows=tuple(rows),
    )
