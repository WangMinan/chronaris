"""Fixed-grid application contexts and guarded simulation targets for G4 smoke."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from chronaris.evaluation.application_tasks.pretraining_smoke_data import (
    load_pretraining_smoke_data,
)
from chronaris.representation import (
    DualStreamObservationBatch,
    ObservationSchema,
    collate_observation_samples,
    load_simulation_observed_context,
)
from chronaris.simulation.aviation_dual_stream.contracts import MANEUVER_STATE_NAMES
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


APPLICATION_CONTEXT_STARTS_S = (30.0, 60.0, 90.0, 120.0)


@dataclass(frozen=True, slots=True)
class ApplicationConsumerSmokeData:
    batch: DualStreamObservationBatch
    schema: ObservationSchema
    role_sample_ids: Mapping[str, tuple[str, ...]]
    sample_manifest_rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class ApplicationConsumerSmokeTargets:
    sample_ids: tuple[str, ...]
    roles: tuple[str, ...]
    future_workload_mean: torch.Tensor
    workload_class: torch.Tensor
    maneuver_state: torch.Tensor
    boundary_mask: torch.Tensor
    manifest: Mapping[str, object]


def load_application_consumer_smoke_data(
    simulation_root: str | Path,
) -> ApplicationConsumerSmokeData:
    base = load_pretraining_smoke_data(simulation_root)
    role_by_profile = {
        row["profile_id"]: row["role"]
        for row in base.data_manifest_rows
    }
    samples = []
    rows = []
    for item in base.data_manifest_rows:
        observed_path = Path(str(item["observed_path"]))
        profile_id = str(item["profile_id"])
        for context_start_s in APPLICATION_CONTEXT_STARTS_S:
            sample = load_simulation_observed_context(
                observed_path,
                context_start_s=context_start_s,
            )
            samples.append(sample)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "profile_id": profile_id,
                    "trajectory_id": item["trajectory_id"],
                    "role": role_by_profile[profile_id],
                    "context_start_s": context_start_s,
                    "context_end_s": context_start_s + 30.0,
                    "observed_path": str(observed_path),
                    "observed_sha256": item["observed_sha256"],
                    "oracle_opened_for_representation": False,
                }
            )
    role_sample_ids = {
        role: tuple(
            row["sample_id"] for row in rows if row["role"] == role
        )
        for role in ("train", "validation", "held_out")
    }
    return ApplicationConsumerSmokeData(
        batch=collate_observation_samples(samples),
        schema=samples[0].schema,
        role_sample_ids=role_sample_ids,
        sample_manifest_rows=tuple(rows),
    )


def build_guarded_application_consumer_targets(
    data: ApplicationConsumerSmokeData,
    *,
    completed_pretraining_checkpoints: Sequence[str | Path],
    smoke_only: bool = True,
    workload_thresholds: tuple[float, float] | None = None,
) -> ApplicationConsumerSmokeTargets:
    _require_five_completed_checkpoints(completed_pretraining_checkpoints)
    role_by_sample = {
        sample_id: role
        for role, sample_ids in data.role_sample_ids.items()
        for sample_id in sample_ids
    }
    workload_values = []
    state_rows = []
    oracle_rows = []
    for item in data.sample_manifest_rows:
        oracle_path = Path(str(item["observed_path"])).with_name("ground_truth.npz")
        with np.load(oracle_path, allow_pickle=False) as archive:
            allowed = {"true_time_s", "workload", "maneuver_state"}
            if not allowed.issubset(archive.files):
                raise ValueError("application oracle lacks allowed target fields")
            times = np.asarray(archive["true_time_s"], dtype=np.float64)
            workload = np.asarray(archive["workload"], dtype=np.float64)
            states = np.asarray(archive["maneuver_state"], dtype=np.int64)
        context_start = float(item["context_start_s"])
        context_end = float(item["context_end_s"])
        future = (times >= context_end) & (times < context_end + 5.0)
        if not future.any():
            raise ValueError("application target has no future workload interval")
        query_absolute = (
            data.batch.query_timestamps_s[0].detach().cpu().numpy()
            + context_start
        )
        source_indices = np.searchsorted(times, query_absolute, side="right") - 1
        source_indices = np.clip(source_indices, 0, len(times) - 1)
        state_row = states[source_indices]
        if not set(np.unique(state_row)).issubset(
            set(range(len(MANEUVER_STATE_NAMES)))
        ):
            raise ValueError("application target contains unknown maneuver state")
        workload_values.append(float(workload[future].mean()))
        state_rows.append(state_row.astype(np.int64))
        oracle_rows.append(
            {
                "sample_id": item["sample_id"],
                "oracle_path": str(oracle_path),
                "oracle_sha256": sha256_file(oracle_path),
                "allowed_fields_opened": [
                    "true_time_s",
                    "workload",
                    "maneuver_state",
                ],
            }
        )
    workload_array = np.asarray(workload_values, dtype=np.float32)
    roles = tuple(role_by_sample[sample_id] for sample_id in data.batch.sample_ids)
    train_mask = np.asarray([role == "train" for role in roles])
    if workload_thresholds is None:
        if not train_mask.any():
            raise ValueError("application targets require train data or frozen thresholds")
        lower, upper = np.quantile(workload_array[train_mask], (1 / 3, 2 / 3))
        threshold_source = "train_only_quantiles"
    else:
        lower, upper = map(float, workload_thresholds)
        if not lower < upper:
            raise ValueError("frozen workload thresholds must be increasing")
        threshold_source = "frozen_clean_train_thresholds"
    workload_class = np.where(
        workload_array <= lower,
        0,
        np.where(workload_array <= upper, 1, 2),
    ).astype(np.int64)
    maneuver_state = np.stack(state_rows)
    boundaries = np.zeros_like(maneuver_state, dtype=bool)
    boundaries[:, 1:] = maneuver_state[:, 1:] != maneuver_state[:, :-1]
    manifest = {
        "format": "chronaris.application_consumer_smoke_targets.v1",
        "sample_count": len(data.batch.sample_ids),
        "context_start_grid_s": list(APPLICATION_CONTEXT_STARTS_S),
        "future_workload_window_relative_s": [0.0, 5.0],
        "workload_thresholds_train_only": [float(lower), float(upper)],
        "workload_threshold_source": threshold_source,
        "maneuver_state_names": list(MANEUVER_STATE_NAMES),
        "query_point_count": int(maneuver_state.shape[1]),
        "oracle_opened_after_checkpoint_count": 5,
        "allowed_oracle_fields": [
            "true_time_s",
            "workload",
            "maneuver_state",
        ],
        "oracle_files": oracle_rows,
        "smoke_only": bool(smoke_only),
    }
    return ApplicationConsumerSmokeTargets(
        sample_ids=data.batch.sample_ids,
        roles=roles,
        future_workload_mean=torch.from_numpy(workload_array),
        workload_class=torch.from_numpy(workload_class),
        maneuver_state=torch.from_numpy(maneuver_state),
        boundary_mask=torch.from_numpy(boundaries),
        manifest=manifest,
    )


def _require_five_completed_checkpoints(paths) -> None:
    resolved = tuple(Path(path) for path in paths)
    if len(resolved) != 5:
        raise ValueError("application targets require five completed checkpoints")
    methods = set()
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != "chronaris.common_pretraining_checkpoint.v1"
            or payload.get("training_status") != "completed"
            or bool(payload.get("label_used_for_encoder_training"))
        ):
            raise ValueError("application target checkpoint guard failed")
        methods.add(str(payload["method_name"]))
    if len(methods) != 5:
        raise ValueError("application target checkpoint methods are incomplete")
