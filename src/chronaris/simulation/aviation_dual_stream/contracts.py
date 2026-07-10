"""Array contracts for latent truth, observed streams and oracle traces."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .config import AviationScenarioConfig, ObservationScenarioConfig


MANEUVER_STATE_NAMES = ("steady", "entry", "sustained", "exit", "recovery")
MANEUVER_TYPE_NAMES = (
    "pitch_dominant",
    "roll_dominant",
    "yaw_turn_dominant",
    "acceleration_dominant",
    "compound",
)
VEHICLE_FEATURE_NAMES = (
    "speed_mps",
    "altitude_m",
    "vertical_speed_mps",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "roll_rate_rps",
    "pitch_rate_rps",
    "yaw_rate_rps",
    "longitudinal_acc_mps2",
    "lateral_acc_mps2",
    "normal_load_g",
)
CONTROL_FEATURE_NAMES = (
    "stick_longitudinal",
    "stick_lateral",
    "pedal",
    "throttle",
)
PHYSIOLOGY_FEATURE_NAMES = (
    "eeg_low_relative",
    "eeg_high_relative",
    "eeg_complexity",
    "spo2_percent",
    "heart_rate_bpm",
    "physiology_variability",
    "individual_baseline",
)
PHYSICAL_RESIDUAL_NAMES = (
    "altitude_vertical_speed",
    "speed_longitudinal_acc",
    "roll_roll_rate",
    "pitch_pitch_rate",
    "yaw_yaw_rate",
    "normal_load_vertical_acc",
)


@dataclass(frozen=True, slots=True)
class ManeuverEvent:
    event_id: int
    maneuver_type: int
    entry_start_s: float
    sustained_start_s: float
    exit_start_s: float
    recovery_start_s: float
    event_end_s: float
    amplitude: float
    direction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LatentTrajectory:
    trajectory_id: str
    config: AviationScenarioConfig
    profile_id: str
    latent_seed: int
    true_time_s: NDArray[np.float64]
    maneuver_state: NDArray[np.int8]
    maneuver_type: NDArray[np.int8]
    controls: NDArray[np.float64]
    vehicle_state: NDArray[np.float64]
    workload: NDArray[np.float64]
    physiology_state: NDArray[np.float64]
    physiology_field_lag_s: NDArray[np.float64]
    physical_residual: NDArray[np.float64]
    events: tuple[ManeuverEvent, ...]
    profile_parameters: Mapping[str, float]

    @property
    def latent_hash(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.true_time_s,
            self.maneuver_state,
            self.maneuver_type,
            self.controls,
            self.vehicle_state,
            self.workload,
            self.physiology_state,
            self.physiology_field_lag_s,
            self.physical_residual,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ObservedStream:
    stream_kind: str
    feature_names: tuple[str, ...]
    true_sample_time_s: NDArray[np.float64]
    observed_time_s: NDArray[np.float64]
    values: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ObservationTrace:
    stream_kind: str
    candidate_true_time_s: NDArray[np.float64]
    retained_mask: NDArray[np.bool_]
    retained_jitter_s: NDArray[np.float64]
    clock_offset_s: float
    clock_drift_ppm: float
    random_missing_rate_config: float
    block_gap_start_s: float | None
    block_gap_duration_s: float


@dataclass(frozen=True, slots=True)
class SimulatedDualStreamSortie:
    sample_id: str
    latent: LatentTrajectory
    observation_config: ObservationScenarioConfig
    observation_seed: int
    vehicle: ObservedStream
    physiology: ObservedStream
    vehicle_trace: ObservationTrace
    physiology_trace: ObservationTrace
    realized_physiology_lag_s: NDArray[np.float64]

    @property
    def latent_hash(self) -> str:
        return self.latent.latent_hash
