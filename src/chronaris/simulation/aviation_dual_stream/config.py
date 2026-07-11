"""Frozen configuration for latent and observation generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


GeneratorFamily = Literal["g1_state_space", "g2_event_spline"]


@dataclass(frozen=True, slots=True)
class AviationScenarioConfig:
    """Latent trajectory configuration shared by all observation variants."""

    generator_family: GeneratorFamily
    duration_s: float = 180.0
    truth_rate_hz: float = 20.0
    trajectory_index: int = 0
    process_noise_scale: float = 0.015
    minimum_complete_events: int = 2
    generator_version: str = "aviation_dual_stream.v1"

    def __post_init__(self) -> None:
        if self.duration_s < 30.0:
            raise ValueError("duration_s must be at least 30 seconds")
        if self.truth_rate_hz < 10.0:
            raise ValueError("truth_rate_hz must be at least 10 Hz")
        if self.minimum_complete_events < 1:
            raise ValueError("minimum_complete_events must be positive")
        if self.process_noise_scale < 0:
            raise ValueError("process_noise_scale cannot be negative")

    @property
    def dt_s(self) -> float:
        return 1.0 / self.truth_rate_hz

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationScenarioConfig:
    """Sensor sampling, clock, missingness and noise configuration."""

    scenario_id: str
    vehicle_rate_hz: float = 10.0
    physiology_rate_hz: float = 2.0
    vehicle_jitter_std_ms: float = 2.0
    physiology_jitter_std_ms: float = 5.0
    vehicle_clock_offset_s: float = 0.0
    physiology_clock_offset_s: float = 0.0
    vehicle_clock_drift_ppm: float = 0.0
    physiology_clock_drift_ppm: float = 0.0
    vehicle_random_missing_rate: float = 0.0
    physiology_random_missing_rate: float = 0.0
    vehicle_block_gap_s: float = 0.0
    physiology_block_gap_s: float = 0.0
    additional_physiology_lag_s: float = 0.0
    observation_snr_db: float = 30.0

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        if not 5.0 <= self.vehicle_rate_hz <= 20.0:
            raise ValueError("vehicle_rate_hz must be in [5, 20]")
        if not 0.5 <= self.physiology_rate_hz <= 5.0:
            raise ValueError("physiology_rate_hz must be in [0.5, 5]")
        for value in (
            self.vehicle_random_missing_rate,
            self.physiology_random_missing_rate,
        ):
            if not 0.0 <= value <= 0.8:
                raise ValueError("random missing rate must be in [0, 0.8]")
        if min(
            self.vehicle_jitter_std_ms,
            self.physiology_jitter_std_ms,
            self.vehicle_block_gap_s,
            self.physiology_block_gap_s,
            self.additional_physiology_lag_s,
        ) < 0:
            raise ValueError("jitter, gap and additional lag cannot be negative")
        if self.observation_snr_db <= 0:
            raise ValueError("observation_snr_db must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_observation_scenarios() -> tuple[ObservationScenarioConfig, ...]:
    """Return the six locked paired observation scenarios."""

    return (
        ObservationScenarioConfig(scenario_id="clean_asynchronous"),
        ObservationScenarioConfig(
            scenario_id="sampling_jitter",
            vehicle_jitter_std_ms=20.0,
            physiology_jitter_std_ms=50.0,
            observation_snr_db=20.0,
        ),
        ObservationScenarioConfig(
            scenario_id="clock_offset_and_drift",
            vehicle_clock_offset_s=-0.25,
            physiology_clock_offset_s=1.0,
            vehicle_clock_drift_ppm=-50.0,
            physiology_clock_drift_ppm=100.0,
            observation_snr_db=20.0,
        ),
        ObservationScenarioConfig(
            scenario_id="random_missing",
            vehicle_random_missing_rate=0.10,
            physiology_random_missing_rate=0.30,
            observation_snr_db=20.0,
        ),
        ObservationScenarioConfig(
            scenario_id="block_missing_and_long_lag",
            vehicle_block_gap_s=5.0,
            physiology_block_gap_s=15.0,
            additional_physiology_lag_s=15.0,
            observation_snr_db=15.0,
        ),
        ObservationScenarioConfig(
            scenario_id="mixed_severe",
            vehicle_jitter_std_ms=100.0,
            physiology_jitter_std_ms=100.0,
            vehicle_clock_offset_s=-3.0,
            physiology_clock_offset_s=3.0,
            vehicle_clock_drift_ppm=-250.0,
            physiology_clock_drift_ppm=250.0,
            vehicle_random_missing_rate=0.30,
            physiology_random_missing_rate=0.30,
            vehicle_block_gap_s=15.0,
            physiology_block_gap_s=15.0,
            additional_physiology_lag_s=30.0,
            observation_snr_db=10.0,
        ),
    )


def locked_stress_observation_scenarios() -> tuple[ObservationScenarioConfig, ...]:
    """Return the frozen single-factor levels plus the mixed-severe scenario."""

    scenarios = []
    for value in (0.0, 20.0, 50.0, 100.0):
        scenarios.append(
            _stress_base(
                f"timestamp_jitter_{int(value):03d}ms",
                vehicle_jitter_std_ms=value,
                physiology_jitter_std_ms=value,
            )
        )
    for value in (0.0, 0.25, 1.0, 3.0):
        signs = (1.0,) if value == 0 else (-1.0, 1.0)
        scenarios.extend(
            _stress_base(
                f"clock_offset_{sign * value:+.2f}s",
                physiology_clock_offset_s=sign * value,
            )
            for sign in signs
        )
    for value in (0.0, 50.0, 100.0, 250.0):
        signs = (1.0,) if value == 0 else (-1.0, 1.0)
        scenarios.extend(
            _stress_base(
                f"clock_drift_{int(sign * value):+04d}ppm",
                physiology_clock_drift_ppm=sign * value,
            )
            for sign in signs
        )
    for value in (0.0, 0.10, 0.30, 0.50):
        scenarios.append(
            _stress_base(
                f"random_missing_{int(value * 100):02d}pct",
                vehicle_random_missing_rate=value,
                physiology_random_missing_rate=value,
            )
        )
    for value in (0.0, 5.0, 15.0, 30.0):
        scenarios.append(
            _stress_base(
                f"contiguous_gap_{int(value):02d}s",
                vehicle_block_gap_s=value,
                physiology_block_gap_s=value,
            )
        )
    for value in (2.0, 5.0, 15.0, 30.0):
        scenarios.append(
            _stress_base(
                f"physiology_lag_{int(value):02d}s",
                additional_physiology_lag_s=value,
            )
        )
    for value in (30.0, 20.0, 10.0, 5.0):
        scenarios.append(
            _stress_base(
                f"observation_snr_{int(value):02d}db",
                observation_snr_db=value,
            )
        )
    scenarios.append(
        _stress_base(
            "mixed_severe",
            vehicle_jitter_std_ms=100.0,
            physiology_jitter_std_ms=100.0,
            vehicle_clock_offset_s=-3.0,
            physiology_clock_offset_s=3.0,
            vehicle_clock_drift_ppm=-250.0,
            physiology_clock_drift_ppm=250.0,
            vehicle_random_missing_rate=0.30,
            physiology_random_missing_rate=0.30,
            vehicle_block_gap_s=15.0,
            physiology_block_gap_s=15.0,
            additional_physiology_lag_s=30.0,
            observation_snr_db=10.0,
        )
    )
    return tuple(scenarios)


def _stress_base(scenario_id: str, **overrides) -> ObservationScenarioConfig:
    values = {
        "scenario_id": scenario_id,
        "vehicle_jitter_std_ms": 0.0,
        "physiology_jitter_std_ms": 0.0,
        "observation_snr_db": 30.0,
        **overrides,
    }
    return ObservationScenarioConfig(**values)
