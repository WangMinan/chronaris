"""Public latent and paired-observation generation entrypoints."""

from __future__ import annotations

import numpy as np

from .config import AviationScenarioConfig, ObservationScenarioConfig
from .contracts import LatentTrajectory, SimulatedDualStreamSortie
from .maneuvers import generate_maneuver_plan, render_maneuver_states
from .observation import render_observations
from .physiology import generate_physiology
from .profiles import PilotProfile
from .vehicle_g1 import generate_g1_vehicle
from .vehicle_g2 import generate_g2_vehicle
from .workload import generate_workload


def generate_latent_trajectory(
    config: AviationScenarioConfig,
    pilot: PilotProfile,
    latent_seed: int,
) -> LatentTrajectory:
    """Generate one reusable latent truth trajectory."""

    seed_sequence = np.random.SeedSequence(latent_seed)
    event_rng, vehicle_rng, workload_rng, physiology_rng = (
        np.random.default_rng(child) for child in seed_sequence.spawn(4)
    )
    true_time_s = np.arange(
        0.0,
        config.duration_s,
        config.dt_s,
        dtype=np.float64,
    )
    events = generate_maneuver_plan(config, event_rng)
    maneuver_state, maneuver_type = render_maneuver_states(true_time_s, events)
    if config.generator_family == "g1_state_space":
        controls, vehicle_state, physical_residual = generate_g1_vehicle(
            true_time_s,
            events,
            dt_s=config.dt_s,
            process_noise_scale=config.process_noise_scale,
            rng=vehicle_rng,
        )
    else:
        controls, vehicle_state, physical_residual = generate_g2_vehicle(
            true_time_s,
            events,
            dt_s=config.dt_s,
            process_noise_scale=config.process_noise_scale,
            rng=vehicle_rng,
        )
    workload = generate_workload(
        family=config.generator_family,
        true_time_s=true_time_s,
        maneuver_state=maneuver_state,
        controls=controls,
        vehicle_state=vehicle_state,
        profile=pilot,
        dt_s=config.dt_s,
        rng=workload_rng,
    )
    physiology_state, field_lags = generate_physiology(
        family=config.generator_family,
        true_time_s=true_time_s,
        workload=workload,
        profile=pilot,
        dt_s=config.dt_s,
        rng=physiology_rng,
    )
    trajectory_id = (
        f"{config.generator_family}__{pilot.profile_id}__"
        f"trajectory_{config.trajectory_index:03d}__seed_{latent_seed}"
    )
    return LatentTrajectory(
        trajectory_id=trajectory_id,
        config=config,
        profile_id=pilot.profile_id,
        latent_seed=latent_seed,
        true_time_s=true_time_s,
        maneuver_state=maneuver_state,
        maneuver_type=maneuver_type,
        controls=controls,
        vehicle_state=vehicle_state,
        workload=workload,
        physiology_state=physiology_state,
        physiology_field_lag_s=field_lags,
        physical_residual=physical_residual,
        events=events,
        profile_parameters={
            key: float(value)
            for key, value in pilot.to_dict().items()
            if key != "profile_id"
        },
    )


def generate_sortie(
    config: AviationScenarioConfig,
    pilot: PilotProfile,
    latent_seed: int,
    observation_seed: int,
    observation: ObservationScenarioConfig | None = None,
) -> SimulatedDualStreamSortie:
    """Generate one latent trajectory and one method-independent observation version."""

    latent = generate_latent_trajectory(config, pilot, latent_seed)
    observation_config = observation or ObservationScenarioConfig(
        scenario_id="clean_asynchronous"
    )
    return render_sortie(
        latent,
        observation_config,
        observation_seed=observation_seed,
    )


def render_sortie(
    latent: LatentTrajectory,
    observation: ObservationScenarioConfig,
    *,
    observation_seed: int,
) -> SimulatedDualStreamSortie:
    """Render one additional observation version without changing latent truth."""

    vehicle, physiology, vehicle_trace, physiology_trace, realized_lag = render_observations(
        latent,
        observation,
        observation_seed=observation_seed,
    )
    return SimulatedDualStreamSortie(
        sample_id=f"{latent.trajectory_id}__{observation.scenario_id}",
        latent=latent,
        observation_config=observation,
        observation_seed=observation_seed,
        vehicle=vehicle,
        physiology=physiology,
        vehicle_trace=vehicle_trace,
        physiology_trace=physiology_trace,
        realized_physiology_lag_s=realized_lag,
    )
