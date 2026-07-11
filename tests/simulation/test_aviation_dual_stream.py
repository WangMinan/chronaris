"""Core method independence, determinism and oracle tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from chronaris.simulation.aviation_dual_stream import (
    AviationScenarioConfig,
    ObservationScenarioConfig,
    SimulationBenchmarkConfig,
    canonical_observation_scenarios,
    generate_benchmark,
    generate_latent_trajectory,
    generate_sortie,
    locked_stress_observation_scenarios,
    sample_pilot_profile,
    smoke_observation_scenarios,
    smoke_split_specs,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    write_deterministic_npz,
)
from chronaris.simulation.aviation_dual_stream.observation import render_observations
from chronaris.simulation.aviation_dual_stream.storage import (
    load_model_inputs,
    store_scenario,
)
from chronaris.simulation.aviation_dual_stream.validation import (
    split_identity_audit,
    validate_paired_observations,
    validate_sortie,
)


def _profile(index: int = 0):
    return sample_pilot_profile(split_id="smoke", profile_index=index, seed=100 + index)


def _config(family: str, index: int = 0):
    return AviationScenarioConfig(
        generator_family=family,
        duration_s=60.0,
        trajectory_index=index,
    )


def test_generator_api_has_no_method_argument() -> None:
    parameters = set(inspect.signature(generate_sortie).parameters)
    forbidden = {"method", "methods", "method_name", "model_name", "checkpoint"}
    assert not parameters & forbidden
    source_root = Path("src/chronaris/simulation/aviation_dual_stream")
    source_text = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in source_root.glob("*.py")
    )
    assert "contiformer" not in source_text
    assert "candidate_id" not in source_text


def test_generator_is_seed_reproducible() -> None:
    first = generate_latent_trajectory(_config("g1_state_space"), _profile(), 17)
    second = generate_latent_trajectory(_config("g1_state_space"), _profile(), 17)
    assert first.latent_hash == second.latent_hash
    assert np.array_equal(first.vehicle_state, second.vehicle_state)
    assert np.array_equal(first.physiology_state, second.physiology_state)


def test_latent_trajectory_unchanged_across_observation_scenarios() -> None:
    latent = generate_latent_trajectory(_config("g1_state_space"), _profile(), 17)
    sorties = []
    for scenario_index, observation in enumerate(canonical_observation_scenarios()[:3]):
        vehicle, physiology, vehicle_trace, physiology_trace, realized_lag = render_observations(
            latent,
            observation,
            observation_seed=200 + scenario_index,
        )
        from chronaris.simulation.aviation_dual_stream.contracts import SimulatedDualStreamSortie

        sorties.append(
            SimulatedDualStreamSortie(
                sample_id=f"{latent.trajectory_id}__{observation.scenario_id}",
                latent=latent,
                observation_config=observation,
                observation_seed=200 + scenario_index,
                vehicle=vehicle,
                physiology=physiology,
                vehicle_trace=vehicle_trace,
                physiology_trace=physiology_trace,
                realized_physiology_lag_s=realized_lag,
            )
        )
    audit = validate_paired_observations(sorties)
    assert audit["latent_hash_shared"]
    assert audit["trajectory_id_shared"]
    assert audit["scenario_ids_unique"]


def test_locked_stress_grid_has_frozen_single_factor_levels() -> None:
    scenarios = locked_stress_observation_scenarios()
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}

    assert len(scenarios) == 35
    assert len(by_id) == 35
    assert {
        scenario.physiology_jitter_std_ms
        for scenario in scenarios
        if scenario.scenario_id.startswith("timestamp_jitter_")
    } == {0.0, 20.0, 50.0, 100.0}
    assert {
        abs(scenario.physiology_clock_offset_s)
        for scenario in scenarios
        if scenario.scenario_id.startswith("clock_offset_")
    } == {0.0, 0.25, 1.0, 3.0}
    assert {
        abs(scenario.physiology_clock_drift_ppm)
        for scenario in scenarios
        if scenario.scenario_id.startswith("clock_drift_")
    } == {0.0, 50.0, 100.0, 250.0}
    assert by_id["mixed_severe"].additional_physiology_lag_s == 30.0
    assert by_id["mixed_severe"].observation_snr_db == 10.0


def test_g1_g2_use_distinct_generation_paths() -> None:
    g1 = generate_latent_trajectory(_config("g1_state_space"), _profile(), 17)
    g2 = generate_latent_trajectory(_config("g2_event_spline"), _profile(), 17)
    assert g1.latent_hash != g2.latent_hash
    assert not np.array_equal(g1.vehicle_state, g2.vehicle_state)


def test_ground_truth_clock_mapping_is_invertible() -> None:
    sortie = generate_sortie(
        _config("g2_event_spline"),
        _profile(),
        17,
        29,
        ObservationScenarioConfig(
            scenario_id="clock-test",
            vehicle_clock_offset_s=-1.0,
            physiology_clock_offset_s=2.0,
            vehicle_clock_drift_ppm=-100.0,
            physiology_clock_drift_ppm=250.0,
            vehicle_jitter_std_ms=50.0,
            physiology_jitter_std_ms=100.0,
        ),
    )
    audit = validate_sortie(sortie)
    assert audit["vehicle_clock_mapping_max_error_s"] < 1e-10
    assert audit["physiology_clock_mapping_max_error_s"] < 1e-10


def test_maneuver_state_boundary_and_numeric_contracts() -> None:
    for family in ("g1_state_space", "g2_event_spline"):
        sortie = generate_sortie(_config(family), _profile(), 17, 29)
        audit = validate_sortie(sortie)
        assert audit["event_count"] >= 2
        assert audit["all_states_present"]
        assert audit["vehicle_values_finite"]
        assert audit["physiology_values_finite"]
        assert audit["physical_residual_abs_median"] < 0.10
        if family == "g2_event_spline":
            assert audit["physical_residual_abs_q95"] < 0.35


def test_missingness_matches_configuration() -> None:
    sortie = generate_sortie(
        _config("g1_state_space"),
        _profile(),
        17,
        29,
        ObservationScenarioConfig(
            scenario_id="missing-test",
            vehicle_random_missing_rate=0.30,
            physiology_random_missing_rate=0.30,
        ),
    )
    audit = validate_sortie(sortie)
    assert abs(audit["vehicle_missing_ratio"] - 0.30) < 0.05
    assert abs(audit["physiology_missing_ratio"] - 0.30) < 0.08


def test_deterministic_npz_and_model_loader_exclude_oracle(tmp_path) -> None:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    arrays = {"b": np.arange(4), "a": np.eye(2)}
    assert write_deterministic_npz(first_path, arrays) == write_deterministic_npz(
        second_path, arrays
    )
    sortie = generate_sortie(_config("g1_state_space"), _profile(), 17, 29)
    stored = store_scenario(sortie, output_root=tmp_path / "bundle", split_id="smoke")
    model_inputs = load_model_inputs(stored.raw_dual_stream_path)
    assert "workload" not in model_inputs
    assert "maneuver_state" not in model_inputs
    assert set(model_inputs) == {
        "vehicle_observed_time_s",
        "vehicle_values",
        "vehicle_feature_names",
        "physiology_observed_time_s",
        "physiology_values",
        "physiology_feature_names",
    }


def test_profile_and_seed_split_is_disjoint() -> None:
    audit = split_identity_audit(
        {
            "train": (("train_profile_000", 17), ("train_profile_001", 18)),
            "validation": (("validation_profile_000", 117),),
            "locked_test": (("locked_test_profile_000", 217),),
        }
    )
    assert audit["disjoint"]


def test_smoke_benchmark_writes_paired_scenarios_and_resumes(tmp_path) -> None:
    config = SimulationBenchmarkConfig(
        run_id="simulation-smoke",
        output_root=str(tmp_path),
        duration_s=60.0,
        split_specs=smoke_split_specs(),
        observation_scenarios=smoke_observation_scenarios(),
        resume=True,
        paired_observation_seed=True,
    )
    first = generate_benchmark(config)

    assert first.latent_sortie_count == 4
    assert first.observed_scenario_count == 8
    assert first.resumed_scenario_count == 0
    assert first.split_identity["disjoint"]
    assert all(row["latent_hash_shared"] for row in first.paired_rows)
    assert Path(first.simulation_manifest_path).exists()
    manifest = json.loads(Path(first.simulation_manifest_path).read_text(encoding="utf-8"))
    seeds_by_trajectory = {}
    for row in manifest["scenario_rows"]:
        seeds_by_trajectory.setdefault(row["trajectory_id"], set()).add(
            row["observation_seed"]
        )
    assert manifest["paired_observation_seed"] is True
    assert all(len(values) == 1 for values in seeds_by_trajectory.values())

    second = generate_benchmark(config)
    assert second.resumed_scenario_count == 8
    assert second.validation_rows == first.validation_rows
