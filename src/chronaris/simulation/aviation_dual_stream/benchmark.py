"""Resumable smoke and formal benchmark generation orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .config import AviationScenarioConfig, ObservationScenarioConfig, canonical_observation_scenarios
from .deterministic_npz import sha256_file
from .generator import generate_latent_trajectory, render_sortie
from .profiles import sample_pilot_profile
from .storage import StoredScenario, store_scenario
from .validation import split_identity_audit, validate_paired_observations, validate_sortie


ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class SimulationSplitSpec:
    split_id: str
    generator_family: str
    profile_count: int
    trajectories_per_profile: int
    profile_seed_base: int
    latent_seed_base: int
    observation_seed_base: int

    @property
    def latent_sortie_count(self) -> int:
        return self.profile_count * self.trajectories_per_profile

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulationBenchmarkConfig:
    run_id: str
    output_root: str = "artifacts/application_evaluation"
    duration_s: float = 180.0
    truth_rate_hz: float = 20.0
    split_specs: tuple[SimulationSplitSpec, ...] = ()
    observation_scenarios: tuple[ObservationScenarioConfig, ...] = ()
    resume: bool = True
    paired_observation_seed: bool = False

    @property
    def run_root(self) -> Path:
        return Path(self.output_root) / self.run_id


@dataclass(frozen=True, slots=True)
class SimulationBenchmarkResult:
    run_id: str
    status: str
    run_root: str
    simulation_manifest_path: str
    latent_sortie_count: int
    observed_scenario_count: int
    validation_rows: tuple[Mapping[str, object], ...]
    paired_rows: tuple[Mapping[str, object], ...]
    split_identity: Mapping[str, object]
    resumed_scenario_count: int


def formal_split_specs() -> tuple[SimulationSplitSpec, ...]:
    return (
        SimulationSplitSpec("train", "g1_state_space", 16, 6, 10_000, 100_000, 1_000_000),
        SimulationSplitSpec("validation", "g1_state_space", 4, 6, 20_000, 200_000, 2_000_000),
        SimulationSplitSpec("locked_test", "g2_event_spline", 8, 6, 30_000, 300_000, 3_000_000),
    )


def smoke_split_specs() -> tuple[SimulationSplitSpec, ...]:
    return (
        SimulationSplitSpec("smoke_g1", "g1_state_space", 2, 1, 40_000, 400_000, 4_000_000),
        SimulationSplitSpec("smoke_g2", "g2_event_spline", 2, 1, 50_000, 500_000, 5_000_000),
    )


def smoke_observation_scenarios() -> tuple[ObservationScenarioConfig, ...]:
    canonical = {scenario.scenario_id: scenario for scenario in canonical_observation_scenarios()}
    return (canonical["clean_asynchronous"], canonical["mixed_severe"])


def generate_benchmark(
    config: SimulationBenchmarkConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> SimulationBenchmarkResult:
    """Generate all latent trajectories and paired observation variants."""

    split_specs = config.split_specs or formal_split_specs()
    scenarios = config.observation_scenarios or canonical_observation_scenarios()
    config.run_root.mkdir(parents=True, exist_ok=True)
    validation_rows: list[Mapping[str, object]] = []
    paired_rows: list[Mapping[str, object]] = []
    scenario_rows: list[Mapping[str, object]] = []
    identity_rows: dict[str, list[tuple[str, int]]] = {}
    resumed_count = 0
    latent_count = 0
    for split in split_specs:
        identity_rows[split.split_id] = []
        for profile_index in range(split.profile_count):
            profile_seed = split.profile_seed_base + profile_index
            profile = sample_pilot_profile(
                split_id=split.split_id,
                profile_index=profile_index,
                seed=profile_seed,
            )
            for trajectory_index in range(split.trajectories_per_profile):
                split_trajectory_index = (
                    profile_index * split.trajectories_per_profile + trajectory_index
                )
                latent_seed = (
                    split.latent_seed_base
                    + split_trajectory_index
                )
                identity_rows[split.split_id].append((profile.profile_id, latent_seed))
                latent = generate_latent_trajectory(
                    AviationScenarioConfig(
                        generator_family=split.generator_family,
                        duration_s=config.duration_s,
                        truth_rate_hz=config.truth_rate_hz,
                        trajectory_index=split_trajectory_index,
                    ),
                    profile,
                    latent_seed,
                )
                latent_count += 1
                paired_sorties = []
                for scenario_index, scenario in enumerate(scenarios):
                    observation_seed = (
                        split.observation_seed_base
                        + (profile_index * split.trajectories_per_profile + trajectory_index) * 100
                        + (0 if config.paired_observation_seed else scenario_index)
                    )
                    sortie = render_sortie(
                        latent,
                        scenario,
                        observation_seed=observation_seed,
                    )
                    paired_sorties.append(sortie)
                    stored, resumed = _store_or_resume(
                        sortie,
                        output_root=config.run_root,
                        split_id=split.split_id,
                        resume=config.resume,
                    )
                    resumed_count += int(resumed)
                    validation = {
                        "split_id": split.split_id,
                        "profile_id": profile.profile_id,
                        "profile_seed": profile_seed,
                        "trajectory_index": trajectory_index,
                        "split_trajectory_index": split_trajectory_index,
                        **validate_sortie(sortie),
                    }
                    validation_rows.append(validation)
                    validation_path = Path(stored.scenario_root) / "validation.json"
                    _write_json(validation_path, validation)
                    scenario_rows.append(
                        {
                            "split_id": split.split_id,
                            "profile_id": profile.profile_id,
                            "trajectory_id": latent.trajectory_id,
                            "scenario_id": scenario.scenario_id,
                            "observation_seed": observation_seed,
                            "scenario_manifest_path": stored.scenario_manifest_path,
                            "raw_sha256": stored.raw_sha256,
                            "ground_truth_sha256": stored.ground_truth_sha256,
                            "resumed": resumed,
                        }
                    )
                paired_rows.append(
                    {
                        "split_id": split.split_id,
                        "profile_id": profile.profile_id,
                        **validate_paired_observations(paired_sorties),
                    }
                )
                _notify(
                    progress_callback,
                    "latent_trajectory_completed",
                    {
                        "split_id": split.split_id,
                        "profile_id": profile.profile_id,
                        "trajectory_index": trajectory_index,
                        "completed_latent_count": latent_count,
                        "completed_scenario_count": len(scenario_rows),
                    },
                )
    split_identity = split_identity_audit(identity_rows)
    manifest = {
        "run_id": config.run_id,
        "status": "completed",
        "duration_s": config.duration_s,
        "truth_rate_hz": config.truth_rate_hz,
        "split_specs": [split.to_dict() for split in split_specs],
        "observation_scenarios": [scenario.to_dict() for scenario in scenarios],
        "latent_sortie_count": latent_count,
        "observed_scenario_count": len(scenario_rows),
        "resumed_scenario_count": resumed_count,
        "paired_observation_seed": config.paired_observation_seed,
        "split_identity": split_identity,
        "scenario_rows": scenario_rows,
    }
    manifest_path = config.run_root / "simulation_manifest.json"
    _write_json(manifest_path, manifest)
    _write_jsonl(config.run_root / "validation_rows.jsonl", validation_rows)
    _write_jsonl(config.run_root / "paired_rows.jsonl", paired_rows)
    return SimulationBenchmarkResult(
        run_id=config.run_id,
        status="completed",
        run_root=str(config.run_root),
        simulation_manifest_path=str(manifest_path),
        latent_sortie_count=latent_count,
        observed_scenario_count=len(scenario_rows),
        validation_rows=tuple(validation_rows),
        paired_rows=tuple(paired_rows),
        split_identity=split_identity,
        resumed_scenario_count=resumed_count,
    )


def _store_or_resume(
    sortie,
    *,
    output_root: Path,
    split_id: str,
    resume: bool,
) -> tuple[StoredScenario, bool]:
    scenario_root = (
        output_root
        / split_id
        / sortie.latent.profile_id
        / sortie.latent.trajectory_id
        / sortie.observation_config.scenario_id
    )
    manifest_path = scenario_root / "scenario_manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_path = Path(manifest["raw_dual_stream_path"])
        truth_path = Path(manifest["ground_truth_path"])
        if (
            manifest.get("latent_hash") == sortie.latent_hash
            and raw_path.exists()
            and truth_path.exists()
            and sha256_file(raw_path) == manifest.get("raw_dual_stream_sha256")
            and sha256_file(truth_path) == manifest.get("ground_truth_sha256")
        ):
            return (
                StoredScenario(
                    scenario_root=str(scenario_root),
                    scenario_manifest_path=str(manifest_path),
                    raw_dual_stream_path=str(raw_path),
                    ground_truth_path=str(truth_path),
                    task_manifest_path=str(scenario_root / "task_manifest.jsonl"),
                    raw_sha256=str(manifest["raw_dual_stream_sha256"]),
                    ground_truth_sha256=str(manifest["ground_truth_sha256"]),
                ),
                True,
            )
    return store_scenario(sortie, output_root=output_root, split_id=split_id), False


def _notify(
    callback: ProgressCallback | None,
    event: str,
    fields: Mapping[str, object],
) -> None:
    if callback is not None:
        callback(event, fields)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
