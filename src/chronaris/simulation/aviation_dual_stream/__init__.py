"""Method-independent aviation human-machine dual-stream generator."""

from .config import (
    AviationScenarioConfig,
    ObservationScenarioConfig,
    canonical_observation_scenarios,
    locked_stress_observation_scenarios,
)
from .benchmark import (
    SimulationBenchmarkConfig,
    SimulationBenchmarkResult,
    SimulationSplitSpec,
    formal_split_specs,
    generate_benchmark,
    smoke_observation_scenarios,
    smoke_split_specs,
)
from .contracts import (
    LatentTrajectory,
    ManeuverEvent,
    ObservedStream,
    ObservationTrace,
    SimulatedDualStreamSortie,
)
from .generator import generate_latent_trajectory, generate_sortie, render_sortie
from .profiles import PilotProfile, sample_pilot_profile

__all__ = [
    "AviationScenarioConfig",
    "LatentTrajectory",
    "ManeuverEvent",
    "ObservationScenarioConfig",
    "ObservedStream",
    "ObservationTrace",
    "PilotProfile",
    "SimulatedDualStreamSortie",
    "SimulationBenchmarkConfig",
    "SimulationBenchmarkResult",
    "SimulationSplitSpec",
    "canonical_observation_scenarios",
    "locked_stress_observation_scenarios",
    "generate_latent_trajectory",
    "generate_benchmark",
    "generate_sortie",
    "render_sortie",
    "formal_split_specs",
    "sample_pilot_profile",
    "smoke_observation_scenarios",
    "smoke_split_specs",
]
