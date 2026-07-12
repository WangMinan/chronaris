"""Generate and seal an independent, method-neutral v2 confirmation family."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_protocol import (
    write_sealed_confirmation_manifest,
)
from chronaris.simulation.aviation_dual_stream import (
    SimulationBenchmarkConfig,
    SimulationSplitSpec,
    generate_benchmark,
    locked_stress_observation_scenarios,
)


@dataclass(frozen=True, slots=True)
class ChronarisV2ConfirmationFamilyConfig:
    heavy_run_id: str = "2026-07-12_chronaris-v2-independent-confirmation-family"
    compact_run_id: str = "2026-07-12_chronaris-v2-confirmation-family-seal"
    heavy_output_root: str = "artifacts/application_evaluation"
    compact_output_root: str = "docs/artifacts/runs"
    profile_count: int = 8
    trajectories_per_profile: int = 6
    full_stress_scenarios: bool = True
    resume: bool = True

    def __post_init__(self) -> None:
        if self.profile_count <= 0 or self.trajectories_per_profile <= 0:
            raise ValueError("confirmation family counts must be positive")


def prepare_chronaris_v2_confirmation_family(
    config: ChronarisV2ConfirmationFamilyConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2ConfirmationFamilyConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.compact_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    split = SimulationSplitSpec(
        split_id="sealed_confirmation",
        generator_family="g2_event_spline",
        profile_count=resolved.profile_count,
        trajectories_per_profile=resolved.trajectories_per_profile,
        profile_seed_base=130_000,
        latent_seed_base=1_300_000,
        observation_seed_base=13_000_000,
    )
    stress = locked_stress_observation_scenarios()
    scenarios = stress if resolved.full_stress_scenarios else stress[:2]
    result = generate_benchmark(
        SimulationBenchmarkConfig(
            run_id=resolved.heavy_run_id,
            output_root=resolved.heavy_output_root,
            split_specs=(split,),
            observation_scenarios=scenarios,
            resume=resolved.resume,
            paired_observation_seed=True,
        )
    )
    heavy_root = Path(result.run_root)
    payload_paths = tuple(sorted(heavy_root.glob("**/*.npz")))
    expected_payload_count = result.observed_scenario_count * 2
    if len(payload_paths) != expected_payload_count:
        raise ValueError("confirmation family payload count changed")
    protocol = {
        "format": "chronaris.v2_independent_confirmation_generation.v1",
        "split": split.to_dict(),
        "scenario_count": len(scenarios),
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "method_scope": [
            "physiology_only",
            "vehicle_only",
            "naive_time_sync",
            "mult",
            "contiformer",
            "chronaris",
        ],
        "generated_before_configuration_lock": True,
        "available_to_model_selection": False,
    }
    sealed = write_sealed_confirmation_manifest(
        compact_root / "sealed_confirmation_manifest.json",
        family_id=resolved.heavy_run_id,
        generator_protocol=protocol,
        payload_paths=payload_paths,
    )
    acceptance = (
        _check("generation_completed", result.status == "completed"),
        _check(
            "latent_family_complete",
            result.latent_sortie_count
            == resolved.profile_count * resolved.trajectories_per_profile,
        ),
        _check("payload_count_complete", len(payload_paths) == expected_payload_count),
        _check(
            "seed_ranges_independent",
            split.profile_seed_base > 30_000
            and split.latent_seed_base > 300_000
            and split.observation_seed_base > 3_000_000,
        ),
        _check("all_six_methods_declared", sealed.method_count == 6),
        _check("family_remains_locked", sealed.unlocked is False),
    )
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(compact_root / "generation_protocol.json", protocol)
    _write_json(
        compact_root / "evidence_manifest.json",
        {
            "format": "chronaris.v2_confirmation_family_seal_evidence.v1",
            "run_id": resolved.compact_run_id,
            "status": (
                "sealed"
                if all(row["passed"] for row in acceptance)
                else "partial"
            ),
            "family_id": resolved.heavy_run_id,
            "latent_sortie_count": result.latent_sortie_count,
            "observed_scenario_count": result.observed_scenario_count,
            "payload_file_count": len(payload_paths),
            "payload_sha256": sealed.payload_sha256,
            "configuration_locked": False,
            "available_to_model_selection": False,
            "confirmed_metrics_changed": False,
            "heavy_run_root": str(heavy_root),
        },
    )
    (compact_root / "report.md").write_text(
        "\n".join(
            (
                "# Chronaris v2 独立仿真确认族封存",
                "",
                f"已生成 {result.latent_sortie_count} 条独立潜在轨迹和 {result.observed_scenario_count} 个统一观测场景，封存 {len(payload_paths)} 个原始观测/真值文件。",
                "profile、潜在轨迹和观测随机种子区间与既有开发/锁定仿真分离；六种方法共享同一确认族。",
                "当前只允许校验封存哈希，模型配置锁定前禁止表示导出、任务评价或候选选择访问。",
                "",
            )
        ),
        encoding="utf-8",
    )
    return compact_root


def _check(name: str, passed: bool):
    return {"check": name, "passed": bool(passed)}


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
