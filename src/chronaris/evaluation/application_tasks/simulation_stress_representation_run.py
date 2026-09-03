"""Export six-method locked representations for all paired G2 stress scenarios."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
    _encode_in_batches,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.evaluation.application_tasks.simulation_locked_representation_run import (
    _resolve_device,
    load_locked_seed_adapters,
    require_complete_locked_checkpoint_set,
)
from chronaris.evaluation.application_tasks.simulation_stress_context_data import (
    load_simulation_stress_context_data,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import TRAINABLE_FUSION_METHODS
from chronaris.representation import (
    load_fusion_stream_batch,
    validate_fusion_method_alignment,
    write_fusion_stream_batch,
)
from chronaris.simulation.aviation_dual_stream import (
    locked_stress_observation_scenarios,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_stress_representations")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationStressRepresentationConfig:
    run_id: str = "2026-07-12_simulation-locked-stress-representations"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_simulation-locked-pretraining"
    clean_representation_run_id: str = "2026-07-12_simulation-locked-representations"
    stress_generation_run_id: str = "2026-07-12_aviation-simulation-locked-stress"
    stress_audit_run_id: str = "2026-07-12_aviation-simulation-locked-stress-audit"
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    formal_simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    export_batch_size: int = 32
    baseline_device: str = "auto"
    chronaris_device: str = "cpu"
    require_valid_mask_match: bool = True
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationStressRepresentationResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    export_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_stress_representations(config: SimulationStressRepresentationConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_root = Path(config.heavy_output_root) / config.pretraining_run_id
    clean_representation_root = (
        Path(config.heavy_output_root) / config.clean_representation_run_id
    )
    stress_root = Path(config.heavy_output_root) / config.stress_generation_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    _require_completed_evidence(
        Path(config.compact_output_root) / config.clean_representation_run_id
    )
    _require_completed_evidence(
        Path(config.compact_output_root) / config.stress_audit_run_id
    )
    selected = json.loads(Path(config.selected_candidates_path).read_text(encoding="utf-8"))
    selected_ids = {
        method: str(selected[method]["candidate_id"])
        for method in TRAINABLE_FUSION_METHODS
    }
    checkpoints = require_complete_locked_checkpoint_set(
        pretraining_root,
        seeds=config.seeds,
        selected_ids=selected_ids,
    )
    pretraining_data = load_simulation_locked_pretraining_data(
        config.formal_simulation_root
    )
    scenario_ids = tuple(
        scenario.scenario_id for scenario in locked_stress_observation_scenarios()
    )
    baseline_device = _resolve_device(config.baseline_device)
    chronaris_device = _resolve_device(config.chronaris_device)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_locked_stress_representation_export",
        logger=LOGGER,
        initial_progress={
            "seeds": list(config.seeds),
            "scenario_count": len(scenario_ids),
            "task_oracle_opened": False,
            "baseline_device": baseline_device,
            "chronaris_device": chronaris_device,
        },
    ) as progress:
        export_rows = []
        scenario_rows = []
        data_rows = []
        for seed in config.seeds:
            adapters, _checkpoint_rows = load_locked_seed_adapters(
                seed=seed,
                checkpoints=checkpoints,
                selected_ids=selected_ids,
                pretraining_data=pretraining_data,
                heavy_root=clean_representation_root,
                resume=True,
                baseline_device=baseline_device,
                chronaris_device=chronaris_device,
            )
            for scenario_id in scenario_ids:
                data = load_simulation_stress_context_data(
                    stress_root,
                    scenario_id=scenario_id,
                )
                if seed == config.seeds[0]:
                    data_rows.extend(data.sample_manifest_rows)
                outputs = []
                for method in APPLICATION_METHODS:
                    destination = (
                        heavy_root
                        / "representations"
                        / f"seed_{seed}"
                        / method
                        / scenario_id
                    )
                    output, status = _export_one(
                        adapter=adapters[method],
                        batch=data.batch,
                        destination=destination,
                        batch_size=config.export_batch_size,
                        resume=config.resume,
                    )
                    outputs.append(output)
                    export_rows.append(
                        {
                            "seed": seed,
                            "method_name": method,
                            "scenario_id": scenario_id,
                            "status": status,
                            "sample_count": len(output.sample_ids),
                            "checkpoint_sha256": output.checkpoint_sha256,
                            "representation_sha256": sha256_file(
                                destination / "fusion_stream.npz"
                            ),
                            "output_root": str(destination),
                            "task_oracle_opened": False,
                        }
                    )
                alignment = validate_fusion_method_alignment(
                    outputs,
                    require_valid_mask_match=config.require_valid_mask_match,
                )
                scenario_rows.append(
                    {
                        "seed": seed,
                        "scenario_id": scenario_id,
                        "method_count": len(outputs),
                        "context_count": len(data.batch.sample_ids),
                        "alignment_sha256": alignment,
                        "task_oracle_opened": False,
                    }
                )
                progress.update(
                    "stress_scenario_representation_complete",
                    seed=seed,
                    scenario_id=scenario_id,
                )
        acceptance = _acceptance_rows(config, export_rows, scenario_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            export_rows=export_rows,
            scenario_rows=scenario_rows,
            data_rows=data_rows,
            acceptance=acceptance,
            status=status,
            baseline_device=baseline_device,
            chronaris_device=chronaris_device,
        )
        progress.finish(
            status=status,
            export_count=len(export_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationStressRepresentationResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        export_count=len(export_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _export_one(*, adapter, batch, destination, batch_size, resume):
    if resume and (destination / "fusion_stream.npz").is_file():
        output = load_fusion_stream_batch(destination)
        if (
            output.sample_ids == batch.sample_ids
            and output.checkpoint_sha256 == adapter.checkpoint_sha256
        ):
            return output, "resumed"
    output = _encode_in_batches(adapter, batch, batch_size=batch_size)
    write_fusion_stream_batch(output, root=destination, export_role="stress_held_out")
    return load_fusion_stream_batch(destination), "completed"


def _require_completed_evidence(root):
    payload = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"stress representations require completed evidence: {root}")


def _acceptance_rows(config, exports, scenarios):
    expected_scenarios = len(config.seeds) * 35
    expected_exports = expected_scenarios * 6
    return (
        _check("all_seed_scenarios", len(scenarios) == expected_scenarios, len(scenarios), expected_scenarios),
        _check("six_methods_per_scenario", all(row["method_count"] == 6 for row in scenarios), [row["method_count"] for row in scenarios], 6),
        _check("four_contexts_per_trajectory", all(row["context_count"] == 192 for row in scenarios), [row["context_count"] for row in scenarios], 192),
        _check("all_representation_exports", len(exports) == expected_exports, len(exports), expected_exports),
        _check("task_oracle_closed", all(not row["task_oracle_opened"] for row in exports), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "scenarios": root / "scenario_inventory.csv",
        "exports": root / "representation_inventory.csv",
        "data": root / "data_manifest.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame(values["scenario_rows"]).to_csv(paths["scenarios"], index=False)
    pd.DataFrame(values["export_rows"]).to_csv(paths["exports"], index=False)
    pd.DataFrame(values["data_rows"]).to_csv(paths["data"], index=False)
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_stress_representation_protocol.v1",
        "config": asdict(values["config"]),
        "task_oracle_opened": False,
        "frozen_checkpoint_reuse": True,
        "baseline_device": values["baseline_device"],
        "chronaris_device": values["chronaris_device"],
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# G2 锁定压力场景表示导出",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"共导出 {len(values['export_rows'])} 份三随机种子、六方法、35 场景冻结表示。",
        "本阶段只读取原始观测，不打开任务真值或指标。",
        "",
    )), encoding="utf-8")
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_stress_representations.py "
        f"--run-id {values['config'].run_id} --export-batch-size {values['config'].export_batch_size} "
        f"--baseline-device {values['baseline_device']} --chronaris-device {values['chronaris_device']} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_stress_representation_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "export_count": len(values["export_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "task_oracle_opened": False,
        "baseline_device": values["baseline_device"],
        "chronaris_device": values["chronaris_device"],
        "heavy_run_root": str(values["heavy_root"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
