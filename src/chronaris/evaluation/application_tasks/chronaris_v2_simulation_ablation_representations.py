"""Export clean G1/G2 representations for locked Chronaris v2 ablations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.chronaris_v2_simulation_ablation_pretraining import (
    V2_FORMAL_ABLATION_STRUCTURES,
)
from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_representation_run import (
    _export_role,
)
from chronaris.evaluation.application_tasks.simulation_locked_context_data import (
    load_simulation_locked_context_data,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import LOCKED_SEEDS
from chronaris.modeling.training import TrainedFusionAdapter, load_common_pretraining_checkpoint
from chronaris.representation import select_observation_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class ChronarisV2SimulationAblationRepresentationConfig:
    run_id: str = "2026-07-13_chronaris-v2-simulation-ablation-representations"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-13_chronaris-v2-simulation-ablation-pretraining"
    simulation_root: str = "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    seeds: tuple[int, ...] = LOCKED_SEEDS
    variants: tuple[str, ...] = tuple(V2_FORMAL_ABLATION_STRUCTURES)
    export_batch_size: int = 32
    device: str = "cuda"
    resume: bool = True


def run_chronaris_v2_simulation_ablation_representations(
    config: ChronarisV2SimulationAblationRepresentationConfig | None = None,
) -> Path:
    resolved = config or ChronarisV2SimulationAblationRepresentationConfig()
    compact_root = Path(resolved.compact_output_root) / resolved.run_id
    heavy_root = Path(resolved.heavy_output_root) / resolved.run_id
    pretraining_compact = Path(resolved.compact_output_root) / resolved.pretraining_run_id
    pretraining_heavy = Path(resolved.heavy_output_root) / resolved.pretraining_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((pretraining_compact / "evidence_manifest.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "completed":
        raise ValueError("v2 ablation representation export requires completed pretraining")
    checkpoints = _require_checkpoints(pretraining_heavy, resolved)
    data = load_simulation_locked_context_data(resolved.simulation_root)
    rows = []
    inventory = []
    for seed in resolved.seeds:
        for variant in resolved.variants:
            checkpoint = checkpoints[(seed, variant)]
            encoder, _heads, normalizer, payload = load_common_pretraining_checkpoint(
                checkpoint, device=resolved.device
            )
            adapter = TrainedFusionAdapter(
                encoder=encoder,
                normalizer=normalizer,
                fold_id=data.fold.fold_id,
                checkpoint_sha256=sha256_file(checkpoint),
            )
            method_name = f"chronaris_{variant}"
            for role in ("train", "validation", "held_out"):
                destination = heavy_root / "representations" / f"seed_{seed}" / method_name / role
                output, status = _export_role(
                    adapter=adapter,
                    method_name=method_name,
                    batch=select_observation_batch(data.batch, data.role_sample_ids[role]),
                    destination=destination,
                    batch_size=resolved.export_batch_size,
                    resume=resolved.resume,
                )
                rows.append({
                    "seed": seed, "variant": variant, "method_name": method_name,
                    "role": role, "status": status, "sample_count": len(output.sample_ids),
                    "checkpoint_sha256": output.checkpoint_sha256,
                    "representation_sha256": sha256_file(destination / "fusion_stream.npz"),
                    "output_root": str(destination),
                })
            inventory.append({
                "seed": seed,
                "variant": variant,
                "structure_candidate_id": payload["candidate_config"]["structure_candidate_id"],
                "checkpoint_path": str(checkpoint),
                "task_oracle_opened": False,
            })
            del encoder, adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    expected = len(resolved.seeds) * len(resolved.variants) * 3
    acceptance = (
        _check("all_variant_role_exports", len(rows) == expected),
        _check("fixed_output_contract", all(row["sample_count"] > 0 for row in rows)),
        _check("task_oracle_closed", all(not row["task_oracle_opened"] for row in inventory)),
    )
    status = "completed" if all(row["passed"] for row in acceptance) else "partial"
    pd.DataFrame(inventory).to_csv(compact_root / "checkpoint_inventory.csv", index=False)
    pd.DataFrame(rows).to_csv(compact_root / "representation_inventory.csv", index=False)
    pd.DataFrame(acceptance).to_csv(compact_root / "acceptance.csv", index=False)
    _write_json(compact_root / "protocol.json", {
        "format": "chronaris.v2_simulation_ablation_representation_protocol.v1",
        "config": asdict(resolved),
        "task_oracle_opened": False,
        "representation_family": "frozen_task_agnostic_v2_ablation",
    })
    _write_json(compact_root / "evidence_manifest.json", {
        "format": "chronaris.v2_simulation_ablation_representation_evidence.v1",
        "run_id": resolved.run_id,
        "status": status,
        "export_count": len(rows),
        "heavy_run_root": str(heavy_root),
    })
    (compact_root / "report.md").write_text(
        "\n".join((
            "# Chronaris v2 正式机制消融表示导出",
            "",
            f"状态：{status}；完成 {len(rows)}/{expected} 份训练、验证与锁定测试表示。",
            "表示阶段只读取原始观测，不计算下游任务指标。",
            "",
        )), encoding="utf-8",
    )
    (compact_root / "resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/"
        "run_chronaris_v2_simulation_ablation_representations.py "
        f"--run-id {resolved.run_id} "
        f"--pretraining-run-id {resolved.pretraining_run_id} "
        f"--device {resolved.device} --resume\n",
        encoding="utf-8",
    )
    return compact_root


def _require_checkpoints(root, config):
    paths = {}
    for seed in config.seeds:
        for variant in config.variants:
            parent = root / "checkpoints" / f"seed_{seed}" / variant / "chronaris"
            matches = tuple(parent.glob("*/last.pt"))
            if len(matches) != 1:
                raise ValueError(f"v2 ablation checkpoint is not unique: {parent}")
            path = matches[0]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            candidate_id = payload["candidate_config"]["candidate_id"]
            if (
                payload.get("training_status") != "completed"
                or int(payload["config"]["seed"]) != seed
                or not candidate_id.endswith(f"__{variant}")
                or payload["candidate_config"]["structure_candidate_id"]
                != V2_FORMAL_ABLATION_STRUCTURES[variant]
            ):
                raise ValueError("v2 ablation checkpoint is incomplete or mislabeled")
            paths[(seed, variant)] = path
    return paths


def _check(name, passed):
    return {"check": name, "passed": bool(passed)}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
