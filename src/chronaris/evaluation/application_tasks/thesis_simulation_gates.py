"""Final simulation hard-gate audit before public outer results open."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import perturb_future_observations
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import load_simulation_locked_pretraining_data
from chronaris.evaluation.application_tasks.simulation_mechanism_consumer_run import TARGETS, _evaluate_predictions
from chronaris.evaluation.application_tasks.simulation_mechanism_context_data import load_simulation_mechanism_context_data
from chronaris.evaluation.application_tasks.simulation_mechanism_targets import build_simulation_mechanism_targets
from chronaris.evaluation.application_tasks.simulation_stress_context_data import load_simulation_stress_context_data
from chronaris.modeling.fusion_encoders.chronaris_physics import physics_audit_to_rows
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training import TrainedFusionAdapter, load_common_pretraining_checkpoint
from chronaris.representation import select_observation_batch
from chronaris.simulation.aviation_dual_stream import canonical_observation_scenarios, locked_stress_observation_scenarios
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


REPO = Path(__file__).resolve().parents[4]
SIMULATION_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
STRESS_ROOT = REPO / "artifacts/application_evaluation/2026-07-12_aviation-simulation-locked-stress"
PRETRAINING = REPO / "artifacts/application_evaluation/2026-09-03_thesis-simulation-pretraining-v3p2"
ABLATION = REPO / "artifacts/application_evaluation/2026-09-03_thesis-simulation-ablation-pretraining-v3p2p1"
FULL_MECHANISM = REPO / "docs/artifacts/runs/2026-09-04_thesis-simulation-mechanism-consumers-v3p2p3/metric_long.csv"
CLEAN_METRICS = REPO / "docs/artifacts/runs/2026-09-03_thesis-simulation-consumers-v3p2p1/metric_long.csv"
CANDIDATE_GATES = REPO / "docs/artifacts/runs/2026-09-02_candidate-screen-v3p2/gate_audit.json"
ORCHESTRATION = REPO / "docs/artifacts/runs/2026-09-04_thesis-simulation-v3p2p3/run_state.json"
TRAINING_TRUST = REPO / "docs/artifacts/runs/2026-09-01_training-trust-repair/acceptance_checks.csv"
NATIVE_TIME = REPO / "docs/artifacts/runs/2026-09-01_native-time-euler-repair/acceptance_checks.csv"
SEMANTIC_OBJECTIVES = REPO / "docs/artifacts/runs/2026-09-01_learnable-semantic-objectives/acceptance_checks.csv"
ODE_VALIDATION = REPO / "docs/artifacts/runs/2026-09-01_ode-solver-validation/acceptance_checks.csv"
NATIVE_PREPROCESSING = REPO / (
    "docs/artifacts/runs/2026-09-03_native-physiology-integrity/"
    "acceptance_checks.csv"
)
PROTOCOL = REPO / "docs/requirements/thesis-frozen-paper-evaluation-v3.2.3.md"
SEEDS = (17, 29, 43)


def run_simulation_gate_audit(
    *, compact_root, heavy_root, device="cuda", resume=True
):
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("simulation gate audit requires CUDA")
    if not resume:
        raise RuntimeError("simulation gate audit requires resume protection")
    _require_completed_orchestration()
    compact = Path(compact_root)
    heavy = Path(heavy_root)
    compact.mkdir(parents=True, exist_ok=True)
    heavy.mkdir(parents=True, exist_ok=True)
    physical_rows, future_rows = _physical_and_future(device=device)
    no_continuous_rows = _no_continuous_recovery(
        heavy,
        device=device,
        resume=resume,
    )
    continuous_rows, continuous_pass = _continuous_gate(no_continuous_rows)
    bypass_rows, bypass_pass = _bypass_gate()
    candidate = json.loads(CANDIDATE_GATES.read_text(encoding="utf-8"))
    protocol_evidence = {
        "training_trust": _acceptance_ids_pass(
            TRAINING_TRUST,
            {f"TR{index:02d}" for index in range(1, 12)},
        ),
        "native_time": _acceptance_ids_pass(
            NATIVE_TIME,
            {f"NT{index:02d}" for index in range(1, 13)},
        ),
        "native_preprocessing_integrity": _acceptance_ids_pass(
            NATIVE_PREPROCESSING,
            {f"NP{index:02d}" for index in range(1, 7)},
        ),
        "semantic_objectives": _acceptance_ids_pass(
            SEMANTIC_OBJECTIVES,
            {f"SO{index:02d}" for index in range(1, 15)},
        ),
        "ode_selection": _acceptance_ids_pass(
            ODE_VALIDATION,
            {"OV01", "OV02", "OV03", "OV04", "OV05", "OV06", "OV08", "OV09"},
        ),
        "candidate_screen": (
            candidate.get("protocol_gate_passed") is True
            and candidate.get("physical_scope_gate_passed") is True
        ),
    }
    protocol_gate_passed = (
        all(protocol_evidence.values())
        and len(candidate.get("explicit_shift_gate", {}).get("rows", ()))
        == len(SEEDS)
        and len(candidate.get("event_pair_gate", {}).get("rows", ()))
        == len(SEEDS)
    )
    gates = {
        "explicit_time_shift": bool(candidate["explicit_shift_gate"]["passed"]),
        "event_response_pairing": bool(candidate["event_pair_gate"]["passed"]),
        "continuous_evolution": continuous_pass,
        "physical_consistency": _physical_gate(physical_rows),
        "future_information_isolation": max(
            row["maximum_historical_delta"] for row in future_rows
        )
        <= 1e-6,
        "safe_single_stream_bypass": bypass_pass,
    }
    audit = {
        "format": "chronaris.thesis_simulation_hard_gates.v1",
        "protocol_version": "v3.2.3",
        "outer_results_authorized": False,
        "safe_bypass_aggregation": "every_frozen_task_consumer_seed_must_pass",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
        ).strip(),
        "evaluation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "device": device,
        "resume": resume,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "gates": gates,
        "protocol_gate_passed": protocol_gate_passed,
        "protocol_evidence": protocol_evidence,
        "all_hard_gates_passed": protocol_gate_passed and all(gates.values()),
        "physical_rows": physical_rows,
        "future_rows": future_rows,
        "continuous_rows": continuous_rows,
        "no_continuous_metric_count": len(no_continuous_rows),
        "bypass_rows": bypass_rows,
        "source_files": {
            "orchestration_sha256": sha256_file(ORCHESTRATION),
            "full_mechanism_sha256": sha256_file(FULL_MECHANISM),
            "clean_metrics_sha256": sha256_file(CLEAN_METRICS),
            "candidate_gate_sha256": sha256_file(CANDIDATE_GATES),
            "training_trust_sha256": sha256_file(TRAINING_TRUST),
            "native_time_sha256": sha256_file(NATIVE_TIME),
            "semantic_objectives_sha256": sha256_file(SEMANTIC_OBJECTIVES),
            "ode_validation_sha256": sha256_file(ODE_VALIDATION),
            "evaluation_protocol_sha256": sha256_file(PROTOCOL),
            "native_preprocessing_sha256": (
                sha256_file(NATIVE_PREPROCESSING)
                if NATIVE_PREPROCESSING.is_file()
                else None
            ),
            "training_checkpoint_sha256": _checkpoint_hashes(),
        },
    }
    _write_json(compact / "gate_audit.json", audit)
    _write_csv(compact / "physical_residuals.csv", physical_rows)
    _write_csv(compact / "future_information.csv", future_rows)
    _write_csv(
        compact / "no_continuous_metric_long.csv",
        no_continuous_rows,
    )
    _write_csv(compact / "continuous_recovery.csv", continuous_rows)
    _write_csv(compact / "safe_bypass.csv", bypass_rows)
    (compact / "report.md").write_text(_report(audit), encoding="utf-8")
    (compact / "resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/research/run_thesis_simulation_gates.py "
        "--protocol-version v3.2.3 --device cuda --resume\n",
        encoding="utf-8",
    )
    return audit


def _physical_and_future(*, device):
    data = load_simulation_locked_pretraining_data(SIMULATION_ROOT)
    physical_rows = []
    future_rows = []
    sample_ids = data.fold.validation_sample_ids
    for seed in SEEDS:
        full = _adapter(
            PRETRAINING / f"checkpoints/seed_{seed}/chronaris/best.pt",
            data.fold.fold_id,
            device,
        )
        no_physics = _adapter(
            ABLATION
            / f"checkpoints/seed_{seed}/no_physics/chronaris/best.pt",
            data.fold.fold_id,
            device,
        )
        no_physics.encoder.backbone.config = replace(
            no_physics.encoder.backbone.config,
            variant="full",
        )
        for variant, adapter in (("full", full), ("no_physics", no_physics)):
            totals = {}
            counts = {}
            for offset in range(0, len(sample_ids), 4):
                raw = select_observation_batch(
                    data.batch,
                    sample_ids[offset : offset + 4],
                )
                normalized = move_observation_batch(
                    adapter.normalizer.transform(raw),
                    device=device,
                )
                adapter.encoder.eval()
                with torch.inference_mode():
                    output = adapter.encoder(
                        normalized,
                        compute_chronaris_diagnostics=True,
                    )
                for row in physics_audit_to_rows(
                    output.auxiliary["physical_consistency"]
                ):
                    if row["raw_value"] is None:
                        continue
                    name = row["component_name"]
                    totals[name] = totals.get(name, 0.0) + (
                        row["raw_value"] * row["count"]
                    )
                    counts[name] = counts.get(name, 0) + row["count"]
            physical_rows.extend(
                {
                    "seed": seed,
                    "variant": variant,
                    "component": name,
                    "raw_residual": totals[name] / counts[name],
                    "count": counts[name],
                }
                for name in sorted(totals)
            )
        raw = select_observation_batch(data.batch, sample_ids[:4])
        baseline = full(raw)
        changed = full(perturb_future_observations(raw, cutoff_s=15.0))
        past = baseline.timestamps_s <= 15.0
        delta = (
            baseline.sequence_embedding[past]
            - changed.sequence_embedding[past]
        ).abs()
        future_rows.append(
            {
                "seed": seed,
                "cutoff_s": 15.0,
                "maximum_historical_delta": float(delta.max().detach().cpu()),
            }
        )
        del full, no_physics
        torch.cuda.empty_cache()
    return physical_rows, future_rows


def _no_continuous_recovery(heavy, *, device, resume):
    fold = load_simulation_locked_pretraining_data(SIMULATION_ROOT).fold
    adapters = {
        seed: _adapter(
            ABLATION
            / f"checkpoints/seed_{seed}/no_continuous_evolution/chronaris/best.pt",
            fold.fold_id,
            device,
        )
        for seed in SEEDS
    }
    scenarios = tuple(canonical_observation_scenarios())
    g1 = {}
    g1_ids = {}
    g1_manifest = []
    for role in ("train", "validation"):
        for scenario in scenarios:
            data = load_simulation_mechanism_context_data(
                SIMULATION_ROOT,
                role=role,
                scenario_id=scenario.scenario_id,
            )
            g1_ids[(role, scenario.scenario_id)] = data.batch.sample_ids
            g1_manifest.extend(data.sample_manifest_rows)
            for seed, adapter in adapters.items():
                g1[(seed, role, scenario.scenario_id)] = _pooled(
                    adapter,
                    data.batch,
                    heavy
                    / "pooled"
                    / f"seed{seed}"
                    / role
                    / f"{scenario.scenario_id}.npz",
                    resume=resume,
                )
    g1_targets = build_simulation_mechanism_targets(
        g1_manifest,
        scenarios=scenarios,
        representation_evidence_completed=True,
    )
    models = {}
    for seed in SEEDS:
        train_ids = tuple(
            value
            for scenario in scenarios
            for value in g1_ids[("train", scenario.scenario_id)]
        )
        validation_ids = tuple(
            value
            for scenario in scenarios
            for value in g1_ids[("validation", scenario.scenario_id)]
        )
        train_values = np.concatenate(
            [g1[(seed, "train", scenario.scenario_id)] for scenario in scenarios]
        )
        validation_values = np.concatenate(
            [
                g1[(seed, "validation", scenario.scenario_id)]
                for scenario in scenarios
            ]
        )
        for target in TARGETS:
            model, alpha = fit_regressor(
                train_values,
                g1_targets.values(train_ids, target),
                validation_values,
                g1_targets.values(validation_ids, target),
                alpha_values=(0.1, 1.0, 10.0, 100.0),
                scaler_with_mean=True,
            )
            models[(seed, target)] = (model, alpha)
    stress_scenarios = tuple(locked_stress_observation_scenarios())
    stress = {}
    stress_ids = {}
    stress_manifest = []
    for scenario in stress_scenarios:
        data = load_simulation_stress_context_data(
            STRESS_ROOT,
            scenario_id=scenario.scenario_id,
        )
        stress_ids[scenario.scenario_id] = data.batch.sample_ids
        stress_manifest.extend(data.sample_manifest_rows)
        for seed, adapter in adapters.items():
            stress[(seed, scenario.scenario_id)] = _pooled(
                adapter,
                data.batch,
                heavy
                / "pooled"
                / f"seed{seed}"
                / "held_out"
                / f"{scenario.scenario_id}.npz",
                resume=resume,
            )
    stress_targets = build_simulation_mechanism_targets(
        stress_manifest,
        scenarios=stress_scenarios,
        representation_evidence_completed=True,
    )
    rows = []
    for seed in SEEDS:
        for scenario in stress_scenarios:
            ids = stress_ids[scenario.scenario_id]
            values = stress[(seed, scenario.scenario_id)]
            for target in TARGETS:
                model, alpha = models[(seed, target)]
                prediction = model.predict(values)
                metric_rows, _units = _evaluate_predictions(
                    seed=seed,
                    method="chronaris_no_continuous_evolution",
                    scenario_id=scenario.scenario_id,
                    target_name=target,
                    sample_ids=ids,
                    truth=stress_targets.values(ids, target),
                    prediction=prediction,
                    targets=stress_targets,
                )
                rows.extend({**row, "selected_alpha": alpha} for row in metric_rows)
    del adapters
    torch.cuda.empty_cache()
    return rows


def _continuous_gate(no_continuous_rows):
    full = pd.read_csv(FULL_MECHANISM)
    full = full[
        (full["method"] == "chronaris")
        & (full["metric"] == "mae_s")
        & full["value"].notna()
    ][["seed", "scenario_id", "target", "value"]].rename(
        columns={"value": "full_mae_s"}
    )
    ablated = pd.DataFrame(no_continuous_rows)
    ablated = ablated[ablated["metric"] == "mae_s"][
        ["seed", "scenario_id", "target", "value"]
    ].rename(columns={"value": "no_continuous_mae_s"})
    merged = full.merge(
        ablated,
        on=["seed", "scenario_id", "target"],
        how="inner",
        validate="one_to_one",
    )
    if full.empty or len(merged) != len(full) or len(merged) != len(ablated):
        raise RuntimeError("continuous recovery comparison coverage changed")
    merged["gain_s"] = (
        merged["no_continuous_mae_s"] - merged["full_mae_s"]
    )
    rows = []
    for (target, seed), values in merged.groupby(["target", "seed"]):
        rows.append(
            {
                "target": target,
                "seed": int(seed),
                "scenario_count": len(values),
                "full_mae_s_median": float(values["full_mae_s"].median()),
                "no_continuous_mae_s_median": float(
                    values["no_continuous_mae_s"].median()
                ),
                "median_gain_s": float(values["gain_s"].median()),
                "positive_gain": bool(values["gain_s"].median() > 0),
            }
        )
    frame = pd.DataFrame(rows)
    passed = any(
        len(values) == len(SEEDS)
        and int(values["positive_gain"].sum()) >= 2
        and float(values["median_gain_s"].median()) > 0
        for _target, values in frame.groupby("target")
    )
    return rows, passed


def _physical_gate(rows):
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(
        index=["component", "seed"],
        columns="variant",
        values="raw_residual",
    ).dropna()
    pivot["improvement"] = pivot["no_physics"] - pivot["full"]
    return any(
        len(values) == len(SEEDS)
        and int((values["improvement"] > 0).sum()) >= 2
        and float(values["improvement"].median()) > 0
        for component, values in pivot.groupby(level="component")
        if component in {
            "vehicle_rigid_body_translation",
            "vehicle_rigid_body_vertical",
        }
    )


def _bypass_gate():
    frame = pd.read_csv(CLEAN_METRICS)
    specs = (
        ("simulated_future_workload_classification", "macro_f1", "higher"),
        ("simulated_future_workload_regression", "rmse", "lower"),
    )
    rows = []
    for task, metric, direction in specs:
        selected = frame[
            (frame["task"] == task)
            & (frame["metric"] == metric)
            & (frame["role"] == "held_out")
        ]
        selected = selected[selected["method"].isin(
            ("physiology_only", "vehicle_only", "chronaris")
        )]
        keys = ("consumer", "seed", "method")
        expected = {
            (consumer, seed, method)
            for consumer in ("linear", "minirocket")
            for seed in SEEDS
            for method in ("physiology_only", "vehicle_only", "chronaris")
        }
        actual = set(selected[list(keys)].itertuples(index=False, name=None))
        if actual != expected or selected.duplicated(list(keys)).any():
            raise RuntimeError("safe bypass comparison coverage changed")
        if not (selected["status"] == "available").all() or not np.isfinite(
            selected["value"].to_numpy(dtype=float)
        ).all():
            raise RuntimeError("safe bypass comparison contains unavailable values")
        if (selected["value"] < 0).any() or (
            metric == "macro_f1" and (selected["value"] > 1).any()
        ):
            raise RuntimeError("safe bypass comparison contains invalid metric values")
        for (consumer, seed), values in selected.groupby(["consumer", "seed"]):
            by_method = dict(zip(values["method"], values["value"], strict=True))
            singles = (by_method["physiology_only"], by_method["vehicle_only"])
            chronaris = by_method["chronaris"]
            if direction == "higher":
                best = max(singles)
                degradation = best - chronaris
                threshold = 0.05
                passed = chronaris >= best - threshold
            else:
                best = min(singles)
                degradation = (
                    chronaris / best - 1.0 if best > 0
                    else (0.0 if chronaris == 0 else None)
                )
                threshold = 0.05
                passed = chronaris <= best * (1.0 + threshold)
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "consumer": consumer,
                    "seed": int(seed),
                    "chronaris_value": float(chronaris),
                    "best_single_value": float(best),
                    "degradation": degradation,
                    "threshold": threshold,
                    "passed": bool(passed),
                }
            )
    passed = len(rows) == len(specs) * 2 * len(SEEDS) and all(
        row["passed"] for row in rows
    )
    return rows, passed


def _adapter(path, fold_id, device):
    encoder, _heads, normalizer, _payload = load_common_pretraining_checkpoint(
        path,
        device=device,
    )
    return TrainedFusionAdapter(
        encoder=encoder,
        normalizer=normalizer,
        fold_id=fold_id,
        checkpoint_sha256=sha256_file(path),
    )


def _pooled(adapter, batch, cache_path, *, resume):
    cache_path = Path(cache_path)
    if resume and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            if (
                tuple(payload["sample_ids"].tolist()) == batch.sample_ids
                and tuple(payload["source_hashes"].tolist())
                == batch.source_sample_hashes
                and str(payload["checkpoint_sha256"].item())
                == adapter.checkpoint_sha256
            ):
                return payload["pooled"]
    rows = []
    for offset in range(0, len(batch.sample_ids), 32):
        ids = batch.sample_ids[offset : offset + 32]
        rows.append(
            adapter(select_observation_batch(batch, ids))
            .pooled_embedding.detach()
            .cpu()
            .numpy()
        )
    pooled = np.concatenate(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray(batch.sample_ids),
            source_hashes=np.asarray(batch.source_sample_hashes),
            checkpoint_sha256=np.asarray(adapter.checkpoint_sha256),
            pooled=pooled,
        )
    temporary.replace(cache_path)
    return pooled


def _require_completed_orchestration():
    state = json.loads(ORCHESTRATION.read_text(encoding="utf-8"))
    if state.get("protocol_version") != "v3.2.3" or state.get("outer_results_authorized") is not False:
        raise RuntimeError("simulation audit requires v3.2.3 with real outer results closed")
    required = {
        "pretraining",
        "representations",
        "consumers",
        "ablation_pretraining",
        "ablation_representations",
        "ablation_consumers",
        "stress_representations",
        "stress_consumers",
        "mechanism_representations",
        "mechanism_consumers",
    }
    results = state.get("results", {})
    if not required.issubset(results) or any(
        results[name].get("status") not in {"completed", "reused_completed"}
        for name in required
    ):
        raise RuntimeError("simulation evidence chain is incomplete")
    for path in (FULL_MECHANISM, CLEAN_METRICS, CANDIDATE_GATES):
        if not path.is_file():
            raise FileNotFoundError(path)


def _checkpoint_hashes():
    paths = {
        f"full_seed{seed}": PRETRAINING
        / f"checkpoints/seed_{seed}/chronaris/best.pt"
        for seed in SEEDS
    }
    paths.update(
        {
            f"{variant}_seed{seed}": ABLATION
            / f"checkpoints/seed_{seed}/{variant}/chronaris/best.pt"
            for variant in ("no_continuous_evolution", "no_physics")
            for seed in SEEDS
        }
    )
    return {name: sha256_file(path) for name, path in paths.items()}


def _acceptance_ids_pass(path, required_ids):
    if not Path(path).is_file():
        return False
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = {row["check_id"]: row["status"] for row in csv.DictReader(handle)}
    return all(rows.get(check_id) == "pass" for check_id in required_ids)


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _report(audit):
    labels = {
        "explicit_time_shift": "五分类显式时移",
        "event_response_pairing": "事件—响应配对",
        "continuous_evolution": "连续演化",
        "physical_consistency": "运动学一致性",
        "future_information_isolation": "未来信息隔离",
        "safe_single_stream_bypass": "安全单流旁路",
    }
    lines = [
        "# 论文主线受控仿真硬门审计",
        "",
        "结论："
        + (
            "协议门与六项机制硬门全部通过。"
            if audit["all_hard_gates_passed"]
            else "至少一项协议或机制硬门未通过。"
        ),
        "",
        "| 硬门 | 结果 |",
        "|---|---|",
        f"| 训练与评价协议 | {'通过' if audit['protocol_gate_passed'] else '未通过'} |",
    ]
    lines.extend(
        f"| {labels[name]} | {'通过' if passed else '未通过'} |"
        for name, passed in audit["gates"].items()
    )
    lines.extend(
        (
            "",
            "连续演化使用受控压力场景的时钟偏移与生理响应时延恢复误差；运动学一致性比较完整模型与无该目标重训模型的有效平移或垂向残差；安全旁路比较分类宏平均 F1 和回归均方根误差（RMSE）与最佳单流的差距。仿真材料只承担受控机制验证，不等价于新增鼎新数据。",
            "安全旁路要求每个冻结任务、消费者和随机种子单元均满足原阈值，不采用多数通过。所有不利单元保留；公开数据与鼎新外层评价继续关闭。",
            "",
        )
    )
    return "\n".join(lines)
