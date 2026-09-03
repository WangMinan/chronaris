#!/usr/bin/env python3
"""Run the frozen v3.2 simulation evidence chain on CUDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_consumer_run import (
    SimulationChronarisAblationConsumerConfig,
    run_simulation_chronaris_ablation_consumers,
)
from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_pretraining_run import (
    SimulationChronarisAblationPretrainingConfig,
    run_simulation_chronaris_ablation_pretraining,
)
from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_representation_run import (
    SimulationChronarisAblationRepresentationConfig,
    run_simulation_chronaris_ablation_representations,
)
from chronaris.evaluation.application_tasks.simulation_locked_consumer_run import (
    SimulationLockedConsumerConfig,
    run_simulation_locked_consumers,
)
from chronaris.evaluation.application_tasks.simulation_locked_representation_run import (
    SimulationLockedRepresentationConfig,
    run_simulation_locked_representations,
)
from chronaris.evaluation.application_tasks.simulation_mechanism_consumer_run import (
    SimulationMechanismConsumerConfig,
    run_simulation_mechanism_consumers,
)
from chronaris.evaluation.application_tasks.simulation_mechanism_representation_run import (
    SimulationMechanismRepresentationConfig,
    run_simulation_mechanism_representations,
)
from chronaris.evaluation.application_tasks.simulation_stress_consumer_run import (
    SimulationStressConsumerConfig,
    run_simulation_stress_consumers,
)
from chronaris.evaluation.application_tasks.simulation_stress_representation_run import (
    SimulationStressRepresentationConfig,
    run_simulation_stress_representations,
)


REPO = Path(__file__).resolve().parents[2]
SELECTED = REPO / "docs/requirements/thesis-frozen-models-v3.2.json"
PROTOCOL = REPO / "docs/requirements/thesis-frozen-paper-evaluation-v3.2.1.md"
STATE_ROOT = REPO / "docs/artifacts/runs/2026-09-03_thesis-simulation-v3p2p1"
PRETRAINING_COMMIT = "58db4458dc64bb452526431413924283deff63bd"
RUNS = {
    "pretraining": "2026-09-03_thesis-simulation-pretraining-v3p2",
    "representations": "2026-09-03_thesis-simulation-representations-v3p2p1",
    "consumers": "2026-09-03_thesis-simulation-consumers-v3p2p1",
    "stress_representations": "2026-09-03_thesis-simulation-stress-representations-v3p2p1",
    "stress_consumers": "2026-09-03_thesis-simulation-stress-consumers-v3p2p1",
    "mechanism_representations": "2026-09-03_thesis-simulation-mechanism-representations-v3p2p1",
    "mechanism_consumers": "2026-09-03_thesis-simulation-mechanism-consumers-v3p2p1",
    "ablation_pretraining": "2026-09-03_thesis-simulation-ablation-pretraining-v3p2p1",
    "ablation_representations": "2026-09-03_thesis-simulation-ablation-representations-v3p2p1",
    "ablation_consumers": "2026-09-03_thesis-simulation-ablation-consumers-v3p2p1",
}
ABLATIONS = (
    "no_continuous_evolution",
    "no_physics",
    "no_causal_mask",
    "no_single_stream_bypass",
)


def main() -> int:
    args = _parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-facing simulation evaluation requires CUDA")
    state = _load_state(args)
    stages = set(args.stages)
    seeds = tuple(args.seeds)
    common = {
        "selected_candidates_path": str(SELECTED),
        "seeds": seeds,
        "resume": args.resume,
    }
    if "pretraining" in stages:
        _record(state, "pretraining", _verify_pretraining())
    if "clean" in stages:
        _record(
            state,
            "representations",
            run_simulation_locked_representations(
                SimulationLockedRepresentationConfig(
                    run_id=RUNS["representations"],
                    pretraining_run_id=RUNS["pretraining"],
                    baseline_device=args.device,
                    chronaris_device=args.device,
                    require_valid_mask_match=False,
                    **common,
                )
            ),
        )
        _record(
            state,
            "consumers",
            run_simulation_locked_consumers(
                SimulationLockedConsumerConfig(
                    run_id=RUNS["consumers"],
                    pretraining_run_id=RUNS["pretraining"],
                    representation_run_id=RUNS["representations"],
                    tcn_device=args.device,
                    **common,
                )
            ),
        )
    if "ablation" in stages:
        _record(
            state,
            "ablation_pretraining",
            run_simulation_chronaris_ablation_pretraining(
                SimulationChronarisAblationPretrainingConfig(
                    run_id=RUNS["ablation_pretraining"],
                    variants=ABLATIONS,
                    max_epochs=50,
                    batch_size=128,
                    patience=8,
                    device=args.device,
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                    fusion_kind="safe_lag",
                    semantic_event_enabled=True,
                    learnable_semantic_queries=True,
                    explicit_shift_weight=0.1,
                    event_pair_weight=0.1,
                    heartbeat_interval_s=30.0,
                    **common,
                )
            ),
        )
        _record(
            state,
            "ablation_representations",
            run_simulation_chronaris_ablation_representations(
                SimulationChronarisAblationRepresentationConfig(
                    run_id=RUNS["ablation_representations"],
                    pretraining_run_id=RUNS["ablation_pretraining"],
                    variants=ABLATIONS,
                    device=args.device,
                    seeds=seeds,
                    resume=args.resume,
                )
            ),
        )
        _record(
            state,
            "ablation_consumers",
            run_simulation_chronaris_ablation_consumers(
                SimulationChronarisAblationConsumerConfig(
                    run_id=RUNS["ablation_consumers"],
                    pretraining_run_id=RUNS["ablation_pretraining"],
                    representation_run_id=RUNS["ablation_representations"],
                    full_pretraining_run_id=RUNS["pretraining"],
                    full_consumer_run_id=RUNS["consumers"],
                    selected_candidates_path=str(SELECTED),
                    variants=ABLATIONS,
                    seeds=seeds,
                    tcn_device=args.device,
                    resume=args.resume,
                )
            ),
        )
    if "stress" in stages:
        _record(
            state,
            "stress_representations",
            run_simulation_stress_representations(
                SimulationStressRepresentationConfig(
                    run_id=RUNS["stress_representations"],
                    pretraining_run_id=RUNS["pretraining"],
                    clean_representation_run_id=RUNS["representations"],
                    baseline_device=args.device,
                    chronaris_device=args.device,
                    require_valid_mask_match=False,
                    **common,
                )
            ),
        )
        _record(
            state,
            "stress_consumers",
            run_simulation_stress_consumers(
                SimulationStressConsumerConfig(
                    run_id=RUNS["stress_consumers"],
                    pretraining_run_id=RUNS["pretraining"],
                    clean_consumer_run_id=RUNS["consumers"],
                    stress_representation_run_id=RUNS["stress_representations"],
                    **common,
                )
            ),
        )
    if "mechanism" in stages:
        _record(
            state,
            "mechanism_representations",
            run_simulation_mechanism_representations(
                SimulationMechanismRepresentationConfig(
                    run_id=RUNS["mechanism_representations"],
                    pretraining_run_id=RUNS["pretraining"],
                    clean_representation_run_id=RUNS["representations"],
                    baseline_device=args.device,
                    chronaris_device=args.device,
                    require_valid_mask_match=False,
                    **common,
                )
            ),
        )
        _record(
            state,
            "mechanism_consumers",
            run_simulation_mechanism_consumers(
                SimulationMechanismConsumerConfig(
                    run_id=RUNS["mechanism_consumers"],
                    mechanism_representation_run_id=RUNS[
                        "mechanism_representations"
                    ],
                    stress_representation_run_id=RUNS["stress_representations"],
                    seeds=seeds,
                    resume=args.resume,
                )
            ),
        )
    state["completed_stages"] = sorted(
        set(state.get("completed_stages", ())) | stages
    )
    _write_state(state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("pretraining", "clean", "ablation", "stress", "mechanism"),
        default=("pretraining", "clean", "ablation", "stress", "mechanism"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 43))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--protocol-version",
        choices=("v3.2.1",),
        default="v3.2.1",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_state(args):
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
    ).strip()
    identity = {
        "protocol_version": args.protocol_version,
        "source_commit": commit,
        "runner_sha256": _sha256(Path(__file__)),
        "selected_models_sha256": _sha256(SELECTED),
        "protocol_sha256": _sha256(PROTOCOL),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "pretraining_source_commit": PRETRAINING_COMMIT,
        "outer_public_results_opened": False,
        "run_ids": RUNS,
    }
    path = STATE_ROOT / "run_state.json"
    if args.resume and path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise RuntimeError("simulation evaluation resume rejected frozen identity drift")
        return state
    return {
        "format": "chronaris.thesis_frozen_simulation.v3.2.1",
        **identity,
        "results": {},
    }


def _record(state, stage, result):
    state["results"][stage] = result if isinstance(result, dict) else asdict(result)
    _write_state(state)


def _verify_pretraining():
    compact = REPO / "docs/artifacts/runs" / RUNS["pretraining"]
    evidence = json.loads(
        (compact / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    with (compact / "locked_pretraining_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if evidence.get("status") != "completed" or len(rows) != 15:
        raise RuntimeError("frozen simulation pretraining evidence is incomplete")
    for row in rows:
        path = Path(row["checkpoint_path"])
        if not path.is_file() or _sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"frozen pretraining checkpoint changed: {path}")
    return {
        "run_id": RUNS["pretraining"],
        "status": "reused_completed",
        "source_commit": PRETRAINING_COMMIT,
        "checkpoint_count": len(rows),
        "acceptance_pass_count": evidence["acceptance_pass_count"],
        "acceptance_check_count": evidence["acceptance_check_count"],
        "g2_locked_test_opened": evidence["g2_locked_test_opened"],
        "task_targets_opened": evidence["task_targets_opened"],
    }


def _write_state(state):
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path = STATE_ROOT / "run_state.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
