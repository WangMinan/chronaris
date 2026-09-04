#!/usr/bin/env python3
"""Resume frozen simulation inference without retraining or opening real outer results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
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
PROTOCOL = REPO / "docs/requirements/thesis-frozen-paper-evaluation-v3.2.3.md"
STATE_ROOT = REPO / "docs/artifacts/runs/2026-09-04_thesis-simulation-v3p2p3"
PREVIOUS_STATE = REPO / "docs/artifacts/runs/2026-09-03_thesis-simulation-v3p2p1/run_state.json"
PRETRAINING_COMMIT = "58db4458dc64bb452526431413924283deff63bd"
RUNS = {
    "pretraining": "2026-09-03_thesis-simulation-pretraining-v3p2",
    "representations": "2026-09-03_thesis-simulation-representations-v3p2p1",
    "consumers": "2026-09-03_thesis-simulation-consumers-v3p2p1",
    "stress_representations": "2026-09-04_thesis-simulation-stress-representations-v3p2p3",
    "stress_consumers": "2026-09-04_thesis-simulation-stress-consumers-v3p2p3",
    "mechanism_representations": "2026-09-04_thesis-simulation-mechanism-representations-v3p2p3",
    "mechanism_consumers": "2026-09-04_thesis-simulation-mechanism-consumers-v3p2p3",
    "ablation_pretraining": "2026-09-03_thesis-simulation-ablation-pretraining-v3p2p1",
    "ablation_representations": "2026-09-03_thesis-simulation-ablation-representations-v3p2p1",
    "ablation_consumers": "2026-09-03_thesis-simulation-ablation-consumers-v3p2p1",
}


def main() -> int:
    args = _parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-facing simulation evaluation requires CUDA")
    if tuple(args.seeds) != (17, 29, 43) or not args.resume:
        raise RuntimeError("v3.2.3 requires all frozen seeds and resume protection")
    state = _load_state(args)
    stages = set(args.stages)
    seeds = tuple(args.seeds)
    common = {
        "selected_candidates_path": str(SELECTED),
        "seeds": seeds,
        "resume": args.resume,
    }
    _write_state(state)
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
        choices=("stress", "mechanism"),
        default=("stress", "mechanism"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 43))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--protocol-version",
        choices=("v3.2.3",),
        default="v3.2.3",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_state(args):
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts"),
        cwd=REPO, text=True,
    )
    if dirty:
        raise RuntimeError("simulation evaluation requires committed source")
    reused, lineage = _reuse_completed_stages()
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
        "outer_results_authorized": False,
        "reused_evidence_sha256": lineage,
        "run_ids": RUNS,
    }
    path = STATE_ROOT / "run_state.json"
    if args.resume and path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise RuntimeError("simulation evaluation resume rejected frozen identity drift")
        return state
    return {
        "format": "chronaris.thesis_frozen_simulation.v3.2.3",
        **identity,
        "results": reused,
    }


def _reuse_completed_stages():
    previous = json.loads(PREVIOUS_STATE.read_text(encoding="utf-8"))
    if previous["source_commit"] != "7937334239e2cdc0263754de03328769ab90987c":
        raise RuntimeError("frozen simulation source changed")
    stages = ("pretraining", "representations", "consumers", "ablation_pretraining",
              "ablation_representations", "ablation_consumers")
    reused = {}
    lineage = {str(PREVIOUS_STATE.relative_to(REPO)): _sha256(PREVIOUS_STATE)}
    for stage in stages:
        result = previous["results"][stage]
        root = REPO / "docs/artifacts/runs" / RUNS[stage]
        evidence = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
        if result["run_id"] != RUNS[stage] or evidence.get("status") != "completed":
            raise RuntimeError(f"frozen evidence is incomplete: {stage}")
        with (root / "acceptance.csv").open(encoding="utf-8", newline="") as handle:
            checks = list(csv.DictReader(handle))
        if not checks or any(row["passed"] != "True" for row in checks):
            raise RuntimeError(f"frozen acceptance is incomplete: {stage}")
        for path in sorted(root.glob("*")):
            if path.suffix in {".json", ".csv"}:
                lineage[str(path.relative_to(REPO))] = _sha256(path)
        table = {
            "pretraining": "locked_pretraining_results.csv",
            "ablation_pretraining": "ablation_pretraining_results.csv",
            "representations": "representation_inventory.csv",
            "ablation_representations": "representation_inventory.csv",
        }.get(stage)
        if table:
            with (root / table).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected = {"pretraining": 15, "ablation_pretraining": 12,
                        "representations": 54, "ablation_representations": 36}[stage]
            if len(rows) != expected:
                raise RuntimeError(f"frozen artifact coverage changed: {stage}")
            for row in rows:
                is_checkpoint = "checkpoint_path" in row
                path = REPO / (row["checkpoint_path"] if is_checkpoint
                               else str(Path(row["output_root"]) / "fusion_stream.npz"))
                expected_hash = row["checkpoint_sha256" if is_checkpoint else "representation_sha256"]
                if _sha256(path) != expected_hash:
                    raise RuntimeError(f"frozen artifact changed: {path}")
        reused[stage] = {**result, "status": "reused_completed"}
    return reused, lineage


def _record(state, stage, result):
    state["results"][stage] = result if isinstance(result, dict) else asdict(result)
    _write_state(state)


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
    with _periodic_training_heartbeat("frozen_simulation_inference", 30.0):
        raise SystemExit(main())
