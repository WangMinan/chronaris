#!/usr/bin/env python3
"""Export locked six-method representations across all G2 stress levels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.application_tasks import (  # noqa: E402
    SimulationStressRepresentationConfig,
    run_simulation_stress_representations,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-12_simulation-locked-stress-representations")
    parser.add_argument("--pretraining-run-id", default="2026-07-12_simulation-locked-pretraining")
    parser.add_argument("--baseline-pretraining-run-id")
    parser.add_argument("--locked-configuration-path")
    parser.add_argument("--confirmation-locked-configuration-path")
    parser.add_argument("--clean-representation-run-id", default="2026-07-12_simulation-locked-representations")
    parser.add_argument("--stress-generation-run-id", default="2026-07-12_aviation-simulation-locked-stress")
    parser.add_argument("--stress-audit-run-id", default="2026-07-12_aviation-simulation-locked-stress-audit")
    parser.add_argument("--sealed-manifest-path")
    parser.add_argument("--confirmation-access-path")
    parser.add_argument("--split-id", default="locked_test")
    parser.add_argument("--profile-prefix", default="locked_test_profile_")
    parser.add_argument("--export-batch-size", type=int, default=32)
    parser.add_argument("--baseline-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chronaris-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_simulation_stress_representations(
        SimulationStressRepresentationConfig(
            run_id=args.run_id,
            pretraining_run_id=args.pretraining_run_id,
            baseline_pretraining_run_id=args.baseline_pretraining_run_id,
            locked_configuration_path=args.locked_configuration_path,
            confirmation_locked_configuration_path=(
                args.confirmation_locked_configuration_path
            ),
            clean_representation_run_id=args.clean_representation_run_id,
            stress_generation_run_id=args.stress_generation_run_id,
            stress_audit_run_id=args.stress_audit_run_id,
            sealed_manifest_path=args.sealed_manifest_path,
            confirmation_access_path=args.confirmation_access_path,
            split_id=args.split_id,
            profile_prefix=args.profile_prefix,
            export_batch_size=args.export_batch_size,
            baseline_device=args.baseline_device,
            chronaris_device=args.chronaris_device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
