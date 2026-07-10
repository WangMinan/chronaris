"""Generate and audit smoke or formal aviation dual-stream simulation bundles."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.application_tasks import (  # noqa: E402
    SimulationAuditConfig,
    run_simulation_audit,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--heavy-run-id", default=None)
    parser.add_argument("--compact-run-id", default=None)
    parser.add_argument("--heavy-output-root", default="artifacts/application_evaluation")
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> int:
    configure_task_eval_cli_logging(sys.stderr)
    args = parse_args()
    heavy_default = (
        "2026-07-10_aviation-simulation-smoke"
        if args.mode == "smoke"
        else "2026-07-10_aviation-simulation-formal"
    )
    compact_default = (
        "2026-07-10_aviation-simulation-smoke-audit"
        if args.mode == "smoke"
        else "2026-07-10_aviation-simulation-audit"
    )
    result = run_simulation_audit(
        SimulationAuditConfig(
            mode=args.mode,
            heavy_run_id=args.heavy_run_id or heavy_default,
            compact_run_id=args.compact_run_id or compact_default,
            heavy_output_root=args.heavy_output_root,
            compact_output_root=args.compact_output_root,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
