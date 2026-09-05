"""Run the unified task evaluation evidence closure manifest builder."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.closure_runner import (  # noqa: E402
    StageIEvidenceRunnerConfig,
    run_task_eval_evidence_closure,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-evidence-closure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument(
        "--only",
        action="append",
        choices=("multitask", "rigid_body", "semantic", "runtime", "private_proxy", "public_adapter", "rotation", "all"),
        default=[],
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--test-summary", default="not_run_by_runner")
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--report-root", default="docs/artifacts/runs")
    parser.add_argument("--e-run-manifest", default="docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json")
    parser.add_argument("--f-run-manifest", default="docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_task_eval_evidence_closure(
        StageIEvidenceRunnerConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            report_root=args.report_root,
            only=tuple(args.only) or ("all",),
            reuse_existing=args.reuse_existing,
            skip_heavy=args.skip_heavy,
            test_summary=args.test_summary,
            e_run_manifest_path=args.e_run_manifest,
            f_run_manifest_path=args.f_run_manifest,
        )
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "manifest_path": result.manifest_path,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
