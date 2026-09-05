"""Run task evaluation private proxy component ablation over real feature export assets."""

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

from chronaris.evidence.private_component_ablation import (  # noqa: E402
    StageIPrivateComponentAblationConfig,
    resolve_git_commit,
    run_task_eval_private_component_ablation,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-private-component-ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--e-run-manifest", required=True)
    parser.add_argument("--f-run-manifest", required=True)
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--report-root", default="docs/artifacts/runs")
    parser.add_argument("--target-variant-name", default="chronaris_opt")
    parser.add_argument("--lag-window-points", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_task_eval_private_component_ablation(
        StageIPrivateComponentAblationConfig(
            run_id=args.run_id,
            e_run_manifest_path=args.e_run_manifest,
            f_run_manifest_path=args.f_run_manifest,
            output_root=args.output_root,
            report_root=args.report_root,
            target_variant_name=args.target_variant_name,
            lag_window_points=args.lag_window_points,
            git_commit=resolve_git_commit(cwd=REPO_ROOT),
        )
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "table_path": result.table_path,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
