"""Build the task evaluation optimized model summary package."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.optimized_model_summary import (  # noqa: E402
    StageIOptimizedModelSummaryConfig,
    run_task_eval_optimized_model_summary,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-optimized-model-summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--p30-root", default=None)
    parser.add_argument("--p31-root", default=None)
    parser.add_argument("--p32-root", default=None)
    parser.add_argument("--p34-root", default=None)
    parser.add_argument("--p35-root", default=None)
    parser.add_argument("--p36-root", default=None)
    parser.add_argument("--artifact-root", default="docs/artifacts/runs")
    parser.add_argument("--report-root", default="docs/artifacts/runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = {
        name: _resolve_path(value)
        for name, value in {
            "p30_root": args.p30_root,
            "p31_root": args.p31_root,
            "p32_root": args.p32_root,
            "p34_root": args.p34_root,
            "p35_root": args.p35_root,
            "p36_root": args.p36_root,
        }.items()
        if value
    }
    result = run_task_eval_optimized_model_summary(
        StageIOptimizedModelSummaryConfig(
            run_id=args.run_id,
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            **kwargs,
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else REPO_ROOT / path)


if __name__ == "__main__":
    raise SystemExit(main())
