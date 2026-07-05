"""Build task evaluation public model-comparison tables, figures, and report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.public_datasets.pipelines.model_comparison import (  # noqa: E402
    StageIPublicModelComparisonConfig,
    build_task_eval_public_model_comparison,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-public-model-comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument(
        "--artifact-root",
        default="docs/artifacts/runs",
    )
    parser.add_argument("--report-root", default="docs/artifacts/runs")
    parser.add_argument("--refresh-summary", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_task_eval_public_model_comparison(
        StageIPublicModelComparisonConfig(
            run_id=args.run_id,
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            refresh_summary_path=(
                _resolve_path(args.refresh_summary) if args.refresh_summary else None
            ),
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else REPO_ROOT / path)


if __name__ == "__main__":
    raise SystemExit(main())
