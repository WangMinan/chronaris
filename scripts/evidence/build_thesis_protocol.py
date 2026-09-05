"""Build P38 task evaluation thesis protocol freeze package."""

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

from chronaris.evidence.thesis_protocol import (  # noqa: E402
    StageIThesisProtocolConfig,
    build_task_eval_thesis_protocol,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-thesis-protocol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument(
        "--p30-root",
        default="docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison",
    )
    parser.add_argument(
        "--p31-root",
        default="docs/artifacts/runs/2026-07-02_public-fusion-ablation",
    )
    parser.add_argument(
        "--p32-root",
        default="docs/artifacts/runs/2026-07-02_cross-evidence-matrix",
    )
    parser.add_argument(
        "--p34-root",
        default="docs/artifacts/runs/2026-07-02_task-head-calibration",
    )
    parser.add_argument(
        "--p35-root",
        default="docs/artifacts/runs/2026-07-02_stream-role-fusion",
    )
    parser.add_argument(
        "--p36-root",
        default="docs/artifacts/runs/2026-07-02_selected-model-reevaluation",
    )
    parser.add_argument(
        "--p36-summary-root",
        default="docs/artifacts/runs/2026-07-02_selected-model-summary",
    )
    parser.add_argument(
        "--p37-root",
        default="docs/artifacts/runs/2026-07-02_metric-calibration",
    )
    parser.add_argument("--artifact-root", default="docs/artifacts/runs")
    parser.add_argument("--report-root", default="docs/artifacts/runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_task_eval_thesis_protocol(
        StageIThesisProtocolConfig(
            run_id=args.run_id,
            p30_root=_resolve_path(args.p30_root),
            p31_root=_resolve_path(args.p31_root),
            p32_root=_resolve_path(args.p32_root),
            p34_root=_resolve_path(args.p34_root),
            p35_root=_resolve_path(args.p35_root),
            p36_root=_resolve_path(args.p36_root),
            p36_summary_root=_resolve_path(args.p36_summary_root),
            p37_root=_resolve_path(args.p37_root),
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else REPO_ROOT / path)


if __name__ == "__main__":
    raise SystemExit(main())
