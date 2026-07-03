"""Build P38 Stage I thesis protocol freeze package."""

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

from chronaris.pipelines.stage_i.evidence.thesis_protocol import (  # noqa: E402
    StageIThesisProtocolConfig,
    build_stage_i_thesis_protocol,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-thesis-protocol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument(
        "--p30-root",
        default="docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1",
    )
    parser.add_argument(
        "--p31-root",
        default="docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1",
    )
    parser.add_argument(
        "--p32-root",
        default="docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1",
    )
    parser.add_argument(
        "--p34-root",
        default="docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20",
    )
    parser.add_argument(
        "--p35-root",
        default="docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20",
    )
    parser.add_argument(
        "--p36-root",
        default="docs/artifacts/assets/stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20",
    )
    parser.add_argument(
        "--p36-summary-root",
        default="docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20",
    )
    parser.add_argument(
        "--p37-root",
        default="docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1",
    )
    parser.add_argument("--artifact-root", default="docs/artifacts/assets/stage_i_thesis_protocol")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_stage_i_thesis_protocol(
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
