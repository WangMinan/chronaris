"""Build P32 Stage I private/public cross-evidence matrix."""

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

from chronaris.pipelines.stage_i.evidence.cross_evidence_matrix import (  # noqa: E402
    StageICrossEvidenceMatrixConfig,
    build_stage_i_cross_evidence_matrix,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-cross-evidence-matrix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--private-thirdparty-root", required=True)
    parser.add_argument("--public-ablation-root", required=True)
    parser.add_argument(
        "--private-ablation-root",
        default="docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2",
    )
    parser.add_argument(
        "--public-model-comparison-root",
        default="docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1",
    )
    parser.add_argument(
        "--public-fusion-refresh-root",
        default="docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1",
    )
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_cross_evidence_matrix")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_stage_i_cross_evidence_matrix(
        StageICrossEvidenceMatrixConfig(
            run_id=args.run_id,
            private_thirdparty_root=_resolve_path(args.private_thirdparty_root),
            public_ablation_root=_resolve_path(args.public_ablation_root),
            private_ablation_root=_resolve_path(args.private_ablation_root),
            public_model_comparison_root=_resolve_path(args.public_model_comparison_root),
            public_fusion_refresh_root=_resolve_path(args.public_fusion_refresh_root),
            output_root=_resolve_path(args.output_root),
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
