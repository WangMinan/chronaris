"""Run the Stage I leakage-safe private proxy ablation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.private.leakage_safe_ablation import (  # noqa: E402
    StageILeakageSafeAblationConfig,
    resolve_git_commit,
    run_stage_i_leakage_safe_ablation,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-leakage-safe-ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--e-run-manifest", required=True)
    parser.add_argument("--f-run-manifest", required=True)
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_private_leakage_safe_ablation")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--target-variant-name", default="chronaris_opt")
    parser.add_argument("--lag-window-points", type=int, default=3)
    parser.add_argument("--residual-mode", default="raw_window_stats")
    parser.add_argument("--seeds", default="17,29,43,71,97")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_stage_i_leakage_safe_ablation(
        StageILeakageSafeAblationConfig(
            run_id=args.run_id,
            e_run_manifest_path=args.e_run_manifest,
            f_run_manifest_path=args.f_run_manifest,
            output_root=args.output_root,
            report_root=args.report_root,
            target_variant_name=args.target_variant_name,
            lag_window_points=args.lag_window_points,
            residual_mode=args.residual_mode,
            seeds=seeds,
            git_commit=resolve_git_commit(cwd=REPO_ROOT),
        )
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "report_path": result.report_path,
                "label_feature_audit_json_path": result.label_feature_audit_json_path,
                "seed_metrics_path": result.seed_metrics_path,
                "model_backbone_csv_path": result.model_backbone_csv_path,
                "task_adapter_csv_path": result.task_adapter_csv_path,
                "model_backbone_figure_path": result.model_backbone_figure_path,
                "task_adapter_figure_path": result.task_adapter_figure_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
