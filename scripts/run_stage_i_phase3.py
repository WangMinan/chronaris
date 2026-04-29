"""Run the full Stage I Phase 3 closure workflow."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i_phase3 import StageIPhase3Config, render_stage_i_phase3_report, run_stage_i_phase3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="artifacts/stage_i")
    parser.add_argument(
        "--prior-uab-session-artifact-root",
        default="artifacts/stage_i/20260429T000000Z-stage-i-phase0-1-uab",
    )
    parser.add_argument(
        "--report-path",
        default=f"docs/reports/stage-i-closure-{datetime.now().date().isoformat()}.md",
    )
    parser.add_argument(
        "--planning-path",
        default=f"docs/planning/stage-i-closure-{datetime.now().date().isoformat()}.md",
    )
    return parser.parse_args()


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-phase3-closure")


def main() -> int:
    args = parse_args()
    config = StageIPhase3Config(
        dataset_root=args.dataset_root,
        output_root=str(REPO_ROOT / args.output_root),
        run_id=args.run_id or _default_run_id(),
        prior_uab_session_artifact_root=str(REPO_ROOT / args.prior_uab_session_artifact_root),
    )
    result = run_stage_i_phase3(config)

    report_path = REPO_ROOT / args.report_path
    report_path.write_text(render_stage_i_phase3_report(result.closure_summary) + "\n", encoding="utf-8")

    planning_path = REPO_ROOT / args.planning_path
    planning_path.write_text(_render_closure_gate_note(result) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "closure_summary_path": result.closure_summary_path,
                "closure_report_path": result.closure_report_path,
                "docs_report_path": str(report_path),
                "planning_path": str(planning_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _render_closure_gate_note(result) -> str:
    summary = result.closure_summary
    return "\n".join(
        [
            "# Stage I Closure Gate",
            "",
            f"- 生成时间：`{summary['generated_at_utc']}`",
            f"- Phase 3 机器产物根目录：`{summary['artifact_root']}`",
            "- Gate 判定：`PASS`",
            "- 已完成 UAB window-level workload 主线、NASA CSM attention-state 主线、UAB session-vs-window 对比。",
            "- 详细指标见主报告与 `closure_summary.json`。",
            "",
            "## 机器产物",
            "",
            f"- UAB：`{summary['uab_window']['artifact_root']}`",
            f"- NASA：`{summary['nasa_attention']['artifact_root']}`",
            f"- closure summary：`{result.closure_summary_path}`",
            f"- closure report：`{result.closure_report_path}`",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
