#!/usr/bin/env python3
"""Run the P21 Stage I LLM preprocessing comparison over existing assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.common.run_observer import configure_stage_i_cli_logging  # noqa: E402
from chronaris.pipelines.stage_i.llm.comparison import (  # noqa: E402
    DEFAULT_LLM_COMPARISON_ARTIFACT_ROOT,
    DEFAULT_LLM_COMPARISON_MIDTERM_ROOT,
    DEFAULT_LLM_COMPARISON_REPORT_ROOT,
    DEFAULT_LLM_COMPARISON_RUN_ID,
    DEFAULT_LLM_CONTEXT_PATH,
    StageILLMComparisonConfig,
    run_stage_i_llm_comparison,
)
from chronaris.pipelines.stage_i.llm.preprocessing import (  # noqa: E402
    DEFAULT_RUNTIME_CASE_TABLE_PATH,
    DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH,
    DEFAULT_SUPPORT_SUMMARY_PATH,
    DEFAULT_TASK_MANIFEST_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_LLM_COMPARISON_RUN_ID)
    parser.add_argument("--artifact-root", default=DEFAULT_LLM_COMPARISON_ARTIFACT_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_LLM_COMPARISON_REPORT_ROOT)
    parser.add_argument("--midterm-root", default=DEFAULT_LLM_COMPARISON_MIDTERM_ROOT)
    parser.add_argument("--llm-context-path", default=DEFAULT_LLM_CONTEXT_PATH)
    parser.add_argument("--task-manifest-path", default=DEFAULT_TASK_MANIFEST_PATH)
    parser.add_argument("--support-summary-path", default=DEFAULT_SUPPORT_SUMMARY_PATH)
    parser.add_argument("--runtime-schema-contract-path", default=DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH)
    parser.add_argument("--runtime-case-table-path", default=DEFAULT_RUNTIME_CASE_TABLE_PATH)
    parser.add_argument("--max-human-review-fields", type=int, default=6)
    parser.add_argument("--max-human-review-schema-gaps", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    configure_stage_i_cli_logging(sys.stderr)
    args = parse_args()
    result = run_stage_i_llm_comparison(
        StageILLMComparisonConfig(
            run_id=args.run_id,
            artifact_root=args.artifact_root,
            report_root=args.report_root,
            midterm_root=args.midterm_root,
            llm_context_path=args.llm_context_path,
            task_manifest_path=args.task_manifest_path,
            support_summary_path=args.support_summary_path,
            runtime_schema_contract_path=args.runtime_schema_contract_path,
            runtime_case_table_path=args.runtime_case_table_path,
            max_human_review_fields=args.max_human_review_fields,
            max_human_review_schema_gaps=args.max_human_review_schema_gaps,
        )
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "condition_manifest_path": result.condition_manifest_path,
                "report_path": result.report_path,
                "midterm_summary_path": result.midterm_summary_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
