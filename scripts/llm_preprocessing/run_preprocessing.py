"""Run task evaluation LLM preprocessing on existing MySQL/Influx-derived assets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.llm import resolve_llm_provider  # noqa: E402
from chronaris.llm_preprocessing.preprocessing import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_LIVE_SWEEP_SUMMARY_PATH,
    DEFAULT_MULTITASK_SUMMARY_PATH,
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUNTIME_CASE_TABLE_PATH,
    DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH,
    DEFAULT_RUNTIME_SERVICE_SUMMARY_PATH,
    DEFAULT_SUPPORT_SUMMARY_PATH,
    DEFAULT_TASK_MANIFEST_PATH,
    LLM_PREPROCESSING_MODES,
    StageILLMPreprocessingConfig,
    run_task_eval_llm_preprocessing,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-llm-preprocessing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--mode", choices=LLM_PREPROCESSING_MODES, default="build-context")
    parser.add_argument("--provider", default=os.environ.get("CHRONARIS_LLM_PROVIDER", "deepseek"))
    parser.add_argument("--model", default=os.environ.get("CHRONARIS_LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--multitask-summary-path", default=DEFAULT_MULTITASK_SUMMARY_PATH)
    parser.add_argument("--task-manifest-path", default=DEFAULT_TASK_MANIFEST_PATH)
    parser.add_argument("--live-sweep-summary-path", default=DEFAULT_LIVE_SWEEP_SUMMARY_PATH)
    parser.add_argument("--support-summary-path", default=DEFAULT_SUPPORT_SUMMARY_PATH)
    parser.add_argument("--runtime-schema-contract-path", default=DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH)
    parser.add_argument("--runtime-service-summary-path", default=DEFAULT_RUNTIME_SERVICE_SUMMARY_PATH)
    parser.add_argument("--runtime-case-table-path", default=DEFAULT_RUNTIME_CASE_TABLE_PATH)
    parser.add_argument("--max-schema-fields", type=int, default=36)
    parser.add_argument("--max-window-cards", type=int, default=12)
    parser.add_argument("--max-runtime-cases", type=int, default=12)
    parser.add_argument("--schema-field-chunk-size", type=int, default=24)
    parser.add_argument("--weak-label-task-chunk-size", type=int, default=3)
    parser.add_argument("--schema-gap-group-chunk-size", type=int, default=8)
    parser.add_argument("--runtime-case-chunk-size", type=int, default=4)
    parser.add_argument("--deepseek-base-url", default=os.environ.get("CHRONARIS_LLM_BASE_URL"))
    parser.add_argument("--secrets-path", default=str(REPO_ROOT / "docs" / "SECRETS.md"))
    parser.add_argument("--no-secrets-file", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if args.provider.lower() == "deepseek" and not api_key and not args.no_secrets_file:
        api_key = _resolve_deepseek_api_key(args.secrets_path)
    provider = resolve_llm_provider(
        provider_name=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=args.deepseek_base_url,
    )
    result = run_task_eval_llm_preprocessing(
        StageILLMPreprocessingConfig(
            run_id=args.run_id,
            mode=args.mode,
            artifact_root=args.artifact_root,
            report_root=args.report_root,
            provider_name=args.provider,
            model=args.model,
            multitask_summary_path=args.multitask_summary_path,
            task_manifest_path=args.task_manifest_path,
            live_sweep_summary_path=args.live_sweep_summary_path,
            support_summary_path=args.support_summary_path,
            runtime_schema_contract_path=args.runtime_schema_contract_path,
            runtime_service_summary_path=args.runtime_service_summary_path,
            runtime_case_table_path=args.runtime_case_table_path,
            max_schema_fields=args.max_schema_fields,
            max_window_cards=args.max_window_cards,
            max_runtime_cases=args.max_runtime_cases,
            schema_field_chunk_size=args.schema_field_chunk_size,
            weak_label_task_chunk_size=args.weak_label_task_chunk_size,
            schema_gap_group_chunk_size=args.schema_gap_group_chunk_size,
            runtime_case_chunk_size=args.runtime_case_chunk_size,
        ),
        provider=provider,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "artifact_root": result.artifact_root,
                "context_path": result.context_path,
                "summary_path": result.summary_path,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _resolve_deepseek_api_key(secrets_path: str) -> str | None:
    path = Path(secrets_path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    for key in (
        "DEEPSEEK_API_KEY",
        "deepseek.api.key",
        "deepseek.api_key",
        "deepseek.api-key",
        "deepseek.token",
    ):
        matched = re.search(rf"^\+?\s*{re.escape(key)}\s*[:=]\s*(.+)$", text, flags=re.MULTILINE)
        if matched:
            return matched.group(1).strip()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
