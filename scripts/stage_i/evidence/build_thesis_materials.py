#!/usr/bin/env python3
"""Build Stage I thesis tables and explanatory figures from stable artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.evidence.thesis_materials import (  # noqa: E402
    StageIThesisMaterialsConfig,
    run_stage_i_thesis_materials,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default="docs/artifacts/assets/stage_i_thesis_figures")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--evidence-manifest-path")
    parser.add_argument("--proxy-sweep-summary-path")
    parser.add_argument("--live-sweep-summary-path")
    parser.add_argument("--live-partial-summary-path")
    parser.add_argument("--private-component-summary-path")
    parser.add_argument("--public-calibration-summary-path")
    parser.add_argument("--public-transfer-summary-path")
    parser.add_argument("--rigid-body-summary-path")
    parser.add_argument("--rotation-audit-summary-path")
    parser.add_argument("--runtime-summary-path")
    parser.add_argument("--runtime-service-summary-path")
    parser.add_argument("--runtime-schema-contract-path")
    parser.add_argument("--runtime-case-table-path")
    parser.add_argument("--support-summary-path")
    parser.add_argument("--semantic-event-summary-path")
    parser.add_argument("--llm-preprocessing-summary-path")
    parser.add_argument("--llm-comparison-summary-path")
    args = parser.parse_args()

    base_config = StageIThesisMaterialsConfig(run_id=args.run_id)
    config = StageIThesisMaterialsConfig(
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        report_root=args.report_root,
        evidence_manifest_path=args.evidence_manifest_path or base_config.evidence_manifest_path,
        proxy_sweep_summary_path=args.proxy_sweep_summary_path or base_config.proxy_sweep_summary_path,
        live_sweep_summary_path=args.live_sweep_summary_path or base_config.live_sweep_summary_path,
        live_partial_summary_path=args.live_partial_summary_path or base_config.live_partial_summary_path,
        private_component_summary_path=args.private_component_summary_path or base_config.private_component_summary_path,
        public_calibration_summary_path=args.public_calibration_summary_path or base_config.public_calibration_summary_path,
        public_transfer_summary_path=args.public_transfer_summary_path or base_config.public_transfer_summary_path,
        rigid_body_summary_path=args.rigid_body_summary_path or base_config.rigid_body_summary_path,
        rotation_audit_summary_path=args.rotation_audit_summary_path or base_config.rotation_audit_summary_path,
        runtime_summary_path=args.runtime_summary_path or base_config.runtime_summary_path,
        runtime_service_summary_path=args.runtime_service_summary_path or base_config.runtime_service_summary_path,
        runtime_schema_contract_path=args.runtime_schema_contract_path or base_config.runtime_schema_contract_path,
        runtime_case_table_path=args.runtime_case_table_path or base_config.runtime_case_table_path,
        support_summary_path=args.support_summary_path or base_config.support_summary_path,
        semantic_event_summary_path=args.semantic_event_summary_path or base_config.semantic_event_summary_path,
        llm_preprocessing_summary_path=args.llm_preprocessing_summary_path or base_config.llm_preprocessing_summary_path,
        llm_comparison_summary_path=args.llm_comparison_summary_path or base_config.llm_comparison_summary_path,
    )
    result = run_stage_i_thesis_materials(config)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
