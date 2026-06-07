#!/usr/bin/env python3
"""Build multi-view semantic event support from one Stage H run manifest."""

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

from chronaris.pipelines.causal_fusion import (  # noqa: E402
    StageGSemanticEventSupportConfig,
    build_stage_h_semantic_event_support,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-semantic-support")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--artifact-root", default="docs/artifacts/assets/stage_i_semantic_event_support")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--state-source", choices=("hidden_with_projection_fallback", "hidden", "projection"), default="hidden_with_projection_fallback")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_stage_h_semantic_event_support(
        args.run_manifest,
        config=StageGSemanticEventSupportConfig(
            state_source=args.state_source,
            device=args.device,
        ),
    )
    run_root = REPO_ROOT / args.artifact_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = REPO_ROOT / args.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "semantic_event_support_summary.json"
    report_path = report_root / f"stage-i-semantic-event-support-{args.run_id}.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(args.run_id, summary) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


def _render_report(run_id: str, summary: dict[str, object]) -> str:
    lines = [
        f"# Stage I Semantic Event Support - {run_id}",
        "",
        f"- run_manifest_path: `{summary['run_manifest_path']}`",
        f"- view_count: `{summary['view_count']}`",
        f"- query_names: `{summary['query_names']}`",
        f"- mean_event_token_count: `{summary['mean_event_token_count']:.6f}`",
        f"- mean_query_entropy: `{summary['mean_query_entropy']:.6f}`",
        f"- mean_top_event_attribution: `{summary['mean_top_event_attribution']:.6f}`",
        "",
        "| view | sortie | pilot | state_source | dominant query | mean event tokens | mean query entropy | mean top event attribution | top sample | top offset s |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary.get("view_rows", []):
        lines.append(
            f"| `{row['view_id']}` | `{row['sortie_id']}` | {row['pilot_id']} | `{row['state_source']}` | "
            f"`{row['dominant_query_name']}` | {float(row['mean_event_token_count']):.6f} | "
            f"{float(row['mean_query_entropy']):.6f} | {float(row['mean_top_event_attribution']):.6f} | "
            f"`{row['top_sample_id']}` | {float(row['top_sample_query_event_offset_s']):.6f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
