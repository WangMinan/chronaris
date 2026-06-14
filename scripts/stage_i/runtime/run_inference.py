#!/usr/bin/env python3
"""Run checkpoint-backed Stage I runtime inference on serialized sample windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.serving import (  # noqa: E402
    StageIRuntimeInferenceConfig,
    load_runtime_samples_jsonl,
    run_stage_i_runtime_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--artifact-root", default="docs/artifacts/assets/stage_i_runtime_inference")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-predictions-csv", action="store_true")
    parser.add_argument("--emit-jsonl", action="store_true")
    parser.add_argument("--semantic-event-top-k", type=int, default=4)
    parser.add_argument("--semantic-event-score-quantile", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--strict-feature-schema", action="store_true")
    parser.add_argument("--replay-mode", choices=("batch", "incremental", "both"), default="batch")
    args = parser.parse_args()

    samples = load_runtime_samples_jsonl(args.sample_jsonl)
    result = run_stage_i_runtime_inference(
        StageIRuntimeInferenceConfig(
            run_id=args.run_id,
            checkpoint_path=args.checkpoint_path,
            artifact_root=args.artifact_root,
            report_root=args.report_root,
            device=args.device,
            export_predictions_csv=not args.disable_predictions_csv,
            emit_predictions_jsonl=args.emit_jsonl,
            semantic_event_top_k=args.semantic_event_top_k,
            semantic_event_score_quantile=args.semantic_event_score_quantile,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            strict_feature_schema=args.strict_feature_schema,
            replay_mode=args.replay_mode,
        ),
        samples=samples,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "report_path": result.report_path,
                "predictions_csv_path": result.predictions_csv_path,
                "predictions_jsonl_path": result.predictions_jsonl_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
