#!/usr/bin/env python3
"""Run Stage I runtime/service smoke replay on one view JSONL input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.serving.runtime_service_smoke import (  # noqa: E402
    StageIRuntimeSmokeConfig,
    run_stage_i_runtime_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--artifact-root", default="docs/artifacts/assets/stage_i_runtime_service")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replay-mode", choices=("batch", "incremental", "both"), default="both")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--strict-feature-schema", action="store_true")
    parser.add_argument("--no-export-canonical-payload", action="store_true")
    args = parser.parse_args()

    result = run_stage_i_runtime_smoke(
        StageIRuntimeSmokeConfig(
            run_id=args.run_id,
            checkpoint_path=args.checkpoint_path,
            sample_jsonl_path=args.sample_jsonl,
            artifact_root=args.artifact_root,
            report_root=args.report_root,
            device=args.device,
            replay_mode=args.replay_mode,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            strict_feature_schema=args.strict_feature_schema,
            export_canonical_payload=not args.no_export_canonical_payload,
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
