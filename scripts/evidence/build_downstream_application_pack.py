#!/usr/bin/env python3
"""Build the locked Chinese downstream thesis evidence package."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.downstream_application_pack import (  # noqa: E402
    DownstreamEvidencePackConfig,
    run_downstream_evidence_pack,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-12_downstream-evidence-pack")
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument("--heavy-output-root", default="artifacts/application_evaluation")
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_downstream_evidence_pack(
        DownstreamEvidencePackConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
