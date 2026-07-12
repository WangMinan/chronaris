#!/usr/bin/env python3
"""Run task-independent Chronaris v1 distortion diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.representation_diagnostics.run import (  # noqa: E402
    V1DistortionDiagnosticConfig,
    run_v1_distortion_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-12_chronaris-v1-distortion-diagnostics",
    )
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--diagnostic-batch-size", type=int, default=8)
    args = parser.parse_args()
    root = run_v1_distortion_diagnostics(
        V1DistortionDiagnosticConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            device=args.device,
            diagnostic_batch_size=args.diagnostic_batch_size,
        )
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
