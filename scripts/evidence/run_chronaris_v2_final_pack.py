#!/usr/bin/env python3
"""Build the final Chronaris v2 locked gap-evidence package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evidence.chronaris_v2_final_pack import (  # noqa: E402
    ChronarisV2FinalPackConfig,
    run_chronaris_v2_final_pack,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-13_chronaris-v2-final-evidence-pack")
    args = parser.parse_args()
    print(run_chronaris_v2_final_pack(ChronarisV2FinalPackConfig(run_id=args.run_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
