#!/usr/bin/env python3
"""Generate and seal the independent Chronaris v2 confirmation family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.chronaris_v2_confirmation_family import (  # noqa: E402
    ChronarisV2ConfirmationFamilyConfig,
    prepare_chronaris_v2_confirmation_family,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    config = (
        ChronarisV2ConfirmationFamilyConfig(
            heavy_run_id="2026-07-12_chronaris-v2-confirmation-family-smoke",
            compact_run_id="2026-07-12_chronaris-v2-confirmation-family-seal-smoke",
            profile_count=1,
            trajectories_per_profile=1,
            full_stress_scenarios=False,
            resume=args.resume,
        )
        if args.smoke
        else ChronarisV2ConfirmationFamilyConfig(resume=args.resume)
    )
    print(prepare_chronaris_v2_confirmation_family(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
