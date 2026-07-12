#!/usr/bin/env python3
"""Run the one-shot fail-closed Chronaris v2 promotion audit."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.application_tasks.chronaris_v2_promotion_run import (  # noqa: E402
    ChronarisV2PromotionConfig,
    run_chronaris_v2_promotion_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-13_chronaris-v2-promotion-audit")
    parser.add_argument("--dingxin-consumer-run-id", default="2026-07-13_chronaris-v2-dingxin-locked-consumers")
    parser.add_argument("--simulation-confirmation-consumer-run-id", default="2026-07-13_chronaris-v2-sealed-confirmation-consumers")
    parser.add_argument("--mechanism-consumer-run-id", default="2026-07-13_chronaris-v2-mechanism-consumers")
    args = parser.parse_args()
    result = run_chronaris_v2_promotion_audit(
        ChronarisV2PromotionConfig(
            run_id=args.run_id,
            dingxin_consumer_run_id=args.dingxin_consumer_run_id,
            simulation_confirmation_consumer_run_id=(
                args.simulation_confirmation_consumer_run_id
            ),
            mechanism_consumer_run_id=args.mechanism_consumer_run_id,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
