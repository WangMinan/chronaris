"""Prepare task evaluation deep-baseline sequence assets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.public_datasets.pipelines.sequence_preparation import (  # noqa: E402
    StageISequencePreparationConfig,
    run_task_eval_sequence_preparation,
)
from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("feature_export_case", "uab", "uab_workload_dataset", "nasa", "nasa_csm"),
    )
    parser.add_argument("--dataset-root", default="/home/wangminan/dataset/chronaris")
    parser.add_argument(
        "--feature-export-run-manifest",
        default="docs/artifacts/runs/2026-04-27_feature-export-closure/run_manifest.json",
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--target-steps", type=int, default=64)
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="output directory containing task_manifest/sequence_bundle/schema/summary",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    configure_task_eval_cli_logging(sys.stdout)
    args = parse_args()
    result = run_task_eval_sequence_preparation(
        StageISequencePreparationConfig(
            dataset_id=args.dataset,
            dataset_root=str(REPO_ROOT / args.dataset_root)
            if not Path(args.dataset_root).is_absolute()
            else args.dataset_root,
            feature_export_run_manifest_path=str(REPO_ROOT / args.feature_export_run_manifest)
            if not Path(args.feature_export_run_manifest).is_absolute()
            else args.feature_export_run_manifest,
            artifact_root=str(REPO_ROOT / args.artifact_root)
            if not Path(args.artifact_root).is_absolute()
            else args.artifact_root,
            profile=args.profile or (
                "real_sortie_v1" if args.dataset == "feature_export_case" else "window_v2"
            ),
            target_steps=args.target_steps,
        ),
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
