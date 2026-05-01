"""Prepare Stage I task manifests and feature tables for UAB or NASA CSM."""

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

from chronaris.dataset import build_nasa_csm_task_entries, build_uab_task_entries, dump_stage_i_summary, dump_stage_i_task_entries
from chronaris.features import build_nasa_csm_feature_table, build_uab_feature_table
from chronaris.pipelines.stage_i_phase3 import build_stage_i_dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset", choices=("uab", "nasa_csm"), required=True)
    parser.add_argument("--profile", choices=("session_v1", "window_v2"), default="window_v2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="docs/reports/assets/stage_i")
    return parser.parse_args()


def _default_run_id(dataset: str, profile: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-stage-i-{dataset}-{profile}"


def main() -> int:
    args = parse_args()
    run_id = args.run_id or _default_run_id(args.dataset, args.profile)
    output_dir = REPO_ROOT / args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "uab":
        prepared = build_uab_task_entries(args.dataset_root, profile=args.profile)
        feature_result = build_uab_feature_table(args.dataset_root, prepared.entries)
    elif args.profile != "window_v2":
        raise ValueError("NASA CSM currently supports only --profile window_v2.")
    else:
        prepared = build_nasa_csm_task_entries(args.dataset_root)
        feature_result = build_nasa_csm_feature_table(args.dataset_root, prepared.entries)

    dump_stage_i_task_entries(prepared.entries, path=output_dir / "task_manifest.jsonl")
    feature_result.feature_table.to_parquet(output_dir / "feature_table.parquet", index=False)
    (output_dir / "feature_schema.json").write_text(
        json.dumps(feature_result.feature_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = build_stage_i_dataset_summary(prepared.entries, feature_result)
    dump_stage_i_summary(summary, path=output_dir / "dataset_summary.json")

    print(
        json.dumps(
            {
                "artifact_root": str(output_dir),
                "dataset": args.dataset,
                "profile": args.profile,
                "entry_count": len(prepared.entries),
                "subset_counts": prepared.subset_counts,
                "feature_count": len(feature_result.feature_columns),
                "feature_group_counts": {
                    key: len(value) for key, value in feature_result.feature_group_columns.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
