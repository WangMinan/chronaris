#!/usr/bin/env python3
"""Prepare an independent bounded case; shared assets are read only."""
from chapter6_demo import PROJECT


def main():
    import argparse
    from pathlib import Path
    from chronaris.serving.chapter6.prepare import prepare_dingxin_case
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=PROJECT / "artifacts/chapter6-demo/cases/dingxin-validation")
    args = parser.parse_args()
    if not args.destination.resolve().is_relative_to(PROJECT):
        parser.error("案例包必须写入本 worktree")
    case = prepare_dingxin_case(checkpoint=args.checkpoint, snapshot_root=args.snapshot_root,
        audit_root=PROJECT / "docs/artifacts/runs/2026-07-10_fixed-data-audit", destination=args.destination)
    print(case.root)


if __name__ == "__main__":
    main()
