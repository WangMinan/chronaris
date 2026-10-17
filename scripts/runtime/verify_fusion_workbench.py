#!/usr/bin/env python3
"""Run the bounded CPU/HTTP acceptance against the running local service."""
from run_fusion_workbench import PROJECT


def main():
    import argparse
    from pathlib import Path
    import torch
    from chronaris.serving.fusion_workbench.acceptance import verify_demo
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "artifacts/fusion-workbench")
    args = parser.parse_args()
    if not args.data_root.resolve().is_relative_to(PROJECT):
        parser.error("验收产物必须位于本 worktree")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(verify_demo(args.data_root, port=args.port))


if __name__ == "__main__":
    main()
