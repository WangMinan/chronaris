#!/usr/bin/env python3
"""Launch the isolated fusion workbench without changing editable installs."""
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "src"))


def main():
    import argparse
    import torch
    from chronaris.serving.fusion_workbench.server import DemoStore, make_server
    parser = argparse.ArgumentParser(description="融合特征工作台（本机CPU/GPU）")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "artifacts/fusion-workbench")
    args = parser.parse_args()
    root = args.data_root.resolve()
    if not root.is_relative_to(PROJECT) or root == PROJECT:
        parser.error("产物目录必须位于本 worktree 内")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    store = DemoStore(root / "cases", root / "runs", device=args.device,
        audit_root=PROJECT / "docs/artifacts/runs/2026-07-10_fixed-data-audit",
        context_path=PROJECT / "docs/artifacts/runs/2026-06-14_llm-preprocessing-context/llm_preprocessing_context.json")
    try:
        server = make_server(store, args.port)
        print(f"航空人机时序融合系统：http://127.0.0.1:{server.server_port}（{len(store.cases)} 份观测记录）", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == "__main__":
    main()
