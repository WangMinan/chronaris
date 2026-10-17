"""Prepare or verify the final application configuration using the current interpreter."""
from run_fusion_workbench import PROJECT

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import torch
    from chronaris.evaluation.application_tasks.chapter6_acceptance import (
        prepare_application_acceptance, run_application_acceptance, capture_application_browser)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--browser-only', action='store_true')
    args = parser.parse_args()
    root, evidence = args.root.resolve(), args.evidence.resolve()
    if not root.is_relative_to(PROJECT/'artifacts') or not evidence.is_relative_to(PROJECT):
        parser.error('验收输出须位于当前工作区')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.checkpoint:
        prepare_application_acceptance(project=PROJECT, checkpoint=args.checkpoint, root=root)
    if not args.browser_only:
        run_application_acceptance(project=PROJECT, root=root, evidence=evidence)
    capture_application_browser(project=PROJECT, root=root, evidence=evidence)
