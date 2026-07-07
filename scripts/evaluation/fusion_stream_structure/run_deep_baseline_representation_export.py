"""Train Dingxin MulT/ContiFormer OOF pooled embeddings for E3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_export import (  # noqa: E402
    DEFAULT_EXPORT_RUN_ID,
    DEFAULT_FOUR_METHOD_RUN_ID,
    DeepBaselineRepresentationExportConfig,
    build_four_method_e3_input_table,
    export_oof_pooled_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_EXPORT_RUN_ID)
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--e-run-manifest-path", default="docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json")
    parser.add_argument("--f-run-manifest-path", default="docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json")
    parser.add_argument("--models", nargs="+", default=("mult", "contiformer"))
    parser.add_argument("--task-name", default="T2_next_window_physiology_response")
    parser.add_argument("--task-type", default="regression")
    parser.add_argument("--split-strategy", default="leave_one_view_out")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    parser.add_argument("--tensor-cache", default="auto")
    parser.add_argument("--amp", default="bf16")
    parser.add_argument("--torch-compile", default="off")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    parser.add_argument("--run-four-method-e3", action="store_true")
    parser.add_argument("--four-method-run-id", default=DEFAULT_FOUR_METHOD_RUN_ID)
    parser.add_argument("--min-T", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_folds = 1 if args.smoke and args.max_folds is None else args.max_folds
    epochs = 1 if args.smoke and args.epochs == 20 else args.epochs
    config = DeepBaselineRepresentationExportConfig(
        run_id=args.run_id,
        output_root=args.output_root,
        e_run_manifest_path=args.e_run_manifest_path,
        f_run_manifest_path=args.f_run_manifest_path,
        models=tuple(args.models),
        task_name=args.task_name,
        task_type=args.task_type,
        split_strategy=args.split_strategy,
        seed=int(args.seed),
        epochs=int(epochs),
        batch_size=int(args.batch_size),
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        require_cuda=bool(args.require_cuda),
        tensor_cache=args.tensor_cache,
        amp=args.amp,
        torch_compile=args.torch_compile,
        max_folds=max_folds,
        resume=bool(args.resume),
        skip_completed=bool(args.skip_completed),
    )
    result = export_oof_pooled_embeddings(config)
    payload = {
        "run_id": result.run_id,
        "run_root": result.run_root,
        "status": result.status,
        "training_invoked": result.training_invoked,
        "confirmed_metrics_changed": result.confirmed_metrics_changed,
        "representation_table_path": result.representation_table_path,
        "checkpoint_manifest_path": result.checkpoint_manifest_path,
        "report_path": result.report_path,
        "evidence_manifest_path": result.evidence_manifest_path,
        "completed_fold_count": result.completed_fold_count,
        "expected_fold_count": result.expected_fold_count,
        "model_embedding_status": dict(result.model_embedding_status),
        "blockers_path": result.blockers_path,
    }
    if args.run_four_method_e3 and result.all_models_complete and result.representation_table_path:
        four_root = Path(args.output_root) / args.four_method_run_id
        four_root = four_root if four_root.is_absolute() else REPO_ROOT / four_root
        four_root.mkdir(parents=True, exist_ok=True)
        combined_path = four_root / "combined_four_method_e3_input_long.csv"
        build_four_method_e3_input_table(
            deep_representation_table_path=result.representation_table_path,
            e_run_manifest_path=args.e_run_manifest_path,
            f_run_manifest_path=args.f_run_manifest_path,
            output_path=combined_path,
        )
        command = [
            sys.executable,
            "scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py",
            "--fusion-stream-table",
            str(combined_path),
            "--methods",
            "chronaris",
            "naive_time_sync",
            "mult",
            "contiformer",
            "--output-root",
            args.output_root,
            "--run-id",
            args.four_method_run_id,
            "--min-T",
            str(args.min_T),
            "--source-training-run",
            result.run_root,
            "--representation-family",
            config.representation_family,
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        payload["four_method_e3_command"] = command
        payload["four_method_e3_returncode"] = completed.returncode
        payload["four_method_e3_stdout"] = completed.stdout
        payload["four_method_e3_stderr"] = completed.stderr
        if completed.returncode != 0:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return completed.returncode
    elif args.run_four_method_e3:
        payload["four_method_e3_skipped_reason"] = "deep_baseline_oof_embeddings_incomplete"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
