"""Run E3 fusion stream structure evaluation without model training."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.fusion_stream_structure.clasp_segmentation import run_clasp_for_streams  # noqa: E402
from chronaris.evaluation.fusion_stream_structure.contracts import FusionStreamRunConfig, normalize_method_name  # noqa: E402
from chronaris.evaluation.fusion_stream_structure.dataset_loader import (  # noqa: E402
    FusionStreamDataset,
    build_fusion_streams_from_dingxin_artifacts,
    load_fusion_stream_long_table,
    records_from_validated_frame,
)
from chronaris.evaluation.fusion_stream_structure.metrics import compute_metric_rows  # noqa: E402
from chronaris.evaluation.fusion_stream_structure.preprocessing import preprocess_streams  # noqa: E402
from chronaris.evaluation.fusion_stream_structure.reports import (  # noqa: E402
    plot_fragment_replay,
    plot_state_timeline,
    plot_transition_graph,
    write_e3_long_table,
    write_e3_summary,
    write_evidence_manifest,
    write_markdown_report,
)
from chronaris.evaluation.fusion_stream_structure.stumpy_motif_discord import run_stumpy_for_streams  # noqa: E402

DEFAULT_E_MANIFEST = REPO_ROOT / "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
DEFAULT_F_MANIFEST = REPO_ROOT / "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-stream-table")
    parser.add_argument("--e-run-manifest-path", default=str(DEFAULT_E_MANIFEST))
    parser.add_argument("--f-run-manifest-path", default=str(DEFAULT_F_MANIFEST))
    parser.add_argument("--e-run-manifest", dest="e_run_manifest_path", default=argparse.SUPPRESS)
    parser.add_argument("--f-run-manifest", dest="f_run_manifest_path", default=argparse.SUPPRESS)
    parser.add_argument("--methods", nargs="+", default=("chronaris", "naive_time_sync", "mult", "contiformer"))
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--min-T", dest="min_T", type=int, default=30)
    parser.add_argument("--m-grid", default="auto")
    parser.add_argument("--tol", default="auto")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--small-dingxin-dry-run", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--skip-clasp", action="store_true")
    parser.add_argument("--skip-stumpy", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = _resolve_path(args.output_root) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    config = FusionStreamRunConfig(
        min_T=args.min_T,
        tol=_parse_tol(args.tol),
        m_grid=_parse_m_grid(args.m_grid),
        methods=tuple(normalize_method_name(method) for method in args.methods),
    )
    dataset, input_table_path = _load_dataset(args, run_root)
    input_table_rel = str(input_table_path.relative_to(REPO_ROOT)) if input_table_path and input_table_path.is_relative_to(REPO_ROOT) else str(input_table_path) if input_table_path else None

    preprocessed = preprocess_streams(dataset.records, config=config)
    clasp_results = (
        _skipped_results(preprocessed, "clasp_skipped")
        if args.skip_clasp
        else run_clasp_for_streams(preprocessed)
    )
    stumpy_results = (
        _skipped_results(preprocessed, "stumpy_skipped")
        if args.skip_stumpy
        else run_stumpy_for_streams(preprocessed, m_grid=config.m_grid)
    )
    metric_rows = compute_metric_rows(dataset.records, clasp_results, stumpy_results, config=config)

    metrics_path = write_e3_long_table(metric_rows, run_root / "e3_metrics_long.csv")
    plots = _write_plots(run_root, dataset.records, clasp_results, stumpy_results)
    external_libraries = _external_library_status()
    summary_path = write_e3_summary(
        metric_rows,
        run_root / "e3_summary.json",
        extra={
            "run_id": args.run_id,
            "available_methods": sorted({key[0] for key in dataset.records}),
            "unavailable_methods": list(dataset.unavailable_methods),
            "external_libraries": external_libraries,
            "input_long_table": input_table_rel,
            "plot_count": len(plots),
        },
    )
    output_files = [
        str(metrics_path.relative_to(run_root)),
        str(summary_path.relative_to(run_root)),
        *[str(path.relative_to(run_root)) for path in plots],
    ]
    manifest = {
        "run_id": args.run_id,
        "run_type": "fusion_stream_structure_evaluation",
        "training_invoked": False,
        "metrics_changed": False,
        "confirmed_metrics_changed": False,
        "thesis_protocol_snapshot_modified": False,
        "t3_artifact_deleted": False,
        "uses_stage_or_final_naming": False,
        "input_long_table": input_table_rel,
        "dataset_manifest": dataset.manifest,
        "available_methods": sorted({key[0] for key in dataset.records}),
        "unavailable_methods": list(dataset.unavailable_methods),
        "external_libraries": external_libraries,
        "preprocessing": {
            _key_to_text(key): stream.manifest
            for key, stream in preprocessed.items()
        },
        "clasp_status": {
            _key_to_text(key): result.get("status")
            for key, result in clasp_results.items()
        },
        "stumpy_status": {
            _key_to_text(key): result.get("status")
            for key, result in stumpy_results.items()
        },
        "output_files": output_files,
        "notes": [
            "E3 is an unsupervised structure diagnostic.",
            "E3 does not replace classification/regression task evidence.",
            "Historical retrieval artifacts are not deleted.",
            "Composite scores use fixed manifest weights and do not imply a winner.",
        ],
    }
    manifest_path = write_evidence_manifest(manifest, run_root / "evidence_manifest.json")
    output_files.append(str(manifest_path.relative_to(run_root)))
    report_path = write_markdown_report(
        run_root / "report.md",
        run_id=args.run_id,
        summary=json.loads(summary_path.read_text(encoding="utf-8")),
        manifest=manifest,
        output_files=output_files,
    )
    output_files.append(str(report_path.relative_to(run_root)))
    # Re-write manifest after report path is known.
    manifest["output_files"] = output_files
    write_evidence_manifest(manifest, manifest_path)
    print(json.dumps({
        "run_id": args.run_id,
        "run_root": str(run_root),
        "summary_path": str(summary_path),
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "training_invoked": False,
        "confirmed_metrics_changed": False,
    }, ensure_ascii=False, indent=2))
    return 0


def _load_dataset(args: argparse.Namespace, run_root: Path) -> tuple[FusionStreamDataset, Path | None]:
    if args.synthetic:
        frame = _synthetic_long_table(args.methods)
        input_path = run_root / "e3_input_long_table.csv"
        frame.to_csv(input_path, index=False)
        validation_dataset = load_fusion_stream_long_table(input_path)
        manifest = {
            **validation_dataset.manifest,
            "source_type": "synthetic",
            "generator": "deterministic_multivariate_stream_with_boundaries_motif_discord",
        }
        return FusionStreamDataset(
            records=validation_dataset.records,
            manifest=manifest,
            unavailable_methods=validation_dataset.unavailable_methods,
        ), input_path
    if args.fusion_stream_table:
        return load_fusion_stream_long_table(_resolve_path(args.fusion_stream_table)), Path(args.fusion_stream_table)
    if args.small_dingxin_dry_run:
        return build_fusion_streams_from_dingxin_artifacts(
            e_run_manifest_path=_resolve_path(args.e_run_manifest_path),
            f_run_manifest_path=_resolve_path(args.f_run_manifest_path),
            methods=args.methods,
            max_groups=args.max_groups or 2,
        ), None
    raise SystemExit("Specify --synthetic, --fusion-stream-table, or --small-dingxin-dry-run.")


def _synthetic_long_table(methods: Sequence[str], *, T: int = 72, d: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(20260707)
    rows = []
    labels = ["low"] * 24 + ["medium"] * 24 + ["high"] * (T - 48)
    boundaries = {24, 48}
    for method_index, method in enumerate(methods):
        normalized_method = normalize_method_name(method)
        for view_index, view_id in enumerate(("view_alpha", "view_beta")):
            phase = view_index * 0.15 + method_index * 0.03
            base = np.sin(np.linspace(0, 6 * np.pi, T) + phase)
            repeated = np.sin(np.linspace(0, np.pi, 8))
            values = np.zeros((T, d), dtype=float)
            for feature_index in range(d):
                values[:, feature_index] = (
                    base * (1.0 + 0.08 * feature_index)
                    + 0.12 * np.cos(np.linspace(0, 3 * np.pi, T) + feature_index)
                    + rng.normal(0.0, 0.03, size=T)
                )
            values[8:16, 0] += repeated
            values[36:44, 0] += repeated
            values[55:61, :] += 2.5 + 0.2 * method_index
            if normalized_method == "naive_time_sync":
                values += rng.normal(0.0, 0.08, size=values.shape)
            for time_index in range(T):
                row = {
                    "method_name": normalized_method,
                    "sortie_id": "synthetic_sortie_001",
                    "view_id": view_id,
                    "window_id": f"{view_id}_{time_index:03d}",
                    "time": time_index,
                    "maneuver_proxy_label": labels[time_index],
                    "weak_event_boundary": time_index in boundaries,
                    "physio_fluctuation_interval": 55 <= time_index < 61,
                    "pilot_id": 10000 + view_index,
                    "sample_partition": "synthetic",
                }
                for feature_index in range(d):
                    row[f"fusion_feature_{feature_index + 1}"] = float(values[time_index, feature_index])
                rows.append(row)
    return pd.DataFrame(rows)


def _write_plots(
    run_root: Path,
    records: Mapping[tuple[str, str, str], object],
    clasp_results: Mapping[tuple[str, str, str], Mapping[str, object]],
    stumpy_results: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> list[Path]:
    plot_root = run_root / "plots"
    plot_paths = []
    for index, (key, record) in enumerate(sorted(records.items())):
        if index >= 8:
            break
        safe = _safe_name("_".join(key))
        plot_paths.append(plot_state_timeline(record, clasp_results.get(key, {}), plot_root / f"state_timeline_{safe}.png"))
        plot_paths.append(plot_transition_graph(record, clasp_results.get(key, {}), plot_root / f"transition_graph_{safe}.png"))
        plot_paths.append(plot_fragment_replay(record, stumpy_results.get(key, {}), plot_root / f"fragment_replay_{safe}.png"))
    if not plot_paths:
        placeholder = plot_root / "state_timeline_no_available_streams.png"
        plot_root.mkdir(parents=True, exist_ok=True)
        placeholder.write_text("No available E3 streams.\n", encoding="utf-8")
        plot_paths.append(placeholder)
    return plot_paths


def _skipped_results(streams: Mapping[tuple[str, str, str], object], status: str) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        key: {
            "method_name": key[0],
            "sortie_id": key[1],
            "view_id": key[2],
            "status": status,
        }
        for key in streams
    }


def _external_library_status() -> dict[str, dict[str, object]]:
    status = {}
    for name in ("claspy", "stumpy"):
        try:
            module = importlib.import_module(name)
            status[name] = {
                "status": "available",
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except Exception as exc:
            status[name] = {
                "status": "unavailable",
                "error": repr(exc),
            }
    return status


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_fusion-stream-structure-execution")


def _parse_tol(value: str) -> int | str:
    return "auto" if value == "auto" else int(value)


def _parse_m_grid(value: str) -> str | tuple[int, ...]:
    if value == "auto":
        return "auto"
    return tuple(int(item) for item in re.split(r"[,\\s]+", value.strip()) if item)


def _key_to_text(key: tuple[str, str, str]) -> str:
    return "::".join(key)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
