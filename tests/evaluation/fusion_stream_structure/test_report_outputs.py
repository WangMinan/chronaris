from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_synthetic_cli_writes_report_outputs(tmp_path: Path) -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
    script = repo_root / "scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py"
    run_id = "2026-07-07_fusion-stream-structure-test"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic",
            "--methods",
            "chronaris",
            "naive_time_sync",
            "--output-root",
            str(tmp_path),
            "--run-id",
            run_id,
            "--min-T",
            "30",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert run_id in result.stdout
    run_root = tmp_path / run_id
    metrics_path = run_root / "e3_metrics_long.csv"
    summary_path = run_root / "e3_summary.json"
    manifest_path = run_root / "evidence_manifest.json"
    report_path = run_root / "report.md"
    assert metrics_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()
    assert report_path.exists()
    assert list((run_root / "plots").glob("state_timeline_*.png"))
    assert list((run_root / "plots").glob("fragment_replay_*.png"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["training_invoked"] is False
    assert summary["confirmed_metrics_changed"] is False
    assert manifest["training_invoked"] is False
    assert manifest["confirmed_metrics_changed"] is False
    assert manifest["thesis_protocol_snapshot_modified"] is False
    assert "2026-07-03_thesis-protocol-snapshot" not in "\n".join(manifest["output_files"])
    external = manifest["external_libraries"]
    evaluator_counts = manifest.get("evaluator_status_counts", {})
    if external.get("claspy", {}).get("import_available") is True:
        assert evaluator_counts.get("clasp", {}).get("completed", 0) > 0
    if external.get("stumpy", {}).get("import_available") is True:
        assert evaluator_counts.get("stumpy", {}).get("completed", 0) > 0
