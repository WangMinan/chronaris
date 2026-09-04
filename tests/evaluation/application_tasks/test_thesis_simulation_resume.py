import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _runner():
    path = Path(__file__).resolve().parents[3] / "scripts/research/run_thesis_frozen_simulation.py"
    spec = importlib.util.spec_from_file_location("frozen_simulation_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_reuses_frozen_stages_and_detects_changed_artifact(tmp_path, monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    previous = tmp_path / "previous.json"
    monkeypatch.setattr(runner, "PREVIOUS_STATE", previous)
    state = {"source_commit": "7937334239e2cdc0263754de03328769ab90987c", "results": {}}
    tables = {
        "pretraining": ("locked_pretraining_results.csv", 15),
        "ablation_pretraining": ("ablation_pretraining_results.csv", 12),
        "representations": ("representation_inventory.csv", 54),
        "ablation_representations": ("representation_inventory.csv", 36),
        "consumers": (None, 0), "ablation_consumers": (None, 0),
    }
    for stage, (table, count) in tables.items():
        run_id = runner.RUNS[stage]
        root = tmp_path / "docs/artifacts/runs" / run_id
        root.mkdir(parents=True)
        (root / "evidence_manifest.json").write_text(json.dumps({"status": "completed"}))
        (root / "acceptance.csv").write_text("check_id,passed\ncomplete,True\n")
        state["results"][stage] = {"run_id": run_id, "status": "completed"}
        if not table:
            continue
        rows = []
        for index in range(count):
            artifact = tmp_path / stage / str(index) / "fusion_stream.npz"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(str(index).encode())
            if "pretraining" in stage:
                rows.append({"checkpoint_path": str(artifact), "checkpoint_sha256": runner._sha256(artifact)})
            else:
                rows.append({"output_root": str(artifact.parent), "representation_sha256": runner._sha256(artifact)})
        with (root / table).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    previous.write_text(json.dumps(state))
    reused, lineage = runner._reuse_completed_stages()
    assert set(reused) == set(tables)
    assert all(row["status"] == "reused_completed" for row in reused.values())
    assert lineage
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        runner._reuse_completed_stages()


def test_resume_refuses_seed_subset_before_loading_data(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "_parse_args", lambda: SimpleNamespace(device="cuda", seeds=[17], resume=True))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="all frozen seeds"):
        runner.main()
