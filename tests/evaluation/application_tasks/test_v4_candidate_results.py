import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, root_mean_squared_error

from chronaris.evaluation.application_tasks import v4_candidate_results as results
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _cohort(root, names):
    (root / "cohort_state.json").write_text(json.dumps({"source_code_sha256": "a" * 64,
        "units": [["physiology_only", name] for name in names], "failed_units": []}))


def _candidate(root, name, *, improved=False):
    unit = root / "simulation/physiology_only" / name
    unit.mkdir(parents=True)
    ids = ["v1", "v2", "v3"]
    truth = np.array([0, 1, 2])
    prediction = truth if improved else np.array([0, 0, 2])
    workload = np.array([.1, .5, .9], dtype=np.float64)
    regression = workload + (.01 if improved else .1)
    state_true = np.tile(truth, (3, 1))
    state_pred = state_true if improved else np.zeros((3, 3), dtype=int)
    np.savez(unit / "predictions.npz", validation_sample_ids=ids, validation_workload_class_true=truth,
        validation_linear_class=prediction, validation_workload_true=workload, validation_linear_regression=regression,
        validation_state_true=state_true, validation_causal_tcn_duration_state=state_pred)
    checkpoint = unit / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    fold = {"fold_id": "test__training512", "train_sample_ids": ["t1"], "validation_sample_ids": ids}
    for role in ("train", "validation"):
        directory = unit / "representations/self_supervised/300" / role
        directory.mkdir(parents=True)
        (directory / "fusion_stream.npz").write_bytes(b"representation fixture")
        (directory / "representation_manifest.json").write_text(json.dumps({"checkpoint_sha256": sha256_file(checkpoint),
            "method_name": "physiology_only", "fold_id": fold["fold_id"], "sample_ids": fold[role + "_sample_ids"],
            "label_used_for_encoder_training": False, "representation_archive": "fusion_stream.npz",
            "representation_sha256": sha256_file(directory / "fusion_stream.npz")}))
    state = {"candidate_options": candidate_options("physiology_only", name), "source_code_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64, "seed": 17, "confirmation_opened": False, "fold": fold,
        "completed_consumers": ["self_supervised:300"], "self_supervised_training": {
            "parameter_count": 100 if name == "reference" else 200, "optimizer_updates": 300,
            "training_elapsed_s": 10., "best_checkpoint_path": str(checkpoint)}}
    (unit / "run_state.json").write_text(json.dumps(state))
    scores = [f1_score(truth, prediction, labels=(0, 1, 2), average="macro"), root_mean_squared_error(workload, regression),
              f1_score(state_true.ravel(), state_pred.ravel(), average="macro")]
    rows = [{"task": task, "consumer": consumer, "metric": metric, "direction": direction, "value": float(value),
             "status": "available", "role": "validation", "seed": 17, "smoke_only": False,
             "method": "physiology_only", "fold": fold["fold_id"]}
            for (task, consumer, metric, direction), value in zip(results.PRIMARY_METRICS, scores, strict=True)]
    output = {"metric_rows": rows, "model_manifest": {"consumer_fit_role": "train", "evaluation_roles": ["validation"],
        "label_used_for_encoder_training": False, "method_name": "physiology_only", "fold_id": fold["fold_id"],
        "prediction_path": str(unit / "predictions.npz"), "prediction_sha256": sha256_file(unit / "predictions.npz"),
        "model_files": {name: {"path": str(checkpoint), "sha256": sha256_file(checkpoint)} for name in ("linear", "minirocket", "tcn")}}}
    (unit / "self_supervised_300_consumers.json").write_text(json.dumps(output))
    return unit


def test_screen_waits_for_every_candidate_and_pressure_before_advancing(tmp_path, monkeypatch):
    _cohort(tmp_path, ("reference", "capacity64"))
    _candidate(tmp_path, "reference")
    kwargs = dict(diagnostic_root=tmp_path, pressure_root=tmp_path / "pressure", route="self_supervised", method="physiology_only")
    partial = results.collect_simulation_screen(**kwargs)
    assert partial["rankings"] == partial["advance_to_public_development"] == []
    _candidate(tmp_path, "capacity64", improved=True)
    clean = results.collect_simulation_screen(**kwargs)
    assert clean["rankings"][0]["candidate"] == "capacity64"
    assert clean["advance_to_public_development"] == []
    monkeypatch.setattr(results, "_pressure_p95", lambda root, state, record, method, candidate, route, update: 1. if candidate == "reference" else .9)
    ready = results.collect_simulation_screen(**kwargs)
    assert ready["advance_to_public_development"] == ["capacity64", "reference"]
    assert ready["adoption_decision"] == "requires_three_seed_review"


def test_screen_recomputes_scores_and_rejects_roles_or_modified_evidence(tmp_path):
    _cohort(tmp_path, ("reference",))
    unit = _candidate(tmp_path, "reference")
    kwargs = dict(diagnostic_root=tmp_path, pressure_root=tmp_path / "pressure", route="self_supervised", method="physiology_only")
    path = unit / "self_supervised_300_consumers.json"
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    changed["metric_rows"][0]["value"] = 1.
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from"):
        results.collect_simulation_screen(**kwargs)
    changed["metric_rows"][0]["role"] = "held_out"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="confirmation"):
        results.collect_simulation_screen(**kwargs)
    path.write_text(json.dumps(original))
    (unit / "predictions.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        results.collect_simulation_screen(**kwargs)


@pytest.mark.parametrize("name", ["reference", "capacity64"])
@pytest.mark.parametrize("stage", ["clean", "pressure"])
def test_failed_units_are_terminal_and_never_silently_excluded(tmp_path, name, stage):
    _cohort(tmp_path, ("reference", "capacity64"))
    for candidate in ("reference", "capacity64"):
        if stage != "clean" or candidate != name:
            _candidate(tmp_path, candidate)
    pressure = tmp_path / "pressure"
    pressure.mkdir()
    if stage == "clean":
        path = tmp_path / "cohort_state.json"
        state = json.loads(path.read_text())
        state["failed_units"] = [f"physiology_only/{name}"]
        path.write_text(json.dumps(state))
    else:
        (pressure / "queue_state.json").write_text(json.dumps({
            "failed_units": [f"physiology_only/{name}/self_supervised"]}))
    result = results.collect_simulation_screen(diagnostic_root=tmp_path, pressure_root=pressure,
                                               route="self_supervised", method="physiology_only")
    assert result["status"] == "blocked_by_execution_failure"
    assert result["rankings"] == result["advance_to_public_development"] == []
    assert result["failure_evidence"]["reference_unavailable"] is (name == "reference")
    assert result["failed" if stage == "clean" else "pressure_failed"] == [name]
    assert name not in result["pressure_pending"]


def test_pressure_queue_failure_is_distinct_from_other_route_failure(tmp_path):
    _cohort(tmp_path, ("reference",))
    _candidate(tmp_path, "reference")
    pressure = tmp_path / "pressure"
    pressure.mkdir()
    path = pressure / "queue_state.json"
    path.write_text(json.dumps({"failed_units": ["physiology_only/reference/task_guided"]}))
    kwargs = dict(diagnostic_root=tmp_path, pressure_root=pressure,
                  route="self_supervised", method="physiology_only")
    assert results.collect_simulation_screen(**kwargs)["status"] == "waiting_for_complete_pressure_cohort"
    path.write_text(json.dumps({"status": "failed", "error": "CUDA verification failed"}))
    result = results.collect_simulation_screen(**kwargs)
    assert result["status"] == "blocked_by_execution_failure"
    assert result["failure_evidence"]["pressure_queue_sha256"] == sha256_file(path)
