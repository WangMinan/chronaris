"""Read completed simulation candidates and rank the fixed seed-17 screen."""
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score, root_mean_squared_error

from chronaris.evaluation.application_tasks.v4_candidates import CANDIDATE_CHANGES, candidate_options
from chronaris.evaluation.application_tasks.v4_development_conditions import DEVELOPMENT_CONDITIONS
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


PRIMARY_METRICS = (
    ("simulated_future_workload_classification", "linear", "macro_f1", "higher"),
    ("simulated_future_workload_regression", "linear", "rmse", "lower"),
    ("simulated_maneuver_state_segmentation", "causal_tcn_duration", "frame_macro_f1", "higher"),
)


def _check_file(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"candidate evidence changed: {path}")


def _completed_scores(root, state, route, update, method, candidate):
    result_path = root / f"{route}_{update}_consumers.json"
    result = json.loads(result_path.read_text())
    manifest = result["model_manifest"]
    training = state[route + "_training"]
    if (state["self_supervised_training"]["optimizer_updates"] != 300
        or training["optimizer_updates"] != (300 if route == "self_supervised" else 250)):
        raise ValueError("initial screen update count differs from the frozen budget")
    if (manifest["consumer_fit_role"] != "train" or manifest["evaluation_roles"] != ["validation"]
        or manifest["label_used_for_encoder_training"] is not (route == "task_guided")
        or manifest["fold_id"] != state["fold"]["fold_id"] or manifest["method_name"] != method
        or set(manifest["model_files"]) != {"linear", "minirocket", "tcn"}):
        raise ValueError("candidate consumer method, role or label provenance changed")
    _check_file(manifest["prediction_path"], manifest["prediction_sha256"])
    for item in manifest["model_files"].values():
        _check_file(item["path"], item["sha256"])
    checkpoint_hash = sha256_file(state[route + "_training"]["best_checkpoint_path"])
    for role in ("train", "validation"):
        directory = root / "representations" / route / str(update)
        if route == "task_guided":
            directory /= method
        directory /= role
        exported = json.loads((directory / "representation_manifest.json").read_text())
        if (exported["checkpoint_sha256"] != checkpoint_hash or exported["method_name"] != method
            or exported["fold_id"] != state["fold"]["fold_id"]
            or tuple(exported["sample_ids"]) != tuple(state["fold"][role + "_sample_ids"])
            or exported["label_used_for_encoder_training"] is not (route == "task_guided")):
            raise ValueError("candidate representation checkpoint or data role changed")
        _check_file(directory / exported["representation_archive"], exported["representation_sha256"])
    with np.load(manifest["prediction_path"], allow_pickle=False) as prediction:
        if tuple(prediction["validation_sample_ids"].tolist()) != tuple(state["fold"]["validation_sample_ids"]):
            raise ValueError("candidate predictions changed the complete validation window list")
        scores = (
            float(f1_score(prediction["validation_workload_class_true"], prediction["validation_linear_class"],
                           labels=(0, 1, 2), average="macro", zero_division=0)),
            float(root_mean_squared_error(prediction["validation_workload_true"].astype(float), prediction["validation_linear_regression"].astype(float))),
            float(f1_score(prediction["validation_state_true"].ravel(), prediction["validation_causal_tcn_duration_state"].ravel(),
                           average="macro", zero_division=0)),
        )
    rows = result["metric_rows"]
    if any(row["role"] != "validation" or row["seed"] != 17 or row["smoke_only"]
           or row["method"] != method or row["fold"] != state["fold"]["fold_id"] for row in rows):
        raise ValueError("screen rows contain confirmation, other seeds or engineering scores")
    for (task, consumer, metric, direction), value in zip(PRIMARY_METRICS, scores, strict=True):
        matches = [row for row in rows if (row["task"], row["consumer"], row["metric"]) == (task, consumer, metric)]
        if (len(matches) != 1 or matches[0]["status"] != "available" or matches[0]["direction"] != direction
            or not math.isfinite(value) or not math.isclose(float(matches[0]["value"]), value, abs_tol=1e-10, rel_tol=1e-10)):
            raise ValueError("reported primary score differs from the frozen predictions")
    return {"candidate": candidate, "scores": list(scores), "metric_rows": rows,
        "pretraining_updates": state["self_supervised_training"]["optimizer_updates"],
        "pretraining_checkpoint_sha256": sha256_file(state["self_supervised_training"]["best_checkpoint_path"]),
        "pretraining_shared_between_routes": True,
        "supervised_updates": training["optimizer_updates"] if route == "task_guided" else 0,
        "consumer_neural_updates": len(result.get("tcn_training_rows", ())),
        "encoder_parameters": state["self_supervised_training"]["parameter_count"],
        "training_elapsed_s": state["self_supervised_training"]["training_elapsed_s"]
            + (state["task_guided_training"]["training_elapsed_s"] if route == "task_guided" else 0.),
        "consumer_prediction_sha256": manifest["prediction_sha256"], "result_sha256": sha256_file(result_path)}


def _pressure_p95(root, state, record, method, candidate, route, update):
    path = Path(root) / method / candidate / route / str(update) / "run_state.json"
    if not path.exists():
        return None
    pressure = json.loads(path.read_text())
    if not pressure.get("completed"):
        return None
    source = pressure["source"]
    if (source["evaluation_role"] != "validation" or source["confirmation_opened"] or source["consumer_refit"]
        or source.get("inference_device", "cuda") != "cuda"
        or source["data_manifest_sha256"] != state["data_manifest_sha256"]
        or source["clean_prediction_sha256"] != record["consumer_prediction_sha256"]
        or source["checkpoint_sha256"] != sha256_file(state[route + "_training"]["best_checkpoint_path"])
        or json.dumps(source["candidate_options"], sort_keys=True) != json.dumps(state["candidate_options"], sort_keys=True)
        or set(pressure["conditions"]) != set(DEVELOPMENT_CONDITIONS)):
        raise ValueError("candidate pressure roles, data, device or clean consumer changed")
    for condition in pressure["conditions"].values():
        _check_file(condition["result_path"], condition["result_sha256"])
        _check_file(condition["prediction_path"], condition["prediction_sha256"])
        _check_file(Path(condition["result_path"]).parent / "representation/fusion_stream.npz", condition["representation_sha256"])
    gap = json.loads(Path(pressure["conditions"]["contiguous_gap_30s"]["result_path"]).read_text())
    tails = [row for row in gap["grouped"]["regression_tails"] if row["consumer"] == "linear" and row["profile_id"] == "all_windows"]
    if len(tails) != 1 or not gap["grouped"]["all_windows_retained"]:
        raise ValueError("candidate missingness tail does not preserve every window")
    p95 = float(tails[0]["p95_absolute_error"])
    if not math.isfinite(p95) or p95 < 0:
        raise ValueError("invalid candidate long-tail error")
    return p95


def collect_simulation_screen(*, diagnostic_root, pressure_root, route, method="chronaris"):
    if route not in {"self_supervised", "task_guided"}:
        raise ValueError("unknown representation route")
    root = Path(diagnostic_root)
    cohort = json.loads((root / "cohort_state.json").read_text())
    pressure_queue_path = Path(pressure_root) / "queue_state.json"
    pressure_queue = json.loads(pressure_queue_path.read_text()) if pressure_queue_path.exists() else {}
    names = list(CANDIDATE_CHANGES) if method == "chronaris" else [name for item, name in cohort["units"] if item == method]
    if not names or len(names) != len(set(names)) or "reference" not in names:
        raise ValueError("screen requires a fixed candidate cohort and its repaired reference")
    update = 300 if route == "self_supervised" else 200
    completed, pending, failed = [], [], []
    data_hash = None
    for name in names:
        unit = root / "simulation" / method / name
        path = unit / "run_state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        if f"{route}:{update}" not in state.get("completed_consumers", ()):
            (failed if f"{method}/{name}" in cohort["failed_units"] else pending).append(name)
            continue
        options = candidate_options(method, name)
        if (json.dumps(state["candidate_options"], sort_keys=True) != json.dumps(options, sort_keys=True)
            or state["source_code_sha256"] != cohort["source_code_sha256"]
            or state["seed"] != 17 or state["confirmation_opened"]
            or "__training512" not in state["fold"]["fold_id"]):
            raise ValueError("screen candidate source, configuration or data role changed")
        if data_hash is not None and state["data_manifest_sha256"] != data_hash:
            raise ValueError("candidates did not share the expanded data manifest")
        data_hash = state["data_manifest_sha256"]
        record = _completed_scores(unit, state, route, update, method, name)
        record["missingness_p95"] = _pressure_p95(pressure_root, state, record, method, name, route, update)
        completed.append(record)
    pressure_pending = [row["candidate"] for row in completed if row["missingness_p95"] is None]
    pressure_failed = [name for name in pressure_pending
                       if f"{method}/{name}/{route}" in pressure_queue.get("failed_units", ())]
    pressure_pending = [name for name in pressure_pending if name not in pressure_failed]
    summary = {"format": "chronaris.v4_simulation_screen_summary.v1", "collector_source_sha256": sha256_file(__file__),
        "cohort_source_code_sha256": cohort["source_code_sha256"],
        "method": method, "route": route, "seed": 17, "primary_metrics": PRIMARY_METRICS,
        "completed": completed, "pending": pending, "failed": failed, "pressure_pending": pressure_pending,
        "pressure_failed": pressure_failed,
        "data_manifest_sha256": data_hash, "rankings": [], "advance_to_public_development": [],
        "reference_comparator": "reference", "adoption_decision": "requires_three_seed_review",
        "confirmation_feedback_used": False}
    # Queue failures are execution evidence, not a validated scientific exclusion.
    # Inspect their logs before deciding whether to repair or reject a candidate.
    if failed or pressure_failed or cohort.get("status") == "failed" or pressure_queue.get("status") == "failed":
        summary["failure_evidence"] = {
            "cohort_state_path": str(root / "cohort_state.json"),
            "cohort_state_sha256": sha256_file(root / "cohort_state.json"),
            "pressure_queue_path": str(pressure_queue_path) if pressure_queue else None,
            "pressure_queue_sha256": sha256_file(pressure_queue_path) if pressure_queue else None,
            "reference_unavailable": "reference" in failed or "reference" in pressure_failed,
        }
        return summary | {"status": "blocked_by_execution_failure"}
    if pending or not any(row["candidate"] == "reference" for row in completed):
        return summary | {"status": "waiting_for_complete_clean_cohort"}
    ranks = np.column_stack([rankdata([row["scores"][i] * (-1 if metric[-1] == "higher" else 1) for row in completed],
                                     method="average") for i, metric in enumerate(PRIMARY_METRICS)])
    rankings = [{"candidate": row["candidate"], "mean_task_rank": mean(ranks[i]),
                 "encoder_parameters": row["encoder_parameters"], "training_elapsed_s": row["training_elapsed_s"]}
                for i, row in enumerate(completed)]
    rankings.sort(key=lambda row: (row["mean_task_rank"], row["encoder_parameters"], row["training_elapsed_s"], row["candidate"]))
    summary["rankings"] = rankings
    if pressure_pending:
        return summary | {"status": "waiting_for_complete_pressure_cohort"}
    reference = next(row["missingness_p95"] for row in completed if row["candidate"] == "reference")
    summary["single_seed_tail_guard"] = {row["candidate"]: row["missingness_p95"] <= 1.2 * reference for row in completed}
    summary["advance_to_public_development"] = [row["candidate"] for row in rankings[:3]]
    return summary | {"status": "development_screen_complete_not_final_adoption"}
