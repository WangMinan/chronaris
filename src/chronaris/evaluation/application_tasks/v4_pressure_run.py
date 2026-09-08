"""Eight development conditions evaluated with clean-fitted frozen consumers."""
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import f1_score

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import build_guarded_application_consumer_targets
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.application_frozen_evaluation import evaluate_frozen_application_consumers
from chronaris.evaluation.application_tasks.application_metrics import segmentation_metrics
from chronaris.evaluation.application_tasks.v4_development_conditions import DEVELOPMENT_CONDITIONS, load_development_condition
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_diagnostic_run import CURVE_UPDATES, _require_diagnostic_device
from chronaris.evaluation.application_tasks.v4_candidate_results import _completed_scores
from chronaris.evaluation.application_tasks.v4_encoding_diagnostics import collect_encoding_diagnostics
from chronaris.evaluation.application_tasks.v4_grouped_consumers import regression_metrics
from chronaris.modeling.training import TrainedFusionAdapter
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.representation import load_fusion_stream_batch, select_observation_batch, write_fusion_stream_batch
from chronaris.representation.oof_export import _concatenate_fusion_batches
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _load_development_encoder(checkpoint, *, route, fold, device):
    encoder, normalizer, _ = load_frozen_application_encoder(checkpoint, route=route, fold=fold, device=device,
                                                           allow_diagnostic_snapshot=True)
    return encoder, normalizer


def _paired_representation(encoder, normalizer, checkpoint, fold, batch, root, *, label_used_for_encoder_training=False):
    checkpoint_hash = sha256_file(checkpoint)
    if (root / "representation_manifest.json").exists():
        metadata = json.loads((root / "representation_manifest.json").read_text())
        if metadata["label_used_for_encoder_training"] is not label_used_for_encoder_training:
            raise ValueError("cached pressure representation label provenance changed")
        output = load_fusion_stream_batch(root)
        if (output.sample_ids != batch.sample_ids or output.source_sample_hashes != batch.source_sample_hashes
            or output.checkpoint_sha256 != checkpoint_hash or output.fold_id != fold.fold_id):
            raise ValueError("cached pressure representation provenance changed")
        return output
    adapter = TrainedFusionAdapter(encoder=encoder, normalizer=normalizer, fold_id=fold.fold_id, checkpoint_sha256=checkpoint_hash)
    output = _concatenate_fusion_batches([adapter(select_observation_batch(batch, batch.sample_ids[i:i + 4]))
                                         for i in range(0, len(batch.sample_ids), 4)])
    write_fusion_stream_batch(output, root=root, export_role="development_pressure_validation",
                             label_used_for_encoder_training=label_used_for_encoder_training)
    return output


def summarize_pressure_predictions(result, output, sample_manifest):
    """Retain all windows in tail summaries and resample-ready profile metrics."""
    group = {row["sample_id"]: row["profile_id"] for row in sample_manifest}
    empty = dict(zip(output.sample_ids, (~output.valid_mask.any(dim=1)).tolist(), strict=True))
    rows, tails = [], []
    for consumer in ("linear", "minirocket"):
        predictions = [row for row in result.workload_prediction_rows if row["consumer"] == consumer]
        if tuple(row["sample_id"] for row in predictions) != output.sample_ids:
            raise ValueError("pressure predictions changed the complete window list")
        scopes = [("all_windows", predictions)] + [(profile, [row for row in predictions if group[row["sample_id"]] == profile])
                                                   for profile in sorted(set(group.values()))]
        for scope, selected in scopes:
            ids = [row["sample_id"] for row in selected]
            metric = regression_metrics([row["future_workload_true"] for row in selected],
                [row["future_workload_pred"] for row in selected], sample_ids=ids, no_observation=[empty[sample] for sample in ids])
            tails.append({"consumer": consumer, "profile_id": scope, **metric})
            if scope != "all_windows":
                rows.extend(({"consumer": consumer, "profile_id": scope, "metric": "macro_f1",
                    "value": float(f1_score([row["workload_class_true"] for row in selected],
                        [row["workload_class_pred"] for row in selected], labels=(0, 1, 2), average="macro", zero_division=0)),
                    "task": "workload_classification"},
                    {"consumer": consumer, "profile_id": scope, "metric": "rmse", "value": metric["rmse"], "task": "workload_regression"}))
    with np.load(result.prediction_path, allow_pickle=False) as archive:
        ids = tuple(str(value) for value in archive["validation_sample_ids"])
        if ids != output.sample_ids:
            raise ValueError("segmentation prediction identities changed")
        profiles = np.array([group[sample] for sample in ids])
        for profile in np.unique(profiles):
            selected = profiles == profile
            for consumer in ("causal_tcn_raw", "causal_tcn_duration"):
                metrics = segmentation_metrics(archive["validation_state_true"][selected], archive[f"validation_{consumer}_state"][selected])
                rows.extend({"consumer": consumer, "profile_id": str(profile), "metric": metric, "value": float(value) if value is not None else None,
                    "task": "maneuver_segmentation"} for metric, (value, _) in metrics.items())
    return {"profile_metrics": rows, "regression_tails": tails, "independent_unit": "parameter_profile",
            "all_windows_retained": True, "sample_count": len(output.sample_ids), "no_observation_count": sum(empty.values())}


def _verify_clean_predictions(reference_path, actual_path):
    with np.load(reference_path, allow_pickle=False) as reference, np.load(actual_path, allow_pickle=False) as actual:
        for name in actual.files:
            if name not in reference.files:
                raise ValueError("clean consumer prediction archive keys changed")
            if np.issubdtype(actual[name].dtype, np.floating):
                same = np.allclose(reference[name], actual[name], atol=1e-6, rtol=0)
            else:
                same = np.array_equal(reference[name], actual[name])
            if not same:
                raise ValueError(f"clean fitted consumer predictions changed: {name}")


def run_development_pressure(*, method, route, update, output_root,
    diagnostic_root="artifacts/application_evaluation/2026-09-06_v4-learning-curves",
    condition_root="artifacts/application_evaluation/2026-09-06_v4-development-conditions-repair", candidate_name=None,
    device="cuda", phase="screen", seed=17):
    if phase not in ('screen','review') or seed not in ((17,) if phase=='screen' else (17,29,43)):
        raise ValueError('pressure requires an approved development phase and seed')
    if phase=='review' and candidate_name is None:
        raise ValueError('review pressure requires a selected candidate')
    if device == "cuda":
        _require_diagnostic_device(seed)
    elif device != "cpu":
        raise ValueError("pressure inference device must be cpu or cuda")
    from chronaris.evaluation.application_tasks.v4_candidates import candidate_options, EXPANDED_SIMULATION_ROOT
    options = candidate_options(method, candidate_name) if candidate_name is not None else None
    allowed_updates = ((1500 if phase=='review' else 300,) if route == "self_supervised" else
                       (500 if phase=='review' else 200,)) if options else CURVE_UPDATES
    if route not in {"self_supervised", "task_guided"} or update not in allowed_updates:
        raise ValueError("pressure diagnosis requires an approved learning-curve snapshot")
    torch.set_num_threads(1)
    curve = Path(diagnostic_root) / "simulation" / method
    if options:
        curve = curve / candidate_name
    if phase=='review':curve=curve/'review'/f'seed{seed}'
    training_state = json.loads((curve / "run_state.json").read_text())
    if training_state.get('phase','screen')!=phase or training_state['seed']!=seed:
        raise ValueError('pressure phase or seed differs from clean training')
    if phase=='review':
        plan=json.loads((Path(diagnostic_root)/'selection_plan.json').read_text())
        digest=plan.pop('plan_sha256')
        selected=[unit for unit in plan['units'] if unit['domain']=='simulation' and unit['method']==method
                  and unit['candidate_name']==candidate_name and unit['seed']==seed and route in unit['routes']]
        if (hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()!=digest or len(selected)!=1
            or plan['confirmation_feedback_used'] or plan['source_code_sha256']!=training_state['source_code_sha256']):
            raise ValueError('pressure unit is not in its frozen review selection')
        _completed_scores(curve,training_state,route,update,method,candidate_name,phase=phase,seed=seed)
    if json.dumps(training_state.get("candidate_options"), sort_keys=True) != json.dumps(options, sort_keys=True):
        raise ValueError("pressure candidate configuration differs from clean training")
    if f"{route}:{update}" not in training_state["completed_consumers"]:
        raise ValueError("clean consumer fitting must finish before pressure evaluation")
    training = training_state[route + "_training"]
    checkpoint = (Path(training["best_checkpoint_path"]) if options else Path(training["last_checkpoint_path"]).with_name(
        f"update_{update:06d}.pt" if route == "self_supervised" else f"joint_update_{update:06d}.pt"))
    clean_export = curve / "representations" / route / str(update)
    if route == "task_guided":
        clean_export = clean_export / method
    clean = load_fusion_stream_batch(clean_export / "validation")
    consumer_root = curve / "consumers" / route / str(update)
    consumer_manifest = json.loads((consumer_root / method / "consumer_manifest.json").read_text())
    if sha256_file(consumer_manifest["prediction_path"]) != consumer_manifest["prediction_sha256"]:
        raise ValueError("clean consumer prediction evidence changed")
    data_arguments = {"simulation_root": EXPANDED_SIMULATION_ROOT} if options else {}
    provider, _, fold, _, digest, _, _, data = load_development_inputs("simulation", None, None, **data_arguments)
    if options and ("__training512" not in fold.fold_id or digest != training_state["data_manifest_sha256"]):
        raise ValueError("candidate pressure requires the same expanded training manifest")
    if clean.sample_ids != fold.validation_sample_ids or clean.checkpoint_sha256 != sha256_file(checkpoint):
        raise ValueError("clean representation does not match the pressure checkpoint/validation fold")
    source = {"format": "chronaris.v4_development_pressure.v1", "source_code_sha256": v4_workflow_source_sha256(),
        "method": method, "route": route, "update": update, "checkpoint_sha256": sha256_file(checkpoint),
        "inference_device": device, "phase": phase, "seed": seed,
        "data_manifest_sha256": digest, "consumer_protocol_sha256": consumer_manifest["protocol_sha256"],
        "source_diagnostic_code_sha256": training_state["source_code_sha256"],
        "candidate_options": options,
        "clean_prediction_sha256": sha256_file(consumer_manifest["prediction_path"]),
        "consumer_files": {name: sha256_file(item["path"]) for name, item in consumer_manifest["model_files"].items()},
        "condition_audit_sha256": sha256_file(Path(condition_root) / "development_condition_audit.json"),
        "clean_export_sha256": sha256_file(clean_export / "validation/fusion_stream.npz"),
        "evaluation_role": "validation", "confirmation_opened": False, "consumer_refit": False}
    root = Path(output_root) / method / route / str(update)
    if options:
        root = Path(output_root) / method / candidate_name
        if phase=='review':root=root/'review'/f'seed{seed}'
        root=root / route / str(update)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"source": source, "conditions": {}}
    if json.dumps(state["source"], sort_keys=True) != json.dumps(source, sort_keys=True):
        raise ValueError("pressure sources/data/config changed; use a new run root")
    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(state_path)
    save()
    targets = build_guarded_application_consumer_targets(data,
        completed_pretraining_checkpoints=(training_state["self_supervised_training"]["best_checkpoint_path"],),
        task_guided_development=True, smoke_only=False)
    encoder, normalizer = _load_development_encoder(checkpoint, route=route, fold=fold, device=device)
    with _periodic_training_heartbeat(f"pressure_{method}_{route}", 30., root=root) as progress:
        try:
            for condition in DEVELOPMENT_CONDITIONS:
                progress["condition"] = condition
                observed = load_development_condition(condition=condition, output_root=condition_root,
                    registry_path="docs/requirements/thesis-v4-simulation-manifest.json")
                if observed.batch.sample_ids != fold.validation_sample_ids:
                    raise ValueError("pressure data escaped the fixed validation role")
                result_path = root / condition / "result.json"
                if condition in state["conditions"]:
                    saved = state["conditions"][condition]
                    if (sha256_file(result_path) != saved["result_sha256"]
                        or sha256_file(root / condition / "representation/fusion_stream.npz") != saved["representation_sha256"]
                        or sha256_file(saved["prediction_path"]) != saved["prediction_sha256"]):
                        raise ValueError("saved pressure result changed")
                    continue
                started = time.perf_counter()
                output = _paired_representation(encoder, normalizer, checkpoint, fold, observed.batch, root / condition / "representation",
                                                label_used_for_encoder_training=route == "task_guided")
                representation_s = time.perf_counter() - started
                clean_delta = None
                if condition == "clean_asynchronous":
                    clean_delta = float((clean.sequence_embedding - output.sequence_embedding).abs().max())
                    if not torch.equal(clean.valid_mask, output.valid_mask) or clean_delta > 1e-6:
                        raise ValueError("current execution does not reproduce the clean snapshot representation")
                result = evaluate_frozen_application_consumers(method_name=method, output=output, targets=targets,
                    model_root=consumer_root, output_root=root / condition / "predictions", fold_id=fold.fold_id,
                    evaluation_id=condition, seed=seed, evaluation_role="validation")
                if condition == "clean_asynchronous":
                    _verify_clean_predictions(consumer_manifest["prediction_path"], result.prediction_path)
                diagnostics = collect_encoding_diagnostics(encoder=encoder, normalizer=normalizer, batch=observed.batch)
                record = {"condition": condition, "route": route, "update": update, "seed": seed, "phase": phase,
                    "evidence_scope": "development_diagnostic",
                    "evaluation": asdict(result), "grouped": summarize_pressure_predictions(result, output, observed.sample_manifest_rows),
                    "encoding_diagnostics": diagnostics, "clean_representation_max_delta": clean_delta,
                    "representation_elapsed_s": representation_s, "elapsed_s": time.perf_counter() - started}
                result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                state["conditions"][condition] = {"result_path": str(result_path), "result_sha256": sha256_file(result_path),
                    "representation_sha256": sha256_file(root / condition / "representation/fusion_stream.npz"),
                    "prediction_path": result.prediction_path, "prediction_sha256": result.prediction_sha256}
                save()
            state["completed"] = len(state["conditions"]) == len(DEVELOPMENT_CONDITIONS)
            save()
        except Exception as exc:
            (root / "failure.json").write_text(json.dumps({"condition": progress.get("condition"), "error": str(exc),
                "traceback": traceback.format_exc(), "source": source}, ensure_ascii=False, indent=2) + "\n")
            raise
    return state
