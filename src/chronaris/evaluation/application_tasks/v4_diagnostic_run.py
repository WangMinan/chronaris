"""Full-development learning curves using the shared trainers and consumers."""
from dataclasses import asdict, replace
from pathlib import Path
import gc
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context

import torch

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import build_guarded_application_consumer_targets
from chronaris.evaluation.application_tasks.application_consumer_runtime import ApplicationConsumerProtocol, run_application_method_consumers, prepare_application_cpu_consumers
from chronaris.evaluation.application_tasks.application_consumers import LinearConsumerConfig, MiniRocketConsumerConfig, TCNConsumerConfig
from chronaris.evaluation.application_tasks.application_finetuning import EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method
from chronaris.evaluation.application_tasks.application_finetuning_export import export_finetuned_application_representations, export_loaded_application_encoder
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, development_normalization, v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context, run_native_method_consumers
from chronaris.modeling.training import CandidateScreenConfig, EncoderCandidateConfig, load_common_pretraining_checkpoint, train_pretext_candidate
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.representation import AugmentationPolicy
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


CURVE_UPDATES = (50, 200, 500)


def _prepare_cpu_consumer_artifacts(**kwargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    started = time.time()
    _, status, elapsed, protocol_hash = prepare_application_cpu_consumers(**kwargs)
    return {"pid": os.getpid(), "started_unix_s": started, "finished_unix_s": time.time(),
            "component_status": status, "component_elapsed_s": elapsed, "protocol_sha256": protocol_hash}


def _require_diagnostic_device(seed):
    if seed not in (17, 29, 43) or not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise ValueError("v4 experiments require RTX 4090 and an approved seed")


def _snapshot_outputs(*, encoder, normalizer, checkpoint, provider, fold, root):
    return export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
        provider=provider, fold=fold, root=root, export_roles=("train", "validation"), export_prefix="development_diagnostic")


def run_simulation_diagnostic(*, method, output_root, seed=17):
    return run_development_diagnostic(domain="simulation", method=method, output_root=output_root, seed=seed)


def run_development_diagnostic(*, domain, method, output_root, seed=17, fold_index=0,
                               data_root="artifacts/application_evaluation/2026-09-06_v4-public-development",
                               registry_path="docs/requirements/thesis-v4-public-subjects.json",
                               simulation_root="artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development",
                               candidate_name=None, prefetch_cpu_consumers=False,
                               phase="screen", routes=("self_supervised", "task_guided")):
    """Diagnose both routes on complete development folds, leaving confirmation sealed."""
    _require_diagnostic_device(seed)
    torch.set_num_threads(1)
    from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
    options = candidate_options(method, candidate_name) if candidate_name is not None else None
    routes = tuple(routes)
    if (phase not in ("screen", "review") or not routes or len(set(routes)) != len(routes)
        or not set(routes) <= {"self_supervised", "task_guided"}):
        raise ValueError("invalid development phase or representation routes")
    if (not options or phase == "screen") and (seed != 17 or routes != ("self_supervised", "task_guided")):
        raise ValueError("initial diagnostics/screens require seed 17 and both routes")
    if phase == "review" and options is None:
        raise ValueError("review requires an explicit fixed candidate")
    reviewing = phase == "review"
    if prefetch_cpu_consumers and (not options or domain != "simulation"):
        raise ValueError("CPU consumer prefetch applies to simulation candidate units")
    curve_updates = ((1500,) if reviewing else (300,)) if options else CURVE_UPDATES
    guided_updates = ((500,) if reviewing else (200,)) if options else CURVE_UPDATES
    root = Path(output_root) / domain / method
    if options:
        root = root / candidate_name
    if reviewing:
        root = root / "review" / f"seed{seed}"
    if domain != "simulation":
        root = root / f"fold{fold_index + 1:02d}"
    root.mkdir(parents=True, exist_ok=True)
    source = v4_workflow_source_sha256()
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "format": "chronaris.v4_development_learning_curve.v1", "source_code_sha256": source,
        "method": method, "seed": seed, "domain": domain, "fold_index": fold_index,
        "scope": "single_factor_development" if options else "clean_development_learning_curves", "completed_consumers": [],
        "curve_updates": list(curve_updates), "guided_updates": list(guided_updates),
        "candidate_options": options, "confirmation_opened": False,
        "phase": phase, "routes": list(routes),
        "prefetch_cpu_consumers": prefetch_cpu_consumers, "cpu_prefit_results": {}}
    if state.get("phase", "screen") != phase or tuple(state.get("routes", ("self_supervised", "task_guided"))) != routes:
        raise ValueError("frozen development phase or routes changed")
    state.update(phase=phase, routes=list(routes))
    if state.get("prefetch_cpu_consumers", False) != prefetch_cpu_consumers:
        raise ValueError("development execution schedule changed")
    if json.dumps(state.get("candidate_options"), sort_keys=True) != json.dumps(options, sort_keys=True):
        raise ValueError("development candidate configuration changed")
    if state["source_code_sha256"] != source or state["method"] != method or state["seed"] != seed:
        raise ValueError("learning-curve source/config changed; use a new run root")
    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(state_path)
    with _periodic_training_heartbeat(f"diagnostic_{method}", 30., root=root) as progress, (
        ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) if prefetch_cpu_consumers
        else nullcontext(None)) as cpu_pool:
        provider, schema, fold, hierarchy, digest, targets, definitions, data = load_development_inputs(
            domain, data_root, registry_path, fold_index=fold_index, simulation_root=simulation_root)
        if options and domain == "simulation" and "__training512" not in fold.fold_id:
            raise ValueError("v4 candidates require the activated 512-trajectory training role")
        if state.get("data_manifest_sha256", digest) != digest:
            raise ValueError("learning-curve data changed")
        state.update(data_manifest_sha256=digest, fold=fold.to_dict())
        save()
        normalizer, calibration = development_normalization(domain, provider, schema, fold, digest)
        chronaris = method == "chronaris"
        progress["phase"] = f"pretraining_{curve_updates[-1]}"
        training = train_pretext_candidate(method,
            candidate=EncoderCandidateConfig(candidate_id="C", hidden_dim=options["hidden_dim"] if options else 32, learning_rate=3e-4),
            batch=None, batch_provider=provider, fold=fold, physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=(), normalizer=normalizer,
            output_root=root / "self_supervised", config=CandidateScreenConfig(max_updates=curve_updates[-1], batch_size=4,
                effective_batch_size=16 if domain == "dingxin" else 32, weight_decay=1e-4, device="cuda", early_stopping=reviewing,
                seed=seed, minimum_updates=500, patience=5,
                semantic_event_enabled=chronaris, learnable_semantic_queries=chronaris,
                physics_calibration=calibration if chronaris else None, physics_weight=.05,
                validation_interval=100, validation_updates=curve_updates, retained_updates=curve_updates,
                sampling_hierarchy=hierarchy, data_manifest_sha256=digest, cuda_graph_recurrence=chronaris,
                **(options["training"] if options else {})),
            augmentation_policy=AugmentationPolicy(missingness_mixture=options["missingness_mixture"] if options else False),
            chronaris_fusion_kind="safe_lag" if chronaris else "multiscale", chronaris_mechanism_enabled=chronaris,
            chronaris_explicit_shift_enabled=chronaris, chronaris_explicit_shift_weight=.1 if chronaris else 0.,
            chronaris_event_pair_weight=0.)
        state["self_supervised_training"] = asdict(training)
        save()
        if domain == "simulation":
            targets = build_guarded_application_consumer_targets(data,
                completed_pretraining_checkpoints=(training.best_checkpoint_path,), task_guided_development=True, smoke_only=False)
        roles = {role: getattr(fold, role + "_sample_ids") for role in ("train", "validation", "held_out")}
        protocol = None
        if domain == "simulation":
            protocol = ApplicationConsumerProtocol(linear=LinearConsumerConfig(random_state=seed, tune_on_validation=True),
                minirocket=MiniRocketConsumerConfig(random_state=seed, tune_on_validation=True),
                tcn=TCNConsumerConfig(epochs=40, patience=6, seed=seed, device="cuda"))
        context = native_consumer_context(domain, data, fold) if domain != "simulation" else None
        def evaluate(outputs, route, update):
            consumer_root = root / "consumers" / route / str(update)
            supervised = route == "task_guided"
            if domain == "simulation":
                return asdict(run_application_method_consumers(method_name=method, outputs=outputs, targets=targets,
                    output_root=consumer_root, fold_id=fold.fold_id,
                    protocol=replace(protocol, label_used_for_encoder_training=supervised)))
            return run_native_method_consumers(outputs=outputs, targets=targets, definitions=definitions, context=context,
                output_root=consumer_root, label_used_for_encoder_training=supervised, seed=seed)
        pending_consumers = {}
        for update in curve_updates if "self_supervised" in routes else ():
            key = f"self_supervised:{update}"
            if key in state["completed_consumers"]:
                continue
            progress["phase"] = key
            checkpoint = (Path(training.best_checkpoint_path) if options else
                          Path(training.last_checkpoint_path).with_name(f"update_{update:06d}.pt"))
            encoder, _, normalizer, _ = load_common_pretraining_checkpoint(checkpoint, device="cuda", allow_diagnostic_snapshot=True)
            outputs = _snapshot_outputs(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint, provider=provider,
                fold=fold, root=root / "representations" / "self_supervised" / str(update))
            if cpu_pool is not None and "task_guided" in routes:
                pending_consumers[update] = (cpu_pool.submit(_prepare_cpu_consumer_artifacts,
                    method_name=method, outputs=outputs, targets=targets,
                    output_root=root / "consumers" / "self_supervised" / str(update), fold_id=fold.fold_id, protocol=protocol), outputs)
                del encoder
                continue
            result = evaluate(outputs, "self_supervised", update)
            (root / f"self_supervised_{update}_consumers.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            state["completed_consumers"].append(key)
            save()
            del encoder, outputs, result
            gc.collect()
            torch.cuda.empty_cache()
        if "task_guided" not in routes:
            state["completed"] = all(f"self_supervised:{update}" in state["completed_consumers"] for update in curve_updates)
            save()
            return state
        progress["phase"] = f"task_guided_{guided_updates[-1]}"
        encoder, _, normalizer, _ = load_common_pretraining_checkpoint(training.best_checkpoint_path, device="cuda")
        with isolated_training_rng(seed):
            model = EndToEndApplicationModel(method_name=method, encoder=encoder, normalizer=normalizer,
                naive_encoder=None, task_definitions=definitions)
        guided = train_end_to_end_application_method(model=model, batch=None, batch_provider=provider, targets=targets,
            role_sample_ids=roles, source_checkpoint_path=training.best_checkpoint_path, output_root=root / "task_guided",
            config=EndToEndFineTuningConfig(max_updates=guided_updates[-1], head_warmup_updates=50, batch_size=4,
                effective_batch_size=16 if domain == "dingxin" else 32, weight_decay=1e-4, device="cuda", early_stopping=reviewing,
                seed=seed, minimum_updates=200, patience=4, validation_interval=50,
                retained_updates=guided_updates, sampling_hierarchy=hierarchy, data_manifest_sha256=digest))
        state["task_guided_training"] = asdict(guided)
        save()
        for update, (future, outputs) in pending_consumers.items():
            key = f"self_supervised:{update}"
            progress["phase"] = key
            state.setdefault("cpu_prefit_results", {})[key] = future.result()
            result = evaluate(outputs, "self_supervised", update)
            (root / f"self_supervised_{update}_consumers.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            state["completed_consumers"].append(key)
            save()
        for update in guided_updates:
            key = f"task_guided:{update}"
            if key in state["completed_consumers"]:
                continue
            progress["phase"] = key
            checkpoint = (Path(guided.best_checkpoint_path) if options else
                          Path(guided.last_checkpoint_path).with_name(f"joint_update_{update:06d}.pt"))
            outputs = export_finetuned_application_representations(model=model, checkpoint_path=checkpoint,
                batch=None, batch_provider=provider, role_sample_ids=roles, batch_size=4,
                output_root=root / "representations" / "task_guided" / str(update), export_roles=("train", "validation"),
                allow_diagnostic_snapshot=True)
            result = evaluate(outputs, "task_guided", update)
            (root / f"task_guided_{update}_consumers.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            state["completed_consumers"].append(key)
            save()
        expected = {f"{route}:{update}" for route in routes for update in
                    (curve_updates if route == "self_supervised" else guided_updates)}
        state["completed"] = expected <= set(state["completed_consumers"])
        save()
        return state
