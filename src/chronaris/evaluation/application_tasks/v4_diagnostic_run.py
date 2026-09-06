"""Full-development learning curves using the shared trainers and consumers."""
from dataclasses import asdict, replace
from pathlib import Path
import gc
import json

import torch

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import build_guarded_application_consumer_targets
from chronaris.evaluation.application_tasks.application_consumer_runtime import ApplicationConsumerProtocol, run_application_method_consumers
from chronaris.evaluation.application_tasks.application_consumers import LinearConsumerConfig, MiniRocketConsumerConfig, TCNConsumerConfig
from chronaris.evaluation.application_tasks.application_finetuning import EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method
from chronaris.evaluation.application_tasks.application_finetuning_export import export_finetuned_application_representations
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, development_normalization
from chronaris.modeling.training import CandidateScreenConfig, EncoderCandidateConfig, TrainedFusionAdapter, load_common_pretraining_checkpoint, train_pretext_candidate
from chronaris.modeling.training.candidate_checkpoint import candidate_source_code_sha256
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.representation import load_fusion_stream_batch, write_fusion_stream_batch
from chronaris.representation.oof_export import _concatenate_fusion_batches
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


CURVE_UPDATES = (50, 200, 500)


def _snapshot_outputs(*, encoder, normalizer, checkpoint, provider, fold, root):
    adapter = TrainedFusionAdapter(encoder=encoder, normalizer=normalizer, fold_id=fold.fold_id,
                                  checkpoint_sha256=sha256_file(checkpoint))
    outputs = {}
    for role in ("train", "validation"):
        ids = getattr(fold, role + "_sample_ids")
        path = root / role
        if (path / "representation_manifest.json").exists():
            output = load_fusion_stream_batch(path)
            if output.sample_ids != ids or output.checkpoint_sha256 != adapter.checkpoint_sha256:
                raise ValueError("learning-curve representation provenance changed")
        else:
            output = _concatenate_fusion_batches([adapter(provider(ids[i:i + 4])) for i in range(0, len(ids), 4)])
            write_fusion_stream_batch(output, root=path, export_role=f"development_diagnostic_{role}")
        outputs[role] = output
    return outputs


def run_simulation_diagnostic(*, method, output_root, seed=17):
    """Diagnose both routes on all 1,280 development windows, leaving confirmation sealed."""
    if seed != 17 or not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise ValueError("initial learning curves require CUDA and seed 17")
    torch.set_num_threads(1)
    root = Path(output_root) / "simulation" / method
    root.mkdir(parents=True, exist_ok=True)
    source = candidate_source_code_sha256()
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "format": "chronaris.v4_development_learning_curve.v1", "source_code_sha256": source,
        "method": method, "seed": seed, "scope": "clean_development_learning_curves", "completed_consumers": [],
        "curve_updates": list(CURVE_UPDATES), "confirmation_opened": False}
    if state["source_code_sha256"] != source or state["method"] != method or state["seed"] != seed:
        raise ValueError("learning-curve source/config changed; use a new run root")
    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(state_path)
    with _periodic_training_heartbeat(f"diagnostic_{method}", 30., root=root) as progress:
        provider, schema, fold, hierarchy, digest, _, definitions, data = load_development_inputs("simulation",
            "artifacts/application_evaluation/2026-09-06_v4-public-development", "docs/requirements/thesis-v4-public-subjects.json")
        if state.get("data_manifest_sha256", digest) != digest:
            raise ValueError("learning-curve data changed")
        state.update(data_manifest_sha256=digest, fold=fold.to_dict())
        save()
        normalizer, calibration = development_normalization("simulation", provider, schema, fold, digest)
        chronaris = method == "chronaris"
        progress["phase"] = "pretraining_500"
        training = train_pretext_candidate(method,
            candidate=EncoderCandidateConfig(candidate_id="C", hidden_dim=32, learning_rate=3e-4),
            batch=None, batch_provider=provider, fold=fold, physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=(), normalizer=normalizer,
            output_root=root / "self_supervised", config=CandidateScreenConfig(max_updates=CURVE_UPDATES[-1], batch_size=4,
                effective_batch_size=32, weight_decay=1e-4, device="cuda", early_stopping=False,
                semantic_event_enabled=chronaris, learnable_semantic_queries=chronaris,
                physics_calibration=calibration if chronaris else None, physics_weight=.05,
                validation_interval=100, validation_updates=CURVE_UPDATES, retained_updates=CURVE_UPDATES,
                sampling_hierarchy=hierarchy, data_manifest_sha256=digest, cuda_graph_recurrence=chronaris),
            chronaris_fusion_kind="safe_lag" if chronaris else "multiscale", chronaris_mechanism_enabled=chronaris,
            chronaris_explicit_shift_enabled=chronaris, chronaris_explicit_shift_weight=.1 if chronaris else 0.,
            chronaris_event_pair_weight=0.)
        state["self_supervised_training"] = asdict(training)
        save()
        targets = build_guarded_application_consumer_targets(data,
            completed_pretraining_checkpoints=(training.best_checkpoint_path,), task_guided_development=True, smoke_only=False)
        roles = data.role_sample_ids
        protocol = ApplicationConsumerProtocol(linear=LinearConsumerConfig(random_state=seed, tune_on_validation=True),
            minirocket=MiniRocketConsumerConfig(random_state=seed, tune_on_validation=True),
            tcn=TCNConsumerConfig(epochs=40, patience=6, seed=seed, device="cuda"))
        for update in CURVE_UPDATES:
            key = f"self_supervised:{update}"
            if key in state["completed_consumers"]:
                continue
            progress["phase"] = key
            checkpoint = Path(training.last_checkpoint_path).with_name(f"update_{update:06d}.pt")
            encoder, _, normalizer, _ = load_common_pretraining_checkpoint(checkpoint, device="cuda", allow_diagnostic_snapshot=True)
            outputs = _snapshot_outputs(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint, provider=provider,
                fold=fold, root=root / "representations" / "self_supervised" / str(update))
            result = run_application_method_consumers(method_name=method, outputs=outputs, targets=targets,
                output_root=root / "consumers" / "self_supervised" / str(update), fold_id=fold.fold_id, protocol=protocol)
            (root / f"self_supervised_{update}_consumers.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")
            state["completed_consumers"].append(key)
            save()
            del encoder, outputs, result
            gc.collect()
            torch.cuda.empty_cache()
        progress["phase"] = "task_guided_500"
        encoder, _, normalizer, _ = load_common_pretraining_checkpoint(training.best_checkpoint_path, device="cuda")
        with isolated_training_rng(seed):
            model = EndToEndApplicationModel(method_name=method, encoder=encoder, normalizer=normalizer,
                naive_encoder=None, task_definitions=definitions)
        guided = train_end_to_end_application_method(model=model, batch=None, batch_provider=provider, targets=targets,
            role_sample_ids=roles, source_checkpoint_path=training.best_checkpoint_path, output_root=root / "task_guided",
            config=EndToEndFineTuningConfig(max_updates=CURVE_UPDATES[-1], head_warmup_updates=50, batch_size=4,
                effective_batch_size=32, weight_decay=1e-4, device="cuda", early_stopping=False,
                retained_updates=CURVE_UPDATES, sampling_hierarchy=hierarchy, data_manifest_sha256=digest))
        state["task_guided_training"] = asdict(guided)
        save()
        for update in CURVE_UPDATES:
            key = f"task_guided:{update}"
            if key in state["completed_consumers"]:
                continue
            progress["phase"] = key
            checkpoint = Path(guided.last_checkpoint_path).with_name(f"joint_update_{update:06d}.pt")
            outputs = export_finetuned_application_representations(model=model, checkpoint_path=checkpoint,
                batch=None, batch_provider=provider, role_sample_ids=roles, batch_size=4,
                output_root=root / "representations" / "task_guided" / str(update), export_roles=("train", "validation"),
                allow_diagnostic_snapshot=True)
            result = run_application_method_consumers(method_name=method, outputs=outputs, targets=targets,
                output_root=root / "consumers" / "task_guided" / str(update), fold_id=fold.fold_id,
                protocol=replace(protocol, label_used_for_encoder_training=True))
            (root / f"task_guided_{update}_consumers.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")
            state["completed_consumers"].append(key)
            save()
        state["completed"] = len(state["completed_consumers"]) == 6
        save()
        return state
