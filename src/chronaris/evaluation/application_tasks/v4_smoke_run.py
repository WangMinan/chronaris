"""Real-data engineering closure through the existing two training routes."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, root_mean_squared_error

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import build_guarded_application_consumer_targets
from chronaris.evaluation.application_tasks.application_finetuning import EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method
from chronaris.evaluation.application_tasks.application_finetuning_export import export_finetuned_application_representations
from chronaris.evaluation.application_tasks.application_task_heads import SIMULATION_TASKS, application_targets
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_classifier, fit_regressor
from chronaris.evaluation.application_tasks.v4_correctness import audit_checkpoint_causality
from chronaris.evaluation.application_tasks.v4_dingxin_data import load_v4_dingxin_development
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development
from chronaris.evaluation.application_tasks.v4_simulation_data import load_v4_simulation_development, simulation_sampling_hierarchy
from chronaris.modeling.training import CandidateScreenConfig, EncoderCandidateConfig, TrainedFusionAdapter, load_common_pretraining_checkpoint, train_pretext_candidate
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.models.alignment.calibrated_physics import SIMULATION_RELATIONS, fit_physics_calibration
from chronaris.representation import TrainOnlyRobustNormalizer, select_observation_batch, write_fusion_stream_batch
from chronaris.representation.oof_export import _concatenate_fusion_batches
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _hash_prefix(ids, count):
    return tuple(sorted(ids, key=lambda value: hashlib.sha256(f"v4-engineering-smoke:{value}".encode()).hexdigest())[:count])


def _smoke_inputs(domain, data_root, registry_path):
    simulation = None
    if domain in {"cogpilot", "clare"}:
        data = load_prepared_public_development(domain, output_root=data_root, registry_path=registry_path)
        registry = json.loads(Path(registry_path).read_text())
        fold = data.fold(registry["domains"][domain]["folds"]["development"][0])
        fold = replace(fold, fold_id=fold.fold_id + "__engineering_smoke",
            train_sample_ids=_hash_prefix(fold.train_sample_ids, 32), validation_sample_ids=_hash_prefix(fold.validation_sample_ids, 8))
        provider, schema = data.dataset.batch_provider, data.dataset.schema
        hierarchy, digest = data.sampling_hierarchy(fold), data.prepared_manifest_sha256
        targets, definitions = data.targets, data.task_definitions
    elif domain == "dingxin":
        data = load_v4_dingxin_development()
        fold = data.folds[0]
        provider, schema = data.development_provider(fold), data.index.plan.schema
        hierarchy, digest = data.sampling_by_fold[fold.fold_id], data.data_manifest_sha256
        targets, definitions = data.targets_by_fold[fold.fold_id], data.definitions_by_fold[fold.fold_id]
        fold = replace(fold, fold_id=fold.fold_id + "__engineering_smoke")
    elif domain == "simulation":
        data, fold = load_v4_simulation_development(simulation_root="artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development",
            registry_path="docs/requirements/thesis-v4-simulation-manifest.json")
        fold = replace(fold, fold_id=fold.fold_id + "__engineering_smoke",
            train_sample_ids=_hash_prefix(fold.train_sample_ids, 32), validation_sample_ids=_hash_prefix(fold.validation_sample_ids, 8))
        selected = fold.train_sample_ids + fold.validation_sample_ids
        simulation = replace(data, batch=select_observation_batch(data.batch, selected),
            role_sample_ids={role: getattr(fold, role + "_sample_ids") for role in ("train", "validation", "held_out")},
            sample_manifest_rows=tuple(next(row for row in data.sample_manifest_rows if row["sample_id"] == sample) for sample in selected))
        provider, schema = lambda ids: select_observation_batch(simulation.batch, ids), data.schema
        hierarchy, digest = simulation_sampling_hierarchy(data, fold), sha256_file("docs/requirements/thesis-v4-simulation-manifest.json")
        targets, definitions = None, SIMULATION_TASKS
    else:
        raise ValueError("unknown v4 smoke domain")
    allowed = set(fold.train_sample_ids + fold.validation_sample_ids)
    def guarded(ids):
        if not set(ids) <= allowed:
            raise ValueError("engineering smoke cannot open confirmation observations")
        return provider(ids)
    return guarded, schema, fold, hierarchy, digest, targets, definitions, simulation


def _export_self_supervised(encoder, normalizer, checkpoint, fold, provider, root):
    adapter = TrainedFusionAdapter(encoder=encoder, normalizer=normalizer, fold_id=fold.fold_id,
                                  checkpoint_sha256=sha256_file(checkpoint))
    outputs = {}
    for role in ("train", "validation"):
        ids = getattr(fold, role + "_sample_ids")
        output = _concatenate_fusion_batches([adapter(provider(ids[start:start + 4])) for start in range(0, len(ids), 4)])
        write_fusion_stream_batch(output, root=root / role, export_role=f"v4_smoke_{role}")
        outputs[role] = output
    return outputs


def _consumer_smoke(outputs, targets, definitions):
    """Fresh fixed consumers verify that removing the training heads is sufficient."""
    index = {sample: i for i, sample in enumerate(targets.sample_ids)}
    positions = {role: [index[sample] for sample in output.sample_ids] for role, output in outputs.items()}
    rows = []
    for task in definitions:
        if task.kind == "sequence_classification":
            continue  # The full simulation sequence consumers run in the diagnostic stage.
        for field in range(task.output_dim if task.kind == "regression" else 1):
            value, mask = targets.values[task.name], targets.valid_masks[task.name]
            if value.ndim == 2:
                value, mask = value[:, field], mask[:, field]
            selected = {role: mask[ids].numpy() for role, ids in positions.items()}
            x = {role: output.pooled_embedding.numpy()[selected[role]] for role, output in outputs.items()}
            y = {role: value[ids].numpy()[selected[role]] for role, ids in positions.items()}
            if not len(y["validation"]) or not len(y["train"]):
                rows.append({"task": task.name, "field": field, "status": "unavailable_no_valid_targets"})
                continue
            if task.kind == "classification":
                if len(np.unique(y["train"])) < 2:
                    rows.append({"task": task.name, "status": "unavailable_single_training_class"})
                    continue
                consumer, _ = fit_classifier(x["train"], y["train"], None, None, c_values=(1.,), random_state=17, scaler_with_mean=True)
                metric = float(f1_score(y["validation"], consumer.predict(x["validation"]), average="macro", labels=list(range(task.output_dim)), zero_division=0))
                metric_name = "macro_f1"
            else:
                consumer, _ = fit_regressor(x["train"], y["train"], None, None, alpha_values=(1.,), scaler_with_mean=True)
                metric, metric_name = float(root_mean_squared_error(y["validation"], consumer.predict(x["validation"]))), "rmse"
            rows.append({"task": task.name, "field": field, "status": "completed", "metric": metric_name,
                "value": metric, "train_valid_samples": len(y["train"]), "validation_valid_samples": len(y["validation"])})
    return rows


def run_v4_smoke(*, domain, output_root, task_mode="all",
                 data_root="artifacts/application_evaluation/2026-09-06_v4-public-development",
                 registry_path="docs/requirements/thesis-v4-public-subjects.json"):
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise ValueError("v4 neural smoke requires the designated RTX 4090")
    torch.set_num_threads(1)
    root = Path(output_root) / domain / task_mode
    root.mkdir(parents=True, exist_ok=True)
    with _periodic_training_heartbeat(f"v4_smoke_{domain}", 30., root=root) as progress:
        provider, schema, fold, hierarchy, digest, targets, definitions, simulation = _smoke_inputs(domain, data_root, registry_path)
        roles = {role: getattr(fold, role + "_sample_ids") for role in ("train", "validation", "held_out")}
        normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider, train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids, batch_size=4)
        calibration = fit_physics_calibration(normalizer, provider, train_sample_ids=fold.train_sample_ids,
            vehicle_feature_names=schema.vehicle_feature_names, relations=SIMULATION_RELATIONS if domain == "simulation" else ())
        calibration["domain_relation_scope"] = "generator_equations" if domain == "simulation" else "not_applicable_without_confirmed_units_and_coordinates"
        effective = 16 if domain == "dingxin" else 32
        progress["phase"] = "self_supervised"
        torch.cuda.reset_peak_memory_stats()
        training = train_pretext_candidate("chronaris", candidate=EncoderCandidateConfig(candidate_id="C", hidden_dim=32, learning_rate=3e-4),
            batch=None, batch_provider=provider, fold=fold, physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=(), normalizer=normalizer,
            output_root=root / "self_supervised", config=CandidateScreenConfig(max_updates=10, batch_size=4,
                effective_batch_size=effective, weight_decay=1e-4, device="cuda", semantic_event_enabled=True,
                learnable_semantic_queries=True, physics_calibration=calibration, physics_weight=.05,
                validation_interval=10, early_stopping=False, sampling_hierarchy=hierarchy, data_manifest_sha256=digest,
                cuda_graph_recurrence=True), chronaris_fusion_kind="safe_lag", chronaris_mechanism_enabled=True,
            chronaris_explicit_shift_enabled=True, chronaris_explicit_shift_weight=.1, chronaris_event_pair_weight=0.)
        self_peak = torch.cuda.max_memory_allocated()
        if simulation is not None:
            targets = application_targets(build_guarded_application_consumer_targets(simulation,
                completed_pretraining_checkpoints=(training.best_checkpoint_path,), task_guided_development=True))
        if task_mode == "single":
            definitions = (definitions[0],)
            names = {task.name for task in definitions}
            targets = replace(targets, values={key: value for key, value in targets.values.items() if key in names},
                              valid_masks={key: value for key, value in targets.valid_masks.items() if key in names})
        elif task_mode != "all":
            raise ValueError("unknown engineering task mode")
        encoder, _, normalizer, _ = load_common_pretraining_checkpoint(training.best_checkpoint_path, device="cuda")
        self_outputs = _export_self_supervised(encoder, normalizer, training.best_checkpoint_path, fold, provider, root / "self_supervised_representations")
        progress["phase"] = "task_guided"
        with isolated_training_rng(17):
            model = EndToEndApplicationModel(method_name="chronaris", encoder=encoder, normalizer=normalizer,
                naive_encoder=None, task_definitions=definitions)
        torch.cuda.reset_peak_memory_stats()
        guided = train_end_to_end_application_method(model=model, batch=None, batch_provider=provider,
            targets=targets, role_sample_ids=roles, source_checkpoint_path=training.best_checkpoint_path,
            output_root=root / "task_guided", config=EndToEndFineTuningConfig(max_updates=10, head_warmup_updates=50,
                batch_size=4, effective_batch_size=effective, weight_decay=1e-4, device="cuda", early_stopping=False,
                sampling_hierarchy=hierarchy, data_manifest_sha256=digest))
        guided_peak = torch.cuda.max_memory_allocated()
        guided_outputs = export_finetuned_application_representations(model=model, checkpoint_path=guided.best_checkpoint_path,
            batch=None, batch_provider=provider, role_sample_ids=roles, output_root=root / "task_guided_representations",
            batch_size=4, export_roles=("train", "validation"))
        progress["phase"] = "consumers_and_causality"
        consumers = {route: _consumer_smoke(outputs, targets, definitions) for route, outputs in
                     (("self_supervised", self_outputs), ("task_guided", guided_outputs))}
        audit_batch = provider(fold.validation_sample_ids[:4])
        causality = [audit_checkpoint_causality(path, audit_batch, cutoff_s=5.)
                     for path in (training.best_checkpoint_path, guided.best_checkpoint_path)]
        result = {"domain": domain, "task_mode": task_mode, "scope": "engineering_smoke_not_candidate_or_confirmation_scores",
            "data_manifest_sha256": digest, "fold": fold.to_dict(), "self_supervised": asdict(training), "task_guided": asdict(guided),
            "peak_allocated_bytes": {"self_supervised": self_peak, "task_guided": guided_peak}, "fixed_consumer_checks": consumers,
            "causality": causality, "encoder_backprop_uses_labels": {"self_supervised": False, "task_guided": True},
            "selection_uses_validation_labels": {"self_supervised": False, "task_guided": True},
            "passed": all(row["passed"] for row in causality) and all(any(row["status"] == "completed" for row in rows) for rows in consumers.values())}
        (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if not result["passed"]:
            raise AssertionError("real native smoke did not satisfy correctness/consumer closure")
        return result
