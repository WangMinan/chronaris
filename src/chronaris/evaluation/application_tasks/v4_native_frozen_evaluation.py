"""Evaluate an explicitly selected completed checkpoint on native outer roles.

The confirmation orchestrator must freeze architecture and budgets before calling this
step. This module never selects candidates or trains an encoder from outer outcomes.
"""
from pathlib import Path
import json

from chronaris.evaluation.application_tasks.application_finetuning_export import (
    load_frozen_application_encoder, export_loaded_application_encoder)
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
from chronaris.evaluation.application_tasks.v4_dingxin_data import build_dingxin_outer_consumer_inputs
from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context, run_native_method_consumers
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def run_native_frozen_evaluation(*, domain, fold_index, checkpoint, checkpoint_sha256, route, output_root,
                                 data_root="artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared",
                                 registry_path="docs/requirements/thesis-v4-public-subjects.json",
                                 device="cuda", engineering_only=False, minirocket_kernels=10_000, method=None):
    if domain not in {"cogpilot", "clare", "dingxin"} or route not in {"self_supervised", "task_guided"}:
        raise ValueError("unsupported native frozen evaluation domain or route")
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("selected frozen checkpoint changed")
    nonparametric = method == "naive_time_sync"
    if not engineering_only:
        if device != ("cpu" if nonparametric else "cuda") or minirocket_kernels != 10_000:
            raise ValueError("formal native evaluation requires the fixed CUDA consumer protocol")
        if not nonparametric:
            _require_diagnostic_device(17)
    _, _, encoder_fold, _, digest, _, _, data = load_development_inputs(domain, data_root, registry_path,
        fold_index=fold_index, subject_role="development" if domain == "dingxin" else "confirmation")
    if nonparametric:
        from chronaris.evaluation.application_tasks.v4_naive_baseline import load_v4_naive_encoder
        encoder, normalizer, payload = load_v4_naive_encoder(checkpoint, fold=encoder_fold, data_manifest_sha256=digest)
    else:
        encoder, normalizer, payload = load_frozen_application_encoder(checkpoint, route=route, fold=encoder_fold, device=device)
    if method is not None and encoder.method_name != method:
        raise ValueError("selected native encoder method changed")
    labels_used = route == "task_guided" and not nonparametric
    if payload["config"].get("data_manifest_sha256") != digest:
        raise ValueError("frozen encoder data manifest differs from evaluation inputs")
    if payload["seed"] not in (17, 29, 43) or (not engineering_only and not nonparametric and payload["config"]["device"] != "cuda"):
        raise ValueError("frozen encoder used an unapproved seed or training device")
    if domain == "dingxin":
        fitted = build_dingxin_outer_consumer_inputs(data, encoder_fold.fold_id)
        fold, targets, definitions, context = (fitted[key] for key in ("fold", "targets", "definitions", "context"))
        provider = data.index.load_batch
    else:
        if data.targets.manifest.get("source_role") != "fixed_confirmation_subjects":
            raise ValueError("native confirmation cannot evaluate development subjects")
        fold, targets, definitions = encoder_fold, data.targets, data.task_definitions
        context = native_consumer_context(domain, data, fold)
        provider = data.dataset.batch_provider
    if not fold.held_out_sample_ids:
        raise ValueError("native frozen evaluation requires an outer held-out role")
    allowed = set(fold.train_sample_ids + fold.validation_sample_ids + fold.held_out_sample_ids)
    def export_provider(ids):
        if not set(ids) <= allowed:
            raise ValueError("frozen export requested samples outside the selected fold")
        return provider(ids)
    source = {"format": "chronaris.v4_native_frozen_evaluation.v1", "domain": domain, "route": route,
        "checkpoint_sha256": checkpoint_sha256, "source_code_sha256": v4_workflow_source_sha256(),
        "data_manifest_sha256": digest, "encoder_fold": encoder_fold.to_dict(), "consumer_fold": fold.to_dict(),
        "seed": payload["seed"], "method": encoder.method_name, "engineering_only": engineering_only,
        "device": device, "minirocket_kernels": minirocket_kernels,
        "label_used_for_encoder_training": labels_used,
        "nonparametric_representation_shared_between_routes": nonparametric,
        "encoder_optimizer_updates": payload["optimizer_updates"],
        "encoder_checkpoint_selection_uses_validation_labels": labels_used,
        "configuration_selection_uses_validation_labels": not nonparametric}
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["source"] != source:
            raise ValueError("frozen native evaluation source/data/config changed")
    else:
        state = {"source": source, "completed": False}
    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(state_path)
    save()
    with _periodic_training_heartbeat("native_frozen_evaluation", 30., root=root) as progress:
        progress.update(phase="frozen_encoder_export", checkpoint=str(checkpoint), fold=fold.fold_id)
        outputs = export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
            provider=export_provider, fold=fold, root=root / "representations", export_roles=("train", "validation", "held_out"),
            export_prefix="engineering_outer" if engineering_only else "frozen_outer",
            label_used_for_encoder_training=labels_used)
        progress["phase"] = "fresh_frozen_consumers"
        state["consumers"] = run_native_method_consumers(outputs=outputs, targets=targets, definitions=definitions,
            context=context, output_root=root / "consumers", label_used_for_encoder_training=labels_used,
            seed=payload["seed"], minirocket_kernels=minirocket_kernels)
        state["completed"] = True
        save()
    return state
