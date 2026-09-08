import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_native_frozen_evaluation as evaluation
from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method)
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
from chronaris.modeling.training import (CandidateScreenConfig, EncoderCandidateConfig,
    train_pretext_candidate, load_common_pretraining_checkpoint)
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples, select_observation_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from tests.representation.test_contracts import _sample


@pytest.mark.parametrize("domain", ["clare", "dingxin"])
def test_completed_native_encoders_export_outer_roles_and_refit_consumers(tmp_path, monkeypatch, domain):
    if domain == "dingxin":
        from chronaris.evaluation.application_tasks.v4_dingxin_data import V4DingxinData, dingxin_v4_inner_folds, _task_targets
        from chronaris.evaluation.dingxin.simple_downstream_protocol import fit_simple_loso_targets
        from tests.evaluation.dingxin.test_simple_downstream_protocol import _synthetic_raw_bundle
        raw = _synthetic_raw_bundle()
        folds, embargo = dingxin_v4_inner_folds(raw.contexts)
        fitted = fit_simple_loso_targets(raw, fit_context_ids_by_fold={f.fold_id: f.train_sample_ids for f in folds})
        samples = [_sample(sample, shift=i / 100) for i, sample in enumerate(raw.contexts.context_id)]
    else:
        samples = [_sample(f"native_outer_{i}", shift=i / 10) for i in range(12)]
    batch = collate_observation_samples(samples)
    fold = FoldLineage("native_outer", batch.sample_ids[:6], batch.sample_ids[6:9], batch.sample_ids[9:])
    if domain == "dingxin":
        fold = folds[0]
    roles = {role: getattr(fold, role + "_sample_ids") for role in ("train", "validation", "held_out")}
    accessed = []
    def provider(ids):
        accessed.extend(ids)
        return select_observation_batch(batch, ids)
    def training_provider(ids):
        assert not set(ids) & set(fold.held_out_sample_ids)
        return provider(ids)
    definitions = (ApplicationTaskDefinition("classify", "classification", 3), ApplicationTaskDefinition("response", "regression", 1))
    values = {"classify": torch.arange(len(batch.sample_ids)) % 3, "response": torch.arange(len(batch.sample_ids), dtype=torch.float32) / 10}
    targets = ApplicationTaskTargets(batch.sample_ids, values, {n: torch.ones_like(v, dtype=torch.bool) for n,v in values.items()},
                                     {"source_role": "fixed_confirmation_subjects", "smoke_only": True})
    if domain == "dingxin":
        targets, definitions, _ = _task_targets(fitted, fold, fold.train_sample_ids + fold.validation_sample_ids, "inner_training")
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids)
    schema = samples[0].schema
    digest = "a" * 64
    pretraining = train_pretext_candidate("physiology_only", candidate=EncoderCandidateConfig(candidate_id="C"),
        batch=None, batch_provider=training_provider, fold=fold, physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=(), normalizer=normalizer, output_root=tmp_path / "ssl",
        config=CandidateScreenConfig(max_updates=2, batch_size=2, effective_batch_size=4, device="cpu", seed=17,
                                    validation_interval=1, early_stopping=False, data_manifest_sha256=digest))
    encoder, _, normalizer, _ = load_common_pretraining_checkpoint(pretraining.best_checkpoint_path, device="cpu")
    model = EndToEndApplicationModel(method_name="physiology_only", encoder=encoder, normalizer=normalizer,
                                    naive_encoder=None, task_definitions=definitions)
    guided = train_end_to_end_application_method(model=model, batch=None, batch_provider=training_provider,
        targets=targets, role_sample_ids=roles, source_checkpoint_path=pretraining.best_checkpoint_path,
        output_root=tmp_path / "guided", config=EndToEndFineTuningConfig(max_updates=1, head_warmup_updates=1,
            batch_size=2, effective_batch_size=4, device="cpu", seed=17, validation_interval=1,
            early_stopping=False, data_manifest_sha256=digest))
    assert not set(accessed) & set(fold.held_out_sample_ids)
    data = SimpleNamespace(targets=targets, task_definitions=definitions, dataset=SimpleNamespace(batch_provider=provider),
        sample_manifest=[{"sample_id": sample, "subject_id": role} for role, ids in roles.items() for sample in ids])
    if domain == "dingxin":
        data = V4DingxinData(SimpleNamespace(load_batch=provider), folds, embargo, raw, fitted,
            {fold.fold_id: targets}, {fold.fold_id: definitions}, {}, {}, (), digest)
    monkeypatch.setattr(evaluation, "load_development_inputs", lambda *args, **kwargs:
                        (training_provider, schema, fold, {}, digest, targets, definitions, data))
    for route, checkpoint in (("self_supervised", pretraining.best_checkpoint_path), ("task_guided", guided.best_checkpoint_path)):
        root = tmp_path / "outer" / route
        kwargs = dict(domain=domain, fold_index=0, checkpoint=checkpoint, checkpoint_sha256=sha256_file(checkpoint),
            route=route, output_root=root, device="cpu", engineering_only=True, minirocket_kernels=84)
        result = evaluation.run_native_frozen_evaluation(**kwargs)
        assert result["completed"] and result["source"]["engineering_only"]
        expected_roles = {role: tuple(result["source"]["consumer_fold"][role + "_sample_ids"])
                          for role in roles if result["source"]["consumer_fold"][role + "_sample_ids"]}
        for role in expected_roles:
            manifest = json.loads((root / "representations" / role / "representation_manifest.json").read_text())
            assert manifest["sample_ids"] == list(expected_roles[role])
            assert manifest["label_used_for_encoder_training"] is (route == "task_guided")
        for family in (("linear",) if domain == "dingxin" else ("linear", "minirocket")):
            score = json.loads((root / "consumers" / f"{family}_results.json").read_text())
            assert score["evaluations"]["held_out"]["sample_count"] == len(fold.held_out_sample_ids)
            assert all(not set(row["train_sample_ids"]) & set(fold.held_out_sample_ids) for row in score["fit_rows"])
        before = len(accessed)
        assert evaluation.run_native_frozen_evaluation(**kwargs) == result
        assert len(accessed) == before
        with pytest.raises(ValueError, match="selected frozen checkpoint changed"):
            evaluation.run_native_frozen_evaluation(**(kwargs | {"checkpoint_sha256": "0" * 64}))
        with pytest.raises(ValueError, match="fixed CUDA"):
            evaluation.run_native_frozen_evaluation(**(kwargs | {"engineering_only": False}))
        path = root / "representations/held_out/representation_manifest.json"
        metadata = json.loads(path.read_text())
        metadata["label_used_for_encoder_training"] = route != "task_guided"
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="provenance"):
            evaluation.run_native_frozen_evaluation(**kwargs)
    assert set(accessed) == set(batch.sample_ids)
    source_path = Path(pretraining.best_checkpoint_path)
    source_path.write_bytes(source_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="initialization checkpoint changed"):
        evaluation.run_native_frozen_evaluation(**(kwargs | {"output_root": tmp_path / "changed_initialization"}))
    from chronaris.evaluation.application_tasks.v4_naive_baseline import fit_v4_naive_encoder
    from chronaris.representation import load_fusion_stream_batch
    for seed in (17, 29, 43):
        before = len(accessed)
        naive_path, fit = fit_v4_naive_encoder(provider=training_provider, fold=fold, normalizer=normalizer,
            data_manifest_sha256=digest, output_root=tmp_path / 'naive' / str(seed), seed=seed)
        assert set(accessed[before:]) == set(fold.train_sample_ids)
        assert fit['optimizer_updates'] == 0 and fit['seed'] == seed
        before = len(accessed)
        assert fit_v4_naive_encoder(provider=training_provider, fold=fold, normalizer=normalizer,
            data_manifest_sha256=digest, output_root=tmp_path / 'naive' / str(seed), seed=seed) == (naive_path, fit)
        assert len(accessed) == before
        exported = []
        for route in ('self_supervised', 'task_guided'):
            out = tmp_path / 'naive_outer' / str(seed) / route
            result = evaluation.run_native_frozen_evaluation(domain=domain, fold_index=0, checkpoint=naive_path,
                checkpoint_sha256=sha256_file(naive_path), route=route, output_root=out, device='cpu',
                engineering_only=True, minirocket_kernels=84, method='naive_time_sync')
            assert result['source']['encoder_optimizer_updates'] == 0
            assert result['source']['label_used_for_encoder_training'] is False
            assert result['source']['nonparametric_representation_shared_between_routes']
            exported.append(load_fusion_stream_batch(out / 'representations/held_out'))
        assert torch.equal(exported[0].sequence_embedding, exported[1].sequence_embedding)
        assert torch.equal(exported[0].valid_mask, exported[1].valid_mask)
