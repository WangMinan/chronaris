from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_diagnostic_run as run
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import ApplicationConsumerSmokeTargets
from chronaris.evaluation.application_tasks.application_task_heads import SIMULATION_TASKS
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples, select_observation_batch
from tests.representation.test_contracts import _sample


@pytest.mark.parametrize("device,candidate_name", [("cpu", None), ("cuda", None), ("cpu", "quality_gate")])
def test_learning_curve_runs_real_trainers_exports_and_all_consumers_without_confirmation(tmp_path, monkeypatch, device, candidate_name):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA diagnostic integration")
    samples = [_sample(f"sample_{i}", shift=i / 10) for i in range(9)]
    batch = collate_observation_samples(samples)
    roles = dict(train=batch.sample_ids[:6], validation=batch.sample_ids[6:], held_out=("sealed",))
    fold = FoldLineage("diagnostic_test__training512" if candidate_name else "diagnostic_test", roles["train"], roles["validation"], roles["held_out"])
    accessed = []
    def provider(ids):
        assert "sealed" not in ids
        accessed.extend(ids)
        return select_observation_batch(batch, ids)
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=roles["train"], held_out_sample_ids=roles["validation"])
    state = torch.arange(batch.query_timestamps_s.shape[1]).repeat(9, 1).remainder(5)
    targets = ApplicationConsumerSmokeTargets(batch.sample_ids, ("train",) * 6 + ("validation",) * 3,
        torch.linspace(0., 1., 9), torch.arange(9).remainder(3), state,
        torch.cat((torch.zeros(9, 1, dtype=torch.bool), state[:, 1:] != state[:, :-1]), dim=1), {"smoke_only": True})
    inputs = (provider, samples[0].schema, fold, {s: (s,) for s in roles["train"]}, "a" * 64,
              None, SIMULATION_TASKS, SimpleNamespace(role_sample_ids=roles))
    monkeypatch.setattr(run, "load_development_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(run, "development_normalization", lambda *args, **kwargs: (normalizer, None))
    monkeypatch.setattr(run, "build_guarded_application_consumer_targets", lambda *args, **kwargs: targets)
    monkeypatch.setattr(run, "CURVE_UPDATES", (1, 2, 3))
    pretraining, guidance = run.CandidateScreenConfig, run.EndToEndFineTuningConfig
    rocket, tcn = run.MiniRocketConsumerConfig, run.TCNConsumerConfig
    scheduled = []
    def tiny_pretraining(**kwargs):
        scheduled.append(("pretraining", kwargs))
        return pretraining(**(kwargs | {"effective_batch_size": 4, "device": device, "max_updates": 3}))
    def tiny_guidance(**kwargs):
        scheduled.append(("guidance", kwargs))
        return guidance(**(kwargs | {"max_updates": 3, "effective_batch_size": 4,
            "head_warmup_updates": 2, "validation_interval": 1, "device": device}))
    monkeypatch.setattr(run, "CandidateScreenConfig", tiny_pretraining)
    monkeypatch.setattr(run, "EndToEndFineTuningConfig", tiny_guidance)
    monkeypatch.setattr(run, "MiniRocketConsumerConfig", lambda **kwargs: rocket(**kwargs, n_kernels=84))
    monkeypatch.setattr(run, "TCNConsumerConfig", lambda **kwargs: tcn(**(kwargs | {"epochs": 1, "hidden_channels": 8, "device": device})))
    if device == "cpu":
        monkeypatch.setattr(run, "_require_diagnostic_device", lambda seed: None)
        loader = run.load_common_pretraining_checkpoint
        monkeypatch.setattr(run, "load_common_pretraining_checkpoint", lambda path, **kwargs: loader(path, **(kwargs | {"device": "cpu"})))
    method = "chronaris" if candidate_name else "physiology_only"
    arguments = dict(domain="simulation", method=method, output_root=tmp_path, candidate_name=candidate_name)
    result = run.run_development_diagnostic(**arguments)
    assert result["completed"] and len(result["completed_consumers"]) == (2 if candidate_name else 6)
    assert result["self_supervised_training"]["optimizer_updates"] == 3
    assert result["task_guided_training"]["optimizer_updates"] == 5
    assert set(accessed) == set(batch.sample_ids)
    if candidate_name:
        parallel = run.run_development_diagnostic(**(arguments | {
            "output_root": tmp_path / "parallel", "prefetch_cpu_consumers": True}))
        assert parallel["completed"] and parallel["cpu_prefit_results"]["self_supervised:300"]["pid"] != __import__("os").getpid()
        from chronaris.modeling.training.rng import canonical_training_state_sha256
        for route in ("self_supervised", "task_guided"):
            original = torch.load(result[route + "_training"]["last_checkpoint_path"], weights_only=True)
            concurrent = torch.load(parallel[route + "_training"]["last_checkpoint_path"], weights_only=True)
            name = "encoder_state_dict" if route == "self_supervised" else "model_state_dict"
            assert canonical_training_state_sha256(original[name]) == canonical_training_state_sha256(concurrent[name])
            update = 300 if route == "self_supervised" else 200
            expected = json.loads((tmp_path / "simulation" / method / candidate_name / f"{route}_{update}_consumers.json").read_text())
            actual = json.loads((tmp_path / "parallel/simulation" / method / candidate_name / f"{route}_{update}_consumers.json").read_text())
            assert expected["metric_rows"] == actual["metric_rows"]
            if route == "self_supervised":
                assert actual["component_status"]["linear"] == actual["component_status"]["minirocket"] == "resumed"
        assert actual["component_status"]["minirocket"] == "completed"  # Guided consumer remains freshly fitted.
        for seed, route in ((29, "self_supervised"), (43, "task_guided")):
            review = run.run_development_diagnostic(**(arguments | {
                "output_root": tmp_path / "review", "phase": "review", "seed": seed, "routes": (route,)}))
            assert review["completed"] and len(review["completed_consumers"]) == 1
            payload = torch.load(review[route + "_training"]["last_checkpoint_path"], weights_only=True)
            assert payload["config"]["seed"] == seed
            if route == "self_supervised":
                assert "task_guided_training" not in review
                assert review["completed_consumers"] == ["self_supervised:1500"]
            else:
                assert review["completed_consumers"] == ["task_guided:500"]
        pre = next(config for stage, config in scheduled if stage == "pretraining" and config["seed"] == 29)
        guided = next(config for stage, config in scheduled if stage == "guidance" and config["seed"] == 43)
        assert (pre["max_updates"], pre["minimum_updates"], pre["patience"], pre["early_stopping"]) == (1500, 500, 5, True)
        assert (guided["max_updates"], guided["minimum_updates"], guided["patience"], guided["head_warmup_updates"]) == (500, 200, 4, 50)
    root = tmp_path / "simulation" / method
    if candidate_name:
        root = root / candidate_name
    for route, supervised in (("self_supervised", False), ("task_guided", True)):
        update = (300 if route == "self_supervised" else 200) if candidate_name else 3
        record = json.loads((root / f"{route}_{update}_consumers.json").read_text())
        assert record["model_manifest"]["label_used_for_encoder_training"] is supervised
        assert record["model_manifest"]["evaluation_roles"] == ["validation"]
        assert set(record["component_status"]) == {"linear", "minirocket", "causal_tcn"}
    def no_retraining(**kwargs):
        pytest.fail("completed learning curves should not retrain a consumer")
    monkeypatch.setattr(run, "run_application_method_consumers", no_retraining)
    resumed = run.run_development_diagnostic(**arguments)
    assert resumed["completed"]
    if device == "cpu":
        _check_pressure_pipeline(tmp_path, monkeypatch, inputs, targets, fold, method=method, candidate_name=candidate_name)


def _check_pressure_pipeline(root, monkeypatch, inputs, targets, fold, *, method="physiology_only", candidate_name=None):
    from chronaris.evaluation.application_tasks import v4_pressure_run as pressure
    from chronaris.evaluation.application_tasks.v4_correctness import remove_future_observations
    from chronaris.evaluation.application_tasks.application_consumers import LinearFrozenConsumer, MiniRocketFrozenConsumer
    monkeypatch.setattr(pressure, "_require_diagnostic_device", lambda seed: None)
    monkeypatch.setattr(pressure, "CURVE_UPDATES", (1, 2, 3))
    monkeypatch.setattr(pressure, "load_development_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(pressure, "build_guarded_application_consumer_targets", lambda *args, **kwargs: targets)
    loader = pressure._load_development_encoder
    monkeypatch.setattr(pressure, "_load_development_encoder", lambda path, **kwargs: loader(path, **(kwargs | {"device": "cpu"})))
    validation = inputs[0](fold.validation_sample_ids)
    def observations(*, condition, **kwargs):
        changed = validation if condition == "clean_asynchronous" else remove_future_observations(validation, cutoff_s=-1.)
        return SimpleNamespace(batch=changed, sample_manifest_rows=[{"sample_id": sample, "profile_id": "validation_profile"}
                                                                  for sample in changed.sample_ids])
    monkeypatch.setattr(pressure, "load_development_condition", observations)
    def no_fit(*args, **kwargs):
        pytest.fail("pressure evaluation must not refit a consumer")
    monkeypatch.setattr(LinearFrozenConsumer, "fit", no_fit)
    monkeypatch.setattr(MiniRocketFrozenConsumer, "fit", no_fit)
    condition_root = root / "conditions"
    condition_root.mkdir()
    (condition_root / "development_condition_audit.json").write_text("{}")
    for route in ("self_supervised", "task_guided"):
        update = (300 if route == "self_supervised" else 200) if candidate_name else 3
        kwargs = dict(method=method, route=route, update=update, output_root=root / "pressure",
                      diagnostic_root=root, condition_root=condition_root, candidate_name=candidate_name, device="cpu")
        result = pressure.run_development_pressure(**kwargs)
        assert result["completed"] and len(result["conditions"]) == 8
        missing = json.loads(Path(result["conditions"]["contiguous_gap_30s"]["result_path"]).read_text())
        representation_root = Path(result["conditions"]["contiguous_gap_30s"]["result_path"]).parent / "representation"
        metadata = json.loads((representation_root / "representation_manifest.json").read_text())
        assert metadata["label_used_for_encoder_training"] is (route == "task_guided")
        assert missing["grouped"]["all_windows_retained"] and missing["grouped"]["no_observation_count"] == 3
        assert all(row["role"] == "validation" for row in missing["evaluation"]["metric_rows"])
        if candidate_name:
            assert result["source"]["candidate_options"]["name"] == candidate_name
            assert missing["encoding_diagnostics"]["quality_gate_enabled"]
        assert pressure.run_development_pressure(**kwargs)["completed"]
        if route == "task_guided":
            path = Path(result["conditions"]["contiguous_gap_30s"]["prediction_path"])
            path.write_bytes(path.read_bytes() + b"changed")
            with pytest.raises(ValueError, match="saved pressure result changed"):
                pressure.run_development_pressure(**kwargs)


def test_native_diagnostic_cpu_contract_runs_shared_training_and_grouped_consumers(tmp_path, monkeypatch):
    from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
    samples = [_sample(f"native_{i}", shift=i / 10) for i in range(9)]
    batch = collate_observation_samples(samples)
    fold = FoldLineage("native_diagnostic", batch.sample_ids[:6], batch.sample_ids[6:], (), development_only=True)
    def provider(ids):
        assert set(ids) <= set(batch.sample_ids)
        return select_observation_batch(batch, ids)
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids,
                                                held_out_sample_ids=fold.validation_sample_ids)
    definitions = (ApplicationTaskDefinition("classify", "classification", 3), ApplicationTaskDefinition("response", "regression", 1))
    values = {"classify": torch.arange(9).remainder(3), "response": torch.linspace(0., 1., 9)}
    masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}
    masks["response"][2] = False
    targets = ApplicationTaskTargets(batch.sample_ids, values, masks, {"domain": "clare"})
    data = SimpleNamespace(sample_manifest=[{"sample_id": sample, "subject_id": "train" if i < 6 else "validation"}
                                          for i, sample in enumerate(batch.sample_ids)])
    inputs = (provider, samples[0].schema, fold, {s: (s,) for s in fold.train_sample_ids}, "a" * 64, targets, definitions, data)
    monkeypatch.setattr(run, "load_development_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(run, "development_normalization", lambda *args, **kwargs: (normalizer, None))
    monkeypatch.setattr(run, "_require_diagnostic_device", lambda seed: None)
    monkeypatch.setattr(run, "CURVE_UPDATES", (1, 2, 3))
    pretraining, guidance = run.CandidateScreenConfig, run.EndToEndFineTuningConfig
    checkpoint_loader, consumer_runner = run.load_common_pretraining_checkpoint, run.run_native_method_consumers
    monkeypatch.setattr(run, "CandidateScreenConfig", lambda **kwargs: pretraining(**(kwargs | {"effective_batch_size": 4, "device": "cpu"})))
    monkeypatch.setattr(run, "EndToEndFineTuningConfig", lambda **kwargs: guidance(**(kwargs | {
        "effective_batch_size": 4, "head_warmup_updates": 2, "validation_interval": 1, "device": "cpu"})))
    monkeypatch.setattr(run, "load_common_pretraining_checkpoint", lambda path, **kwargs: checkpoint_loader(path, **(kwargs | {"device": "cpu"})))
    monkeypatch.setattr(run, "run_native_method_consumers", lambda **kwargs: consumer_runner(**kwargs, minirocket_kernels=84))
    result = run.run_development_diagnostic(domain="clare", method="physiology_only", output_root=tmp_path)
    assert result["completed"] and len(result["completed_consumers"]) == 6
    assert result["self_supervised_training"]["optimizer_updates"] == 3
    assert result["task_guided_training"]["optimizer_updates"] == 5
    for route, labels in (("self_supervised", False), ("task_guided", True)):
        root = tmp_path / "clare/physiology_only/fold01/consumers" / route / "3"
        manifest = json.loads((root / "consumer_manifest.json").read_text())
        assert manifest["label_used_for_encoder_training"] is labels
        assert set(manifest["roles"]) == {"train", "validation"}
        for family in ("linear", "minirocket"):
            evidence = json.loads((root / f"{family}_results.json").read_text())
            assert evidence["evaluations"]["validation"]["independent_unit"] == "subject"
            assert len(evidence["fit_rows"][1]["train_sample_ids"]) == 5
    assert run.run_development_diagnostic(domain="clare", method="physiology_only", output_root=tmp_path)["completed"]
    monkeypatch.setattr(run, "CandidateScreenConfig", lambda **kwargs: pretraining(**(kwargs | {
        "max_updates": 2, "effective_batch_size": 4, "device": "cpu"})))
    monkeypatch.setattr(run, "EndToEndFineTuningConfig", lambda **kwargs: guidance(**(kwargs | {
        "max_updates": 1, "head_warmup_updates": 1, "effective_batch_size": 4, "device": "cpu"})))
    for route in ("self_supervised", "task_guided"):
        screened = run.run_development_diagnostic(domain="clare", method="physiology_only",
            output_root=tmp_path / "public_screen" / route, candidate_name="reference", routes=(route,))
        assert screened["completed"] and len(screened["completed_consumers"]) == 1
        assert screened["self_supervised_training"]["optimizer_updates"] == 2
        assert ("task_guided_training" in screened) is (route == "task_guided")
        if route == "task_guided":
            assert screened["task_guided_training"]["optimizer_updates"] == 2
