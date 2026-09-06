from dataclasses import replace
from types import SimpleNamespace
import json

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_diagnostic_run as run
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import ApplicationConsumerSmokeTargets
from chronaris.evaluation.application_tasks.application_task_heads import SIMULATION_TASKS
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples, select_observation_batch
from tests.representation.test_contracts import _sample


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diagnostic integration")
def test_learning_curve_runs_real_trainers_exports_and_all_consumers_without_confirmation(tmp_path, monkeypatch):
    samples = [_sample(f"sample_{i}", shift=i / 10) for i in range(9)]
    batch = collate_observation_samples(samples)
    roles = dict(train=batch.sample_ids[:6], validation=batch.sample_ids[6:], held_out=("sealed",))
    fold = FoldLineage("diagnostic_test", roles["train"], roles["validation"], roles["held_out"])
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
    monkeypatch.setattr(run, "CandidateScreenConfig", lambda **kwargs: pretraining(**(kwargs | {"effective_batch_size": 4})))
    monkeypatch.setattr(run, "EndToEndFineTuningConfig", lambda **kwargs: guidance(**(kwargs | {
        "effective_batch_size": 4, "head_warmup_updates": 2, "validation_interval": 1})))
    monkeypatch.setattr(run, "MiniRocketConsumerConfig", lambda **kwargs: rocket(**kwargs, n_kernels=84))
    monkeypatch.setattr(run, "TCNConsumerConfig", lambda **kwargs: tcn(**(kwargs | {"epochs": 1, "hidden_channels": 8})))
    result = run.run_simulation_diagnostic(method="physiology_only", output_root=tmp_path)
    assert result["completed"] and len(result["completed_consumers"]) == 6
    assert result["self_supervised_training"]["optimizer_updates"] == 3
    assert result["task_guided_training"]["optimizer_updates"] == 5
    assert set(accessed) == set(batch.sample_ids)
    root = tmp_path / "simulation/physiology_only"
    for route, supervised in (("self_supervised", False), ("task_guided", True)):
        record = json.loads((root / f"{route}_3_consumers.json").read_text())
        assert record["model_manifest"]["label_used_for_encoder_training"] is supervised
        assert record["model_manifest"]["evaluation_roles"] == ["validation"]
        assert set(record["component_status"]) == {"linear", "minirocket", "causal_tcn"}
    def no_retraining(**kwargs):
        pytest.fail("completed learning curves should not retrain a consumer")
    monkeypatch.setattr(run, "run_application_method_consumers", no_retraining)
    resumed = run.run_simulation_diagnostic(method="physiology_only", output_root=tmp_path)
    assert resumed["completed"]


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
