"""Measured native-window execution equivalence; never opens confirmation roles."""
from pathlib import Path
import json
import statistics
import time

import torch

from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import perturb_future_observations
from chronaris.evaluation.application_tasks.v4_correctness import remove_future_observations
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.pretraining_encoders import EncoderCandidateConfig, build_trainable_fusion_encoder
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.modeling.training.sample_schedule import BalancedSampleSchedule
from chronaris.representation import TrainOnlyRobustNormalizer


def profile_native_recurrence(*, domain, data_root, registry_path, output_root):
    if not torch.cuda.is_available():
        raise ValueError("native neural profiling requires CUDA")
    torch.set_num_threads(1)
    root = Path(output_root) / domain
    root.mkdir(parents=True, exist_ok=True)
    with _periodic_training_heartbeat("native_recurrence", 30., root=root) as progress, isolated_training_rng(17):
        data = load_prepared_public_development(domain, output_root=data_root, registry_path=registry_path)
        subject_fold = json.loads(Path(registry_path).read_text())["domains"][domain]["folds"]["development"][0]
        fold = data.fold(subject_fold)
        ids = BalancedSampleSchedule(fold.train_sample_ids, data.sampling_hierarchy(fold)).draw(0, 4)
        raw = data.dataset.batch_provider(ids)
        normalizer = TrainOnlyRobustNormalizer().fit(raw, train_sample_ids=ids)
        batch = move_observation_batch(normalizer.transform(raw), device="cuda")
        args = dict(physiology_feature_names=data.dataset.schema.physiology_feature_names,
                    vehicle_feature_names=data.dataset.schema.vehicle_feature_names,
                    candidate_config=EncoderCandidateConfig(candidate_id="C", hidden_dim=32),
                    chronaris_fusion_kind="safe_lag", chronaris_semantic_event_enabled=True,
                    chronaris_learnable_semantic_queries=True)
        eager = build_trainable_fusion_encoder("chronaris", **args).cuda().eval()
        fast = build_trainable_fusion_encoder("chronaris", **args, chronaris_cuda_graph_recurrence=True).cuda().eval()
        fast.load_state_dict(eager.state_dict())
        results, gradients, sequences = {}, {}, {}
        for name, encoder in (("eager", eager), ("cuda_graph", fast)):
            progress["phase"] = name
            start = time.perf_counter()
            with torch.no_grad():
                encoder(batch)
            torch.cuda.synchronize()
            cold_s = time.perf_counter() - start
            forward_s = []
            with torch.no_grad():
                for _ in range(3):
                    start = time.perf_counter()
                    encoder(batch)
                    torch.cuda.synchronize()
                    forward_s.append(time.perf_counter() - start)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = encoder(batch)
            output.sequence_embedding.square().mean().backward()
            torch.cuda.synchronize()
            results[name] = {"cold_forward_s": cold_s, "forward_s": forward_s,
                "forward_median_s": statistics.median(forward_s),
                "forward_backward_s": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            sequences[name] = output.sequence_embedding.detach().clone()
            gradients[name] = {key: value.grad.detach().clone() for key, value in encoder.named_parameters() if value.grad is not None}
            del output
        if gradients["eager"].keys() != gradients["cuda_graph"].keys():
            raise AssertionError("graph execution disconnected parameters")
        output_delta = float((sequences["eager"] - sequences["cuda_graph"]).abs().max())
        gradient_delta = max(float((value - gradients["cuda_graph"][key]).abs().max()) for key, value in gradients["eager"].items())
        for key, expected in gradients["eager"].items():
            torch.testing.assert_close(gradients["cuda_graph"][key], expected, atol=1e-6, rtol=1e-5)
        historical = {}
        with torch.inference_mode():
            baseline = fast(batch)
            past = batch.query_timestamps_s <= 5.
            for name, changed in (("future_values", perturb_future_observations(batch, cutoff_s=5.)),
                                  ("future_missing", remove_future_observations(batch, cutoff_s=5.))):
                actual = fast(changed)
                assert torch.equal(baseline.modality_available_mask[past], actual.modality_available_mask[past])
                historical[name] = float((actual.sequence_embedding[past] - baseline.sequence_embedding[past]).abs().max())
        report = {"scope": "native_execution_equivalence_no_task_scores", "domain": domain,
            "device": torch.cuda.get_device_name(), "sample_ids": ids,
            "data_manifest_sha256": data.prepared_manifest_sha256, "normalizer_fit_ids": ids,
            "measurements": results, "output_max_delta": output_delta, "gradient_max_delta": gradient_delta,
            "historical_max_delta": historical,
            "forward_speedup": results["eager"]["forward_median_s"] / results["cuda_graph"]["forward_median_s"],
            "forward_backward_speedup": results["eager"]["forward_backward_s"] / results["cuda_graph"]["forward_backward_s"],
            "passed": output_delta <= 1e-6 and max(historical.values()) <= 1e-6}
        (root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if not report["passed"]:
            raise AssertionError("native graph output/history equivalence failed")
        return report
