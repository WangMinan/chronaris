"""Read-only review checks for main 207f14a0; run from the repository root.

Writes only the adjacent JSON. Model interventions affect in-memory copies;
no training or public/Dingxin outer evaluation is performed.
"""

import json
import math
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import torch
import numpy as np
import joblib

from chronaris.dataset.clare_native import build_clare_native_dataset
from chronaris.dataset.cogpilot_native import build_cogpilot_difficulty_dataset
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.evaluation.application_tasks.thesis_native_data import CLARE_ROOT, COGPILOT_ROOT
from chronaris.evaluation.application_tasks.thesis_simulation_gates import (
    PRETRAINING, SIMULATION_ROOT, _adapter,
)
from chronaris.modeling.fusion_encoders.multiscale_causal import MultiScaleCausalFusionInput
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.representation import build_batch_augmentation_realizations, select_observation_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def checkpoint_inventory():
    rows = []
    for path in sorted(PRETRAINING.glob("checkpoints/**/last.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        states = payload["optimizer_state_dict"]["state"].values()
        adam_steps = sorted({int(state["step"]) for state in states if "step" in state})
        losses = payload["epoch_rows"]
        rows.append({
            "path": str(path.relative_to(Path.cwd())),
            "sha256": sha256_file(path),
            "method": payload["method_name"], "seed": payload["seed"],
            "steps": payload["step_count"], "adam_steps": adam_steps,
            "epochs": payload["completed_epochs"], "best_epoch": payload["best_epoch"],
            "train_samples": len(payload["fold"]["train_sample_ids"]),
            "batch_size": payload["config"]["batch_size"],
            "parameters": payload["parameter_count"],
            "elapsed_s": payload["training_elapsed_s"],
            "selection_loss_epoch40": losses[39]["public_selection_loss"],
            "selection_loss_epoch50": losses[49]["public_selection_loss"],
        })
    assert len(rows) == 15
    assert all(row["steps"] == 50 and row["adam_steps"] == [50] for row in rows)
    return rows


def data_inventory():
    result = {}
    for name, builder, root, limits in (
        ("cogpilot", build_cogpilot_difficulty_dataset, COGPILOT_ROOT, (20, 1000)),
        ("clare", build_clare_native_dataset, CLARE_ROOT, (16, 1000)),
    ):
        rows = []
        for limit in limits:
            dataset = builder(root, subject_limit=limit, max_memory_cache_bytes=0)
            rows.append({
                "subject_limit": limit, "records": len(dataset.records),
                "subjects": len(set(dataset.group_ids)),
                "subject_ids": sorted(set(dataset.group_ids)),
            })
        result[name] = {"root": str(root), "inventory": rows,
                        "scope": "metadata candidates; not all windows signal-validated"}
    plans = build_batch_augmentation_realizations(
        tuple(f"review_{i}" for i in range(1000)), epoch=1, global_seed=17,
    )
    result["short_window_augmentation"] = {
        str(duration): sum(p.physiology_block_start_s >= duration for p in plans)
        for duration in (10, 12, 30)
    }
    manifest = json.loads(Path("docs/artifacts/runs/2026-07-10_dingxin-input-snapshot/raw_snapshot_manifest.json").read_text())
    snapshot = Path(manifest["snapshot_root"])
    result["dingxin"] = {"sorties": len(manifest["allowed_sortie_ids"]),
                         "views": len(manifest["plans"]), "files": []}
    for row in manifest["files"]:
        path = snapshot / row["relative_path"]
        assert sha256_file(path) == row["sha256"]
        result["dingxin"]["files"].append({"path": str(path), "sha256": row["sha256"],
                                             "points": row["point_count"]})
    return result


def model_checks():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    data = load_simulation_locked_pretraining_data(SIMULATION_ROOT)
    raw = select_observation_batch(data.batch, data.fold.validation_sample_ids[:4])
    results = []
    for seed in (17, 29, 43):
        path = PRETRAINING / f"checkpoints/seed_{seed}/chronaris/best.pt"
        before = sha256_file(path)
        adapter = _adapter(path, data.fold.fold_id, "cuda")
        backbone = adapter.encoder.backbone
        adapter.encoder.eval()
        def evaluate(batch):
            normalized = move_observation_batch(adapter.normalizer.transform(batch), device="cuda")
            with torch.inference_mode():
                return adapter.encoder(normalized, compute_chronaris_diagnostics=True)
        captured = []
        hook = backbone.register_forward_hook(lambda _m, _a, output: captured.append(output))
        baseline = evaluate(raw)
        encoding = captured[-1]
        attention_rows = []
        for weights, mask in zip(encoding.fusion_output.attention_weights,
                                 encoding.fusion_output.lag_masks, strict=True):
            count = mask.sum(-1)
            valid = count > 1
            entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
            ratio = weights.max(-1).values / weights.masked_fill(~mask, float("inf")).min(-1).values
            attention_rows.append({
                "mean_normalized_entropy": float((entropy[valid] / count[valid].float().log()).mean()),
                "max_weight_ratio": float(ratio[valid].max()),
                "query_count": int(valid.sum()),
            })
        residual_before = {c.component_name: float(c.raw_value) for c in encoding.physics_audit.components if c.available}
        decoder = backbone.continuous_backbone.vehicle_stream.decoder
        saved = {k: v.clone() for k, v in decoder.state_dict().items()}
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.zero_()
        zeroed = evaluate(raw)
        zero_encoding = captured[-1]
        residual_zero = {c.component_name: float(c.raw_value) for c in zero_encoding.physics_audit.components if c.available}
        delta = float((baseline.sequence_embedding - zeroed.sequence_embedding).abs().max())
        assert len(residual_zero) >= 2 and delta == 0 and all(value == 0 for value in residual_zero.values())
        decoder.load_state_dict(saved)
        normalized = move_observation_batch(adapter.normalizer.transform(raw), device="cuda")
        from chronaris.modeling.fusion_encoders.alignment_bridge import build_alignment_batch_from_observations
        alignment_batch = build_alignment_batch_from_observations(
            normalized,
            physiology_feature_names=backbone.config.physiology_feature_names,
            vehicle_feature_names=backbone.config.vehicle_feature_names,
        )
        target = alignment_batch.vehicle.values
        valid = alignment_batch.vehicle.feature_valid_mask & alignment_batch.vehicle.mask.unsqueeze(-1)
        reconstruction = encoding.alignment_output.vehicle.reconstructions
        observed_mse = float((reconstruction[valid] - target[valid]).square().mean())
        zero_mse = float(target[valid].square().mean())
        permutation_rows = {}
        if seed == 17:
            for stream in ("physiology", "vehicle"):
                updates = {f.name: getattr(raw, f.name).roll(1, dims=0)
                           for f in fields(raw) if f.name.startswith(stream + "_")
                           and isinstance(getattr(raw, f.name), torch.Tensor)}
                evaluate(replace(raw, **updates))
                other = captured[-1].semantic_event_output
                original = encoding.semantic_event_output
                permutation_rows[stream] = {
                    name: float((other.query_context_states[:, i] - original.query_context_states[:, i]).abs().max())
                    for i, name in enumerate(original.query_names)
                }
            fusion = backbone.causal_fusion
            shape = (1, 4, backbone.config.hidden_dim)
            with torch.inference_mode():
                missing = fusion(MultiScaleCausalFusionInput(
                    physiology_states=torch.zeros(shape, device="cuda"),
                    vehicle_states=torch.ones(shape, device="cuda"),
                    physiology_valid_mask=torch.zeros((1, 4), dtype=torch.bool, device="cuda"),
                    vehicle_valid_mask=torch.ones((1, 4), dtype=torch.bool, device="cuda"),
                    query_timestamps_s=torch.arange(4, device="cuda").float()[None],
                ))
            missing_branch = {
                "unobserved_physiology_branch_max": float(missing.sequence_embedding[..., :24].abs().max()),
                "no_valid_cross_key_branch_max": float(missing.sequence_embedding[..., 48:].abs().max()),
            }
        else:
            missing_branch = None
        hook.remove()
        row = {"seed": seed, "checkpoint_sha256": before,
               "attention": attention_rows, "attention_ratio_bound": math.exp(2 / math.sqrt(backbone.config.hidden_dim)),
               "physics_before": residual_before, "physics_zero_head": residual_zero,
               "zero_head_representation_delta": delta,
               "decoder_mse_normalized_units": observed_mse, "zero_decoder_mse_normalized_units": zero_mse,
               "event_valid_token_counts": encoding.semantic_event_output.event_token_mask.sum(-1).tolist(),
               "stream_permutation_context_delta": permutation_rows,
               "single_stream_missing": missing_branch,
               "checkpoint_unchanged": sha256_file(path) == before}
        assert row["checkpoint_unchanged"]
        results.append(row)
        print(json.dumps(row), flush=True)
    return {"device": "cuda", "sample_ids": list(raw.sample_ids), "rows": results}


def tail_checks():
    root = Path("artifacts/application_evaluation")
    rows = []
    for seed in (17, 29, 43):
        model_path = root / f"2026-09-03_thesis-simulation-consumers-v3p2p1/consumers/seed_{seed}/chronaris/linear.joblib"
        representation_path = root / f"2026-09-04_thesis-simulation-stress-representations-v3p2p3/representations/seed_{seed}/chronaris/contiguous_gap_30s/fusion_stream.npz"
        consumer = joblib.load(model_path)["consumer"].regressor
        scaler, regressor = consumer.named_steps["standardscaler"], consumer.named_steps["ridge"]
        with np.load(representation_path) as values:
            pooled = values["pooled_embedding"]
        standardized = scaler.transform(pooled)
        contributions = standardized * regressor.coef_
        assert np.allclose(contributions.sum(1) + regressor.intercept_, consumer.predict(pooled), rtol=1e-5, atol=1e-5)
        rows.append({"seed": seed, "samples": len(pooled),
                     "consumer_sha256": sha256_file(model_path),
                     "representation_sha256": sha256_file(representation_path),
                     "blocks": {
                         name: {"min_train_std": float(scaler.scale_[start:stop].min()),
                                "max_abs_standardized": float(np.abs(standardized[:, start:stop]).max()),
                                "max_abs_prediction_contribution": float(np.abs(contributions[:, start:stop].sum(1)).max())}
                         for name, start, stop in (("physiology", 0, 24), ("vehicle", 24, 48), ("cross", 48, 64))
                     }})
    return rows


if __name__ == "__main__":
    result = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": sha256_file(__file__), "scope": "read_only_diagnostic_not_training_or_model_selection"}
    result["checkpoints"] = checkpoint_inventory()
    print("Checkpoint inventory complete", flush=True)
    result["datasets"] = data_inventory()
    print(json.dumps(result["datasets"]), flush=True)
    result["models"] = model_checks()
    result["tail"] = tail_checks()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
