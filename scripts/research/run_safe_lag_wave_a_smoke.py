"""Wave-A smoke: does safe-lag fusion preserve vehicle information?

Trains Chronaris (safe_lag) vs Chronaris (multiscale) vs vehicle-only on a small G1
simulation subset and reports three diagnostics that the audit flagged as the failure
mode of the original fusion:

  * effective rank of the frozen window representation (dimension collapse, audit Q8);
  * vehicle-information linear recovery R^2 from the representation (single-stream
    fidelity, audit Q1/Q2 — the catastrophic-loss fix);
  * cross-gate mean for the safe-lag variant (safe-fallback opening, audit Q7).

Read-only on the original simulation data; writes only to a compact run dir under
docs/artifacts/runs. Heavy checkpoints stay under an ignored directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from chronaris.evaluation.representation_diagnostics import (
    compute_representation_geometry,
)
from chronaris.modeling.training.common_pretraining import (
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.representation import (
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)
from chronaris.representation.lineage import FoldLineage
from chronaris.representation.loaders import load_simulation_observed_context

REPO = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-22_safe-lag-wave-a-smoke"
HEAVY_ROOT = REPO / "artifacts/application_evaluation/2026-07-22_safe-lag-wave-a-smoke"

N_TRAIN = 48
N_VAL = 12
N_HELDOUT = 12
EPOCHS = 15
BATCH = 8
SEED = 17


def _profile_contexts(split: str, n: int) -> list[Path]:
    # Collect observation scenarios directly (each profile has 6 trajectories).
    paths = sorted((SIM_ROOT / split).glob("**/raw_dual_stream.npz"))
    if len(paths) < n:
        raise RuntimeError(f"only {len(paths)} {split} contexts available, need {n}")
    return paths[:n]


def _load_samples(paths: list[Path], role: str) -> list:
    samples = []
    for index, path in enumerate(paths):
        samples.append(
            load_simulation_observed_context(
                path, context_start_s=0.0, context_duration_s=30.0,
                sample_id=f"{role}_{index:03d}", group_id=f"{role}_{index:03d}",
            )
        )
    return samples


def _vehicle_window_means(samples) -> np.ndarray:
    means = []
    for sample in samples:
        values = np.asarray(sample.vehicle_values, dtype=np.float64)
        mask = np.asarray(sample.vehicle_feature_mask, dtype=bool)
        valid = np.where(mask, values, np.nan)
        means.append(np.nanmean(valid, axis=0))
    return np.stack(means, axis=0)


def _export(adapter: TrainedFusionAdapter, samples) -> tuple[np.ndarray, np.ndarray]:
    batch = collate_observation_samples(samples)
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.inference_mode():
        out = adapter(batch)
    pooled = out.pooled_embedding.detach().cpu().numpy().astype(np.float64)
    # cross-gate (safe-lag only): run the backbone and read the fusion output.
    cross_gate_mean = float("nan")
    backbone = getattr(adapter.encoder, "backbone", adapter.encoder)
    if hasattr(backbone, "causal_fusion") and hasattr(
        backbone.causal_fusion, "config"
    ):
        with torch.inference_mode():
            encoding = backbone(normalized, compute_diagnostics=False)
        fusion = getattr(encoding, "fusion_output", None)
        gate = getattr(fusion, "cross_gate", None)
        if gate is not None:
            cross_gate_mean = float(gate.mean().detach().cpu())
    return pooled, np.asarray(cross_gate_mean, dtype=np.float64)


def _recovery_r2(
    rep_train: np.ndarray, target_train: np.ndarray,
    rep_eval: np.ndarray, target_eval: np.ndarray,
) -> float:
    model = Ridge(alpha=1.0).fit(rep_train, target_train)
    pred = model.predict(rep_eval)
    ss_res = float(((target_eval - pred) ** 2).sum())
    ss_tot = float(((target_eval - target_eval.mean(axis=0)) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main() -> None:
    torch.manual_seed(SEED)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    HEAVY_ROOT.mkdir(parents=True, exist_ok=True)

    train_samples = _load_samples(_profile_contexts("train", N_TRAIN), "train")
    val_samples = _load_samples(_profile_contexts("validation", N_VAL), "validation")
    held_samples = _load_samples(_profile_contexts("locked_test", N_HELDOUT), "heldout")
    all_samples = train_samples + val_samples + held_samples
    batch = collate_observation_samples(all_samples)
    schema = train_samples[0].schema

    fold = FoldLineage(
        fold_id="wave_a_smoke",
        train_sample_ids=tuple(s.sample_id for s in train_samples),
        validation_sample_ids=tuple(s.sample_id for s in val_samples),
        held_out_sample_ids=tuple(s.sample_id for s in held_samples),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.held_out_sample_ids,
    )
    vehicle_labels = tuple((name, name) for name in schema.vehicle_feature_names)
    config = CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED)

    runs = [
        ("chronaris_safe_lag", "chronaris", "safe_lag"),
        ("chronaris_multiscale", "chronaris", "multiscale"),
        ("vehicle_only", "vehicle_only", "multiscale"),
    ]
    vehicle_train_target = _vehicle_window_means(train_samples)
    vehicle_eval_target = _vehicle_window_means(held_samples)

    results = []
    for label, method, fusion_kind in runs:
        print(f"[wave-a] training {label} ({method}, fusion={fusion_kind})...", flush=True)
        result = train_common_pretext_method(
            method,
            batch=batch,
            fold=fold,
            physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,
            vehicle_field_labels=vehicle_labels,
            normalizer=normalizer,
            output_root=str(HEAVY_ROOT / "checkpoints" / label),
            config=config,
            chronaris_fusion_kind=fusion_kind,
            resume=False,
        )
        encoder, _heads, loaded_normalizer, _payload = load_common_pretraining_checkpoint(
            result.best_checkpoint_path
        )
        adapter = TrainedFusionAdapter(
            encoder=encoder,
            normalizer=loaded_normalizer,
            fold_id=fold.fold_id,
            checkpoint_sha256="0" * 64,
        )
        rep_train, _ = _export(adapter, train_samples)
        rep_eval, cross_gate = _export(adapter, held_samples)
        geometry = compute_representation_geometry(rep_eval, seed=SEED)
        recovery = _recovery_r2(rep_train, vehicle_train_target, rep_eval, vehicle_eval_target)
        record = {
            "label": label,
            "method": method,
            "fusion_kind": fusion_kind,
            "status": result.status,
            "step_count": result.step_count,
            "parameter_count": result.parameter_count,
            "effective_rank": round(geometry.effective_rank, 3),
            "dimension_utilization": round(geometry.dimension_utilization, 3),
            "vehicle_recovery_r2": round(recovery, 3),
            "cross_gate_mean": None if np.isnan(cross_gate) else round(float(cross_gate), 4),
        }
        results.append(record)
        print(f"  -> {record}", flush=True)

    print("\n===== Wave-A smoke diagnostic table =====")
    print(f"{'label':<26} {'eff_rank':>9} {'dim_util':>9} {'veh_R2':>8} {'gate':>7}")
    for r in results:
        gate = "n/a" if r["cross_gate_mean"] is None else f"{r['cross_gate_mean']:.3f}"
        print(
            f"{r['label']:<26} {r['effective_rank']:>9} "
            f"{r['dimension_utilization']:>9} {r['vehicle_recovery_r2']:>8} {gate:>7}"
        )

    (RUN_DIR / "wave_a_metrics.json").write_text(
        json.dumps({"seed": SEED, "epochs": EPOCHS, "results": results}, indent=2)
    )
    print(f"\nwrote {RUN_DIR / 'wave_a_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
