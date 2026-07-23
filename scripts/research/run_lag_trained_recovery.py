"""Lag-trained cross-modal recovery: the decisive fusion-win experiment.

The diagnostic (run_lag_recovery.py) showed single streams are at chance on physiology-lag
recovery (the task genuinely needs both streams), but wave-A checkpoints (trained on clean
data) do not encode lag. Here we train chronaris safe_lag WITH the lag_aware loss on
physiology-lag-augmented G1 data, so the encoder must represent cross-modal lag, then
evaluate lag recovery on held-out trajectories. Single streams remain at chance; safe_lag
should beat them — a genuine fusion win on a task that structurally requires both streams.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.common_pretraining import (
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import TrainOnlyRobustNormalizer, collate_observation_samples
from chronaris.representation.augmentation import AugmentationPolicy
from chronaris.representation.lineage import FoldLineage
from chronaris.representation.loaders import load_simulation_observed_context

REPO = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-23_cross-modal-lag-recovery"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-23_cross-modal-lag-recovery"

TRAIN_LAGS = (-5.0, -2.5, 0.0, 2.5, 5.0)  # random per-copy during training augmentation
EVAL_LAGS = (-5.0, -2.0, 0.0, 2.0, 5.0)
N_TRAIN_TRAJ = 32
N_EVAL_TRAJ = 48
EPOCHS = 20
SEED = 17


def _ctx(split: str, n: int, start: int = 0) -> list[Path]:
    return sorted((SIM_ROOT / split).glob("**/raw_dual_stream.npz"))[start : start + n]


def _lagged(base, tau, idx):
    phys_t = np.asarray(base.physiology_timestamps_s, dtype=np.float64)
    phys_v = np.asarray(base.physiology_values, dtype=np.float64)
    delayed_t = phys_t - tau
    lagged_v = np.stack(
        [np.interp(delayed_t, phys_t, phys_v[:, c]) for c in range(phys_v.shape[1])], axis=1
    )
    return base.__class__(
        sample_id=f"{base.sample_id}__lag{tau}_{idx}", group_id=base.group_id, schema=base.schema,
        physiology_values=lagged_v.astype(np.float32), physiology_timestamps_s=base.physiology_timestamps_s,
        physiology_feature_mask=base.physiology_feature_mask, vehicle_values=base.vehicle_values,
        vehicle_timestamps_s=base.vehicle_timestamps_s, vehicle_feature_mask=base.vehicle_feature_mask,
        source_sample_hash=hashlib.sha256(f"{base.sample_id}:{tau}:{idx}".encode()).hexdigest(),
    )


def _export(adapter, batch):
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.no_grad():
        out = adapter(normalized)
    return out.pooled_embedding.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True); HEAVY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # training set: each train trajectory x 4 random-lag copies
    train_base = [load_simulation_observed_context(p, context_start_s=0.0, context_duration_s=30.0,
                                                   sample_id=f"train_{i:03d}", group_id=f"train_{i:03d}")
                  for i, p in enumerate(_ctx("train", N_TRAIN_TRAJ))]
    train_samples = []
    for i, base in enumerate(train_base):
        for k in range(4):
            tau = float(rng.choice(TRAIN_LAGS))
            train_samples.append(_lagged(base, tau, i * 10 + k))
    schema = train_base[0].schema
    batch = collate_observation_samples(train_samples)
    fold = FoldLineage(fold_id="lag_train",
                       train_sample_ids=tuple(s.sample_id for s in train_samples[:-1]),
                       validation_sample_ids=(),
                       held_out_sample_ids=(train_samples[-1].sample_id,))
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids,
                                                 held_out_sample_ids=fold.held_out_sample_ids)
    veh_labels = tuple((n, n) for n in schema.vehicle_feature_names)

    runs = [("chronaris_safe_lag", "chronaris", "safe_lag", 0.0),
            ("chronaris_multiscale", "chronaris", "multiscale", 0.0),
            ("vehicle_only", "vehicle_only", "multiscale", 0.0),
            ("physiology_only", "physiology_only", "multiscale", 0.0)]
    adapters = {}
    for label, method, fk, lagw in runs:
        t0 = time.perf_counter()
        print(f"[lag-train] training {label} (fusion={fk}, lag_aware_weight={lagw}) ...", flush=True)
        res = train_common_pretext_method(
            method, batch=batch, fold=fold, physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=veh_labels,
            normalizer=normalizer, output_root=str(HEAVY / "checkpoints" / label),
            config=CommonPretrainingConfig(epochs=EPOCHS, batch_size=8, seed=SEED),
            augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk,
            chronaris_lag_aware_weight=lagw, resume=False,
        )
        enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
        adapters[label] = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="lag", checkpoint_sha256="0" * 64)
        print(f"  trained in {time.perf_counter()-t0:.1f}s", flush=True)

    # eval: held-out trajectories x fixed lags
    eval_base = [load_simulation_observed_context(p, context_start_s=0.0, context_duration_s=30.0,
                                                  sample_id=f"eval_{i:03d}", group_id=f"eval_{i:03d}")
                 for i, p in enumerate(_ctx("locked_test", N_EVAL_TRAJ))]
    eval_samples, tau_labels, groups = [], [], []
    for i, base in enumerate(eval_base):
        for tau in EVAL_LAGS:
            eval_samples.append(_lagged(base, tau, i)); tau_labels.append(tau); groups.append(i)
    tau_labels = np.array(tau_labels); groups = np.array(groups)
    eval_batch = collate_observation_samples(eval_samples)
    train_mask = (groups % 2 == 0)

    results = []
    for label, adapter in adapters.items():
        emb = _export(adapter, eval_batch)
        tr, te = train_mask, ~train_mask
        fx = StandardScaler().fit(emb[tr])
        model = Ridge(alpha=1.0).fit(fx.transform(emb[tr]), tau_labels[tr])
        pred = model.predict(fx.transform(emb[te]))
        mae = mean_absolute_error(tau_labels[te], pred)
        true_cls = np.argmin(np.abs(tau_labels[te][:, None] - np.array(EVAL_LAGS)[None, :]), axis=1)
        pred_cls = np.argmin(np.abs(pred[:, None] - np.array(EVAL_LAGS)[None, :]), axis=1)
        acc = float((true_cls == pred_cls).mean())
        rec = {"label": label, "lag_mae_s": round(float(mae), 3), "lag_class_accuracy": round(acc, 3)}
        results.append(rec); print(f"  {label}: lag_MAE={rec['lag_mae_s']}s class_acc={rec['lag_class_accuracy']}", flush=True)

    print("\n===== Lag-trained cross-modal recovery =====")
    print(f"{'method':<34}{'lag_MAE_s':>11}{'class_acc':>11}")
    for r in results:
        print(f"{r['label']:<34}{r['lag_mae_s']:>11}{r['lag_class_accuracy']:>11}")
    (RUN_DIR / "lag_trained_recovery_metrics.json").write_text(json.dumps({
        "eval_lags_s": list(EVAL_LAGS), "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / 'lag_trained_recovery_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
