"""Cross-modal time-lag recovery: the task that genuinely requires both streams.

Reuses the wave-A simulation checkpoints (which were trained with the common
lag_discrimination pretext, so their representations encode cross-modal lag). For each
held-out G1 trajectory we create physiology time-shifted copies (shift physiology
relative to vehicle by a known lag tau) and ask each frozen representation to recover tau
via a Ridge probe. Single-stream methods cannot see cross-modal lag (shifting physiology
does not change vehicle_only's representation, and physiology_only has no vehicle
reference), so this is the clean task where fusion must beat single streams.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from chronaris.modeling.training.common_pretraining import (
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import collate_observation_samples
from chronaris.representation.loaders import load_simulation_observed_context

REPO = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
CKPT_ROOT = REPO / "artifacts/application_evaluation/2026-07-22_safe-lag-wave-a-smoke/checkpoints"
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-23_cross-modal-lag-recovery"

LAGS_S = (-5.0, -2.0, 0.0, 2.0, 5.0)
N_TRAJ = 24


def _contexts(split: str, n: int) -> list[Path]:
    return sorted((SIM_ROOT / split).glob("**/raw_dual_stream.npz"))[:n]


def _lagged_sample(base, tau: float, idx: int):
    """Return a copy of base with the physiology SIGNAL delayed by tau seconds.

    Value-domain lag (interpolate physiology onto (t - tau)) keeps timestamps inside the
    valid context window while making physiology lag vehicle by tau — physically
    equivalent to a physiology time-shift. vehicle is unchanged.
    """
    phys_t = np.asarray(base.physiology_timestamps_s, dtype=np.float64)
    phys_v = np.asarray(base.physiology_values, dtype=np.float64)
    delayed_t = phys_t - tau  # value at sample-time t comes from original time (t - tau)
    lagged_v = np.empty_like(phys_v)
    for c in range(phys_v.shape[1]):
        lagged_v[:, c] = np.interp(delayed_t, phys_t, phys_v[:, c])
    return base.__class__(
        sample_id=f"{base.sample_id}__lag{tau}_{idx}",
        group_id=base.group_id,
        schema=base.schema,
        physiology_values=lagged_v.astype(np.float32),
        physiology_timestamps_s=base.physiology_timestamps_s,
        physiology_feature_mask=base.physiology_feature_mask,
        vehicle_values=base.vehicle_values,
        vehicle_timestamps_s=base.vehicle_timestamps_s,
        vehicle_feature_mask=base.vehicle_feature_mask,
        source_sample_hash=hashlib.sha256(f"{base.sample_id}:{tau}".encode()).hexdigest(),
    )


def _export(adapter, batch):
    adapter.encoder.eval()
    with torch.no_grad():
        out = adapter(batch)
    return out.pooled_embedding.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    base_samples = [
        load_simulation_observed_context(p, context_start_s=0.0, context_duration_s=30.0,
                                         sample_id=f"traj_{i:03d}", group_id=f"traj_{i:03d}")
        for i, p in enumerate(_contexts("locked_test", N_TRAJ))
    ]
    # build lagged copies
    lagged, tau_labels, traj_groups = [], [], []
    for i, base in enumerate(base_samples):
        for j, tau in enumerate(LAGS_S):
            lagged.append(_lagged_sample(base, tau, i))
            tau_labels.append(tau)
            traj_groups.append(i)
    tau_labels = np.array(tau_labels); traj_groups = np.array(traj_groups)
    # LOSO by trajectory: even traj -> train, odd -> test
    train_mask = (traj_groups % 2 == 0)
    batch = collate_observation_samples(lagged)

    runs = [
        ("chronaris_safe_lag", "chronaris", "chronaris_safe_lag"),
        ("chronaris_multiscale", "chronaris", "chronaris_multiscale"),
        ("vehicle_only", "vehicle_only", "vehicle_only"),
        ("physiology_only", "physiology_only", "physiology_only"),
    ]
    results = []
    for label, method, subdir in runs:
        ckpt = CKPT_ROOT / subdir / method / "best.pt"
        if not ckpt.exists():
            print(f"skip {label}: {ckpt} missing"); continue
        enc, _h, ln, _p = load_common_pretraining_checkpoint(ckpt)
        adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="lag", checkpoint_sha256="0" * 64)
        emb = _export(adapter, batch)
        tr, te = train_mask, ~train_mask
        fx = StandardScaler().fit(emb[tr])
        model = Ridge(alpha=1.0).fit(fx.transform(emb[tr]), tau_labels[tr])
        pred = model.predict(fx.transform(emb[te]))
        mae = mean_absolute_error(tau_labels[te], pred)
        # lag-class accuracy (nearest LAGS_S)
        true_cls = np.argmin(np.abs(tau_labels[te][:, None] - np.array(LAGS_S)[None, :]), axis=1)
        pred_cls = np.argmin(np.abs(pred[:, None] - np.array(LAGS_S)[None, :]), axis=1)
        acc = float((true_cls == pred_cls).mean())
        rec = {"label": label, "lag_mae_s": round(float(mae), 3), "lag_class_accuracy": round(acc, 3)}
        results.append(rec)
        print(f"  {label}: lag_MAE={rec['lag_mae_s']}s  class_acc={rec['lag_class_accuracy']}", flush=True)

    print("\n===== Cross-modal lag recovery (single streams cannot do this) =====")
    print(f"{'method':<24}{'lag_MAE_s':>11}{'class_acc':>11}")
    for r in results:
        print(f"{r['label']:<24}{r['lag_mae_s']:>11}{r['lag_class_accuracy']:>11}")
    (RUN_DIR / "lag_recovery_metrics.json").write_text(json.dumps({
        "lags_s": list(LAGS_S), "n_trajectories": N_TRAJ, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / 'lag_recovery_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
