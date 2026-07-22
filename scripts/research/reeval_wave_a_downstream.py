"""Wave-A downstream probe: future-maneuver-intensity prediction.

Uses the wave-A checkpoints (no retraining) to test the core negative-transfer
hypothesis on a vehicle-dominated downstream target: predict future-5s maneuver
intensity (angular rates + accelerations over [30,35]s) from the 30s context
representation. vehicle_only is the reference; safe_lag should approach it while the
old multiscale fusion should lag — the same pattern as the Dingxin maneuver task
(vehicle 0.808 vs old Chronaris 0.195).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.common_pretraining import (
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import collate_observation_samples
from chronaris.representation.loaders import load_simulation_observed_context

REPO = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
CKPT_ROOT = REPO / "artifacts/application_evaluation/2026-07-22_safe-lag-wave-a-smoke/checkpoints"
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-22_safe-lag-wave-a-smoke"

MANEUVER_CHANNELS = (6, 7, 8, 9, 10)  # roll/pitch/yaw rate + longitudinal/lateral acc


def _contexts(split: str, n: int) -> list[Path]:
    return sorted((SIM_ROOT / split).glob("**/raw_dual_stream.npz"))[:n]


def _load(split: str, n: int) -> list:
    return [
        load_simulation_observed_context(
            p, context_start_s=0.0, context_duration_s=30.0,
            sample_id=f"{split}_{i:03d}", group_id=f"{split}_{i:03d}",
        )
        for i, p in enumerate(_contexts(split, n))
    ]


def _future_maneuver_intensity(paths) -> np.ndarray:
    """Mean L2 norm of maneuver channels over the future [30,35]s window per sortie."""
    targets = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        vt = d["vehicle_observed_time_s"]
        vv = np.asarray(d["vehicle_values"], dtype=np.float64)
        future = (vt >= 30.0) & (vt <= 35.0)
        if future.sum() == 0:
            targets.append(np.nan)
            continue
        window = vv[future][:, MANEUVER_CHANNELS]
        targets.append(float(np.linalg.norm(window, axis=1).mean()))
    return np.asarray(targets, dtype=np.float64)


def _export(adapter: TrainedFusionAdapter, samples) -> np.ndarray:
    batch = collate_observation_samples(samples)
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.inference_mode():
        out = adapter(normalized)
    return out.pooled_embedding.detach().cpu().numpy().astype(np.float64)


def _probe(rep_train, y_train, rep_eval, y_eval) -> dict:
    fx = StandardScaler().fit(rep_train)
    model = Ridge(alpha=1.0).fit(fx.transform(rep_train), y_train)
    pred = model.predict(fx.transform(rep_eval))
    ss_res = float(((y_eval - pred) ** 2).sum())
    ss_tot = float(((y_eval - y_eval.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rho = float(spearmanr(y_eval, pred).correlation) if len(y_eval) > 2 else 0.0
    return {"r2": round(r2, 3), "spearman": round(rho, 3)}


def main() -> None:
    train_paths = _contexts("train", 48)
    held_paths = _contexts("locked_test", 24)
    train_samples = _load("train", 48)
    held_samples = _load("locked_test", 24)
    y_train = _future_maneuver_intensity(train_paths)
    y_eval = _future_maneuver_intensity(held_paths)
    valid_train = np.isfinite(y_train)
    valid_eval = np.isfinite(y_eval)

    runs = [
        ("chronaris_safe_lag", "chronaris", "chronaris_safe_lag"),
        ("chronaris_multiscale", "chronaris", "chronaris_multiscale"),
        ("vehicle_only", "vehicle_only", "vehicle_only"),
    ]
    results = []
    for label, method, subdir in runs:
        ckpt = CKPT_ROOT / subdir / method / "best.pt"
        encoder, _h, normalizer, _p = load_common_pretraining_checkpoint(ckpt)
        adapter = TrainedFusionAdapter(
            encoder=encoder, normalizer=normalizer,
            fold_id="wave_a", checkpoint_sha256="0" * 64,
        )
        rep_train = _export(adapter, train_samples)
        rep_eval = _export(adapter, held_samples)
        # generalization (train->heldout) and transductive (heldout->heldout)
        gen = _probe(
            rep_train[valid_train], y_train[valid_train],
            rep_eval[valid_eval], y_eval[valid_eval],
        )
        tra = _probe(
            rep_eval[valid_eval], y_eval[valid_eval],
            rep_eval[valid_eval], y_eval[valid_eval],
        )
        record = {"label": label, "generalization": gen, "transductive": tra}
        results.append(record)
        print(record)

    print("\n===== Wave-A downstream: future-maneuver-intensity prediction =====")
    print(f"{'label':<24}{'gen_R2':>9}{'gen_rho':>9}{'trans_R2':>11}{'trans_rho':>11}")
    for r in results:
        print(
            f"{r['label']:<24}{r['generalization']['r2']:>9}{r['generalization']['spearman']:>9}"
            f"{r['transductive']['r2']:>11}{r['transductive']['spearman']:>11}"
        )
    (RUN_DIR / "wave_a_downstream_maneuver.json").write_text(
        json.dumps({"target": "future_maneuver_intensity_30_35s", "results": results}, indent=2)
    )
    print(f"\nwrote {RUN_DIR / 'wave_a_downstream_maneuver.json'}")


if __name__ == "__main__":
    sys.exit(main())
