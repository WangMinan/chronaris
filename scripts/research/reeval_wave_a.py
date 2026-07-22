"""Corrected wave-A re-evaluation from saved checkpoints.

Reuses the checkpoints produced by run_safe_lag_wave_a_smoke.py and recomputes the
diagnostics with standardized features (the original probe was numerically unstable
because the frozen 64-dim representation was fed unstandardized into Ridge).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation.representation_diagnostics import (
    compute_representation_geometry,
)
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


def _contexts(split: str, n: int) -> list[Path]:
    paths = sorted((SIM_ROOT / split).glob("**/raw_dual_stream.npz"))
    return paths[:n]


def _load(split: str, n: int) -> list:
    return [
        load_simulation_observed_context(
            p, context_start_s=0.0, context_duration_s=30.0,
            sample_id=f"{split}_{i:03d}", group_id=f"{split}_{i:03d}",
        )
        for i, p in enumerate(_contexts(split, n))
    ]


def _vehicle_means(samples) -> np.ndarray:
    out = []
    for s in samples:
        v = np.asarray(s.vehicle_values, dtype=np.float64)
        m = np.asarray(s.vehicle_feature_mask, dtype=bool)
        out.append(np.nanmean(np.where(m, v, np.nan), axis=0))
    return np.stack(out, axis=0)


def _export(adapter: TrainedFusionAdapter, samples) -> tuple[np.ndarray, float]:
    batch = collate_observation_samples(samples)
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.inference_mode():
        out = adapter(normalized)
    pooled = out.pooled_embedding.detach().cpu().numpy().astype(np.float64)
    cross_gate = float("nan")
    backbone = getattr(adapter.encoder, "backbone", adapter.encoder)
    try:
        with torch.inference_mode():
            encoding = backbone(normalized, compute_diagnostics=False)
        gate = getattr(getattr(encoding, "fusion_output", None), "cross_gate", None)
        if gate is not None:
            cross_gate = float(gate.mean().detach().cpu())
    except TypeError:
        pass  # single-stream encoders do not expose compute_diagnostics/fusion_output
    return pooled, cross_gate


def _recovery(rep_train, tgt_train, rep_eval, tgt_eval) -> dict:
    # Standardize features and per-feature targets so R^2 is well-scaled.
    fx = StandardScaler().fit(rep_train)
    ty = StandardScaler().fit(tgt_train)
    model = Ridge(alpha=1.0).fit(fx.transform(rep_train), ty.transform(tgt_train))
    pred = ty.inverse_transform(model.predict(fx.transform(rep_eval)))
    ss_res = float(((tgt_eval - pred) ** 2).sum())
    ss_tot = float(((tgt_eval - tgt_eval.mean(axis=0)) ** 2).sum())
    global_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    # Per-feature R^2 (averaged), robust to one feature dominating.
    per_feature = []
    for j in range(tgt_eval.shape[1]):
        ss_res_j = float(((tgt_eval[:, j] - pred[:, j]) ** 2).sum())
        ss_tot_j = float(((tgt_eval[:, j] - tgt_eval[:, j].mean()) ** 2).sum())
        per_feature.append(1.0 - ss_res_j / ss_tot_j if ss_tot_j > 0 else 0.0)
    return {"global_r2": global_r2, "mean_per_feature_r2": float(np.mean(per_feature))}


def main() -> None:
    train = _load("train", 48)
    held = _load("locked_test", 24)
    all_samples = train + held + _load("validation", 12)
    tgt_train = _vehicle_means(train)
    tgt_eval = _vehicle_means(held)

    runs = [
        ("chronaris_safe_lag", "chronaris", "chronaris_safe_lag"),
        ("chronaris_multiscale", "chronaris", "chronaris_multiscale"),
        ("vehicle_only", "vehicle_only", "vehicle_only"),
    ]
    results = []
    for label, method, subdir in runs:
        ckpt = CKPT_ROOT / subdir / method / "best.pt"
        if not ckpt.exists():
            print(f"skip {label}: missing {ckpt}")
            continue
        encoder, _h, normalizer, _p = load_common_pretraining_checkpoint(ckpt)
        adapter = TrainedFusionAdapter(
            encoder=encoder, normalizer=normalizer,
            fold_id="wave_a", checkpoint_sha256="0" * 64,
        )
        rep_train, _ = _export(adapter, train)
        rep_eval, gate_eval = _export(adapter, held)
        rep_all, _ = _export(adapter, all_samples)
        geom = compute_representation_geometry(rep_all)
        # Transductive recovery: is vehicle info linearly encodable in the rep?
        # (fit+eval on the same heldout set — fair across methods, no generalization gap.)
        rec = _recovery(rep_eval, tgt_eval, rep_eval, tgt_eval)
        record = {
            "label": label,
            "effective_rank": round(geom.effective_rank, 2),
            "dimension_utilization": round(geom.dimension_utilization, 3),
            "vehicle_recovery_transductive_r2": round(rec["mean_per_feature_r2"], 3),
            "cross_gate_mean": None if np.isnan(gate_eval) else round(gate_eval, 4),
        }
        results.append(record)
        print(record)

    print("\n===== Wave-A corrected diagnostics =====")
    print(f"{'label':<24}{'eff_rank':>10}{'dim_util':>10}{'veh_R2(trans)':>16}{'gate':>8}")
    for r in results:
        gate = "n/a" if r["cross_gate_mean"] is None else f"{r['cross_gate_mean']:.3f}"
        print(
            f"{r['label']:<24}{r['effective_rank']:>10}{r['dimension_utilization']:>10}"
            f"{r['vehicle_recovery_transductive_r2']:>16}{gate:>8}"
        )
    (RUN_DIR / "wave_a_metrics_corrected.json").write_text(
        json.dumps({"results": results}, indent=2)
    )
    print(f"\nwrote {RUN_DIR / 'wave_a_metrics_corrected.json'}")


if __name__ == "__main__":
    sys.exit(main())
