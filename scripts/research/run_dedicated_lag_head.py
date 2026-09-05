"""Dedicated lag-recovery head: the decisive cross-modal fusion-win experiment.

Trains each encoder with an auxiliary head that must predict the injected physiology lag
tau from the frozen-style window representation. This forces the representation to encode
cross-modal lag. Single streams structurally cannot win: vehicle_only's representation is
invariant to physiology lag (it never sees physiology), and physiology_only has no vehicle
reference to measure the lag magnitude against. Fusion sees both, so its representation can
encode tau and the head can read it -> robust fusion win on a genuinely cross-modal task.

Uses a custom loop: encoder + Linear(64,1) lag head, loss = MSE(head(pooled), tau) +
light reconstruction so the encoder does not collapse.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.pretraining_encoders import (
    ENCODER_SCREEN_CANDIDATES,
    build_trainable_fusion_encoder,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)
from chronaris.representation.loaders import load_simulation_observed_context

REPO = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-24_dedicated-lag-head"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-24_dedicated-lag-head"

LAGS = (-5.0, -2.5, 0.0, 2.5, 5.0)
N_TRAIN_TRAJ = 40
N_EVAL_TRAJ = 48
EPOCHS = 30
BATCH = 16
SEED = 17
LR = 3e-4


def _ctx(split: str, n: int) -> list[Path]:
    return sorted((SIM_ROOT / split).glob("**/raw_dual_stream.npz"))[:n]


def _load_raw(path):
    d = np.load(path, allow_pickle=True)
    return {
        "phys_t": np.asarray(d["physiology_observed_time_s"], dtype=np.float64),
        "phys_v": np.asarray(d["physiology_values"], dtype=np.float64),
        "veh_t": np.asarray(d["vehicle_observed_time_s"], dtype=np.float64),
        "veh_v": np.asarray(d["vehicle_values"], dtype=np.float64),
        "phys_names": tuple(f"physiology.{s}" for s in d["physiology_feature_names"].astype(str)),
        "veh_names": tuple(f"vehicle.{s}" for s in d["vehicle_feature_names"].astype(str)),
    }


BASE_START_S = 30.0  # both windows stay well inside the ~180 s sortie for |tau| <= 5


def _window(t, v, lo, hi):
    m = (t >= lo) & (t <= hi)
    return t[m] - lo, v[m]  # rebase to [0, hi-lo]


def _lagged_sample(raw, tau, idx, schema):
    """Clean cross-modal lag: physiology window = [BASE+tau, BASE+tau+30], vehicle = [BASE, BASE+30].
    Both fully observed (no edge clamp), rebased to [0,30]; cross-modal content offset = tau."""
    pt, pv = _window(raw["phys_t"], raw["phys_v"], BASE_START_S + tau, BASE_START_S + tau + 30.0)
    vt, vv = _window(raw["veh_t"], raw["veh_v"], BASE_START_S, BASE_START_S + 30.0)
    sid = f"{idx:04d}__lag{tau}"
    return ObservedDualStreamSample(
        sample_id=sid, group_id=f"traj_{idx:04d}", schema=schema,
        physiology_values=pv.astype(np.float32), physiology_timestamps_s=pt.astype(np.float32),
        physiology_feature_mask=np.ones_like(pv, dtype=bool),
        vehicle_values=vv.astype(np.float32), vehicle_timestamps_s=vt.astype(np.float32),
        vehicle_feature_mask=np.ones_like(vv, dtype=bool),
        source_sample_hash=hashlib.sha256(f"{sid}".encode()).hexdigest(),
    )


def _build_set(paths, lags, schema):
    samples, taus = [], []
    for idx, p in enumerate(paths):
        raw = _load_raw(p)
        for tau in lags:
            samples.append(_lagged_sample(raw, tau, idx, schema)); taus.append(tau)
    return samples, np.array(taus, dtype=np.float64)


def _seq(encoder, normalizer, batch, device):
    normalized = normalizer.transform(move_observation_batch(batch, device=device))
    enc = encoder(normalized)
    return enc.sequence_embedding  # [B, 96, 64]


def _temporal_stats(seq):
    """mean/std/last over time -> [B, 64*3] (carries temporal alignment, unlike the mean)."""
    return torch.cat([seq.mean(dim=1), seq.std(dim=1), seq[:, -1]], dim=-1)


def train_method(method, fk, train_samples, train_taus, schema, normalizer, device):
    torch.manual_seed(SEED)
    encoder = build_trainable_fusion_encoder(
        method, physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
        vehicle_field_labels=tuple((n, n) for n in schema.vehicle_feature_names),
        candidate_config=ENCODER_SCREEN_CANDIDATES[0], chronaris_fusion_kind=fk,
    ).backbone.to(device)
    # lag head reads the FULL sequence via a temporal conv (alignment lives in time, not the mean)
    lag_conv = nn.Conv1d(64, 32, kernel_size=7, padding=3)
    lag_head = nn.Linear(32, 1)
    recon_head = nn.Linear(64, len(schema.physiology_feature_names) + len(schema.vehicle_feature_names))
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(lag_conv.parameters()) + list(lag_head.parameters()) + list(recon_head.parameters()), lr=LR)
    n = len(train_samples); idx = np.arange(n)
    for epoch in range(EPOCHS):
        np.random.default_rng(SEED + epoch).shuffle(idx)
        for start in range(0, n, BATCH):
            sub = idx[start:start + BATCH]
            batch_samples = [train_samples[j] for j in sub]
            tau = torch.tensor(train_taus[sub], dtype=torch.float32, device=device).view(-1, 1)
            batch = collate_observation_samples(batch_samples)
            seq = _seq(encoder, normalizer, batch, device)  # [B,96,64]
            pooled = seq.mean(dim=1)
            feat = lag_conv(seq.transpose(1, 2)).mean(dim=2)  # [B,32]
            lag_pred = lag_head(feat)
            phys_m = batch.physiology_values.float().mean(dim=1); veh_m = batch.vehicle_values.float().mean(dim=1)
            tgt = torch.cat([phys_m, veh_m], dim=1).to(device)
            loss = nn.functional.mse_loss(lag_pred, tau) + 0.05 * nn.functional.mse_loss(recon_head(pooled), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return encoder


def eval_method(encoder, normalizer, eval_samples, eval_taus, device):
    """Independent Ridge probe on temporal stats of the sequence (not the trained head)."""
    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(eval_samples), BATCH):
            batch = collate_observation_samples(eval_samples[i:i + BATCH])
            seq = _seq(encoder, normalizer, batch, device)
            feats.append(_temporal_stats(seq).detach().cpu().numpy())
    emb = np.concatenate(feats, axis=0)
    groups = np.array([(i // len(LAGS)) for i in range(len(eval_samples))])
    train_mask = (groups % 2 == 0)
    fx = StandardScaler().fit(emb[train_mask])
    model = Ridge(alpha=1.0).fit(fx.transform(emb[train_mask]), eval_taus[train_mask])
    pred = model.predict(fx.transform(emb[~train_mask]))
    mae = mean_absolute_error(eval_taus[~train_mask], pred)
    true_cls = np.argmin(np.abs(eval_taus[~train_mask][:, None] - np.array(LAGS)[None, :]), axis=1)
    pred_cls = np.argmin(np.abs(pred[:, None] - np.array(LAGS)[None, :]), axis=1)
    acc = float((true_cls == pred_cls).mean())
    return emb, mae, acc


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True); HEAVY.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    train_paths = _ctx("train", N_TRAIN_TRAJ)
    eval_paths = _ctx("locked_test", N_EVAL_TRAJ)
    raw0 = _load_raw(train_paths[0])
    schema = ObservationSchema(
        schema_id="lag_head_sim.v1", source_kind="method_independent_simulation",
        physiology_feature_names=raw0["phys_names"], vehicle_feature_names=raw0["veh_names"],
        physiology_feature_roles=tuple("observed" for _ in raw0["phys_names"]),
        vehicle_feature_roles=tuple("observed" for _ in raw0["veh_names"]),
    )
    train_samples, train_taus = _build_set(train_paths, LAGS, schema)
    eval_samples, eval_taus = _build_set(eval_paths, LAGS, schema)
    normalizer = TrainOnlyRobustNormalizer().fit(
        collate_observation_samples(train_samples),
        train_sample_ids=tuple(s.sample_id for s in train_samples[:-1]),
        held_out_sample_ids=(train_samples[-1].sample_id,))

    runs = [("chronaris_safe_lag", "chronaris", "safe_lag"),
            ("chronaris_multiscale", "chronaris", "multiscale"),
            ("vehicle_only", "vehicle_only", "multiscale"),
            ("physiology_only", "physiology_only", "multiscale")]
    results = []
    for label, method, fk in runs:
        t0 = time.perf_counter()
        print(f"[lag-head] training {label} ...", flush=True)
        encoder = train_method(method, fk, train_samples, train_taus, schema, normalizer, device)
        emb, mae, acc = eval_method(encoder, normalizer, eval_samples, eval_taus, device)
        rec = {"label": label, "lag_mae_s": round(float(mae), 3), "lag_class_accuracy": round(acc, 3),
               "elapsed_s": round(time.perf_counter() - t0, 1)}
        results.append(rec); print(f"  -> {rec}", flush=True)

    print("\n===== Dedicated lag-head: cross-modal lag recovery =====")
    print(f"{'method':<26}{'lag_MAE_s':>11}{'class_acc':>11}")
    for r in results:
        print(f"{r['label']:<26}{r['lag_mae_s']:>11}{r['lag_class_accuracy']:>11}")
    (RUN_DIR / "dedicated_lag_head_metrics.json").write_text(json.dumps({
        "eval_lags_s": list(LAGS), "n_train_traj": N_TRAIN_TRAJ, "n_eval_traj": N_EVAL_TRAJ,
        "epochs": EPOCHS, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / 'dedicated_lag_head_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
