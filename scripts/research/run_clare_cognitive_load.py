"""CLARE cognitive-load: central-EEG vs peripheral fusion (auxiliary public dataset).

Cognitive load reflects BOTH central EEG activity (alpha/theta) AND peripheral arousal
(EDA/HR). The genuinely cross-modal test: does fusing central + peripheral beat either
single stream? Streams: central = EEG amplitude envelope (4 ch), peripheral = EDA
conductance + ECG-derived HR (2 ch). LOSO by subject. Clocks aligned (EDA/ECG clock P
offset from EEG clock N).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.stats import spearmanr

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.common_pretraining import (
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)
from chronaris.representation.augmentation import AugmentationPolicy
from chronaris.representation.lineage import FoldLineage

REPO = Path(__file__).resolve().parents[2]
CLARE = Path("/home/wangminan/dataset/chronaris/clare")
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-24_clare-cognitive-load"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-24_clare-cognitive-load"

WIN_S = 10.0
FS = 10
N_PT = int(WIN_S * FS)
N_SUBJECTS = 16
EPOCHS = 8
BATCH = 8


def _seed() -> int:
    return int(sys.argv[1]) if len(sys.argv) > 1 else 17


SEED = _seed()
CENTRAL_NAMES = ("central.eeg_tp9", "central.eeg_af7", "central.eeg_af8", "central.eeg_tp10")
PERIPH_NAMES = ("peripheral.eda", "peripheral.hr")
SCHEMA = ObservationSchema(schema_id="clare_cogload.v1", source_kind="clare_public",
    physiology_feature_names=CENTRAL_NAMES, vehicle_feature_names=PERIPH_NAMES,
    physiology_feature_roles=tuple("observed" for _ in CENTRAL_NAMES),
    vehicle_feature_roles=tuple("observed" for _ in PERIPH_NAMES))


def _rms_envelope(t, x, q):
    """Per-0.1s-bin RMS of each column -> envelope on grid q (seconds), always finite."""
    x = np.where(np.isfinite(x), x, 0.0)
    out = np.zeros((len(q), x.shape[1]))
    for c in range(x.shape[1]):
        edges = np.append(q, q[-1] + (q[1] - q[0]))
        vals = np.zeros(len(q))
        for i in range(len(q)):
            m = (t >= edges[i]) & (t < edges[i + 1])
            vals[i] = np.sqrt(np.mean(x[m, c] ** 2)) if m.sum() else np.nan
        s = pd.Series(vals).interpolate().bfill().ffill()
        s = s.fillna(0.0)
        out[:, c] = s.to_numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _hr(t, ecg, q):
    thr = np.nanpercentile(ecg, 90)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1e-3
    pk, _ = find_peaks(ecg, height=thr, distance=max(1, int(0.4 / max(dt, 1e-6))))
    if len(pk) < 3:
        return np.full(len(q), 80.0)
    rr_t = 0.5 * (t[pk][1:] + t[pk][:-1]); rr = np.diff(t[pk])
    hr = np.clip(60.0 / np.maximum(rr, 1e-3), 30.0, 200.0)
    return np.interp(q, rr_t, hr)


def _interp_col(t, x, q):
    out = np.zeros(len(q))
    m = np.isfinite(x)
    return np.interp(q, t[m], x[m]) if m.sum() > 1 else out


def build_samples():
    samples, labels, groups = [], [], []
    subs = sorted(CLARE.glob("EEG/[0-9]*"))[:N_SUBJECTS]
    for sub in subs:
        sid = sub.name
        lab_path = CLARE / "Labels" / f"{sid}.csv"
        if not lab_path.exists():
            continue
        lab = pd.read_csv(lab_path)
        for k in range(4):
            eeg_f = CLARE / "EEG" / sid / f"eeg_data_exp_{k}.csv"
            eda_f = CLARE / "EDA" / sid / f"eda_data_experiment_{k}.csv"
            ecg_f = CLARE / "ECG" / sid / f"ecg_data_experiment_{k}.csv"
            if not (eeg_f.exists() and eda_f.exists() and ecg_f.exists()):
                continue
            de = pd.read_csv(eeg_f); dd = pd.read_csv(eda_f); dc = pd.read_csv(ecg_f)
            eeg_t = de["Timestamp"].to_numpy(float); eeg_x = de[["TP9", "AF7", "AF8", "TP10"]].to_numpy(float)
            eda_t = dd["Timestamp"].to_numpy(float); eda_x = dd["GSR Conductance CAL"].to_numpy(float)
            ecg_t = dc["Timestamp"].to_numpy(float); ecg_x = dc["ECG LL-RA CAL"].to_numpy(float)
            off = eda_t[0] - eeg_t[0]  # peripheral clock offset vs EEG
            level_col = f"level_{k}"
            if level_col not in lab.columns:
                continue
            for i, labval in enumerate(lab[level_col].tolist()):
                if i % 2 != 0:  # stride-2 cap windows for tractability
                    continue
                if not np.isfinite(labval):
                    continue
                lo, hi = i * WIN_S, (i + 1) * WIN_S  # session-time (EEG clock)
                q = np.linspace(lo, hi, N_PT + 1)[:N_PT]  # EEG grid
                central = _rms_envelope(eeg_t, eeg_x, q)
                central = np.nan_to_num(central, nan=0.0, posinf=0.0, neginf=0.0)
                # peripheral in its own clock: session-time T -> periph time (T - off)
                qp = q - off
                eda_v = _interp_col(eda_t, eda_x, qp).reshape(-1, 1)
                hr_v = _hr(ecg_t, ecg_x, qp).reshape(-1, 1)
                periph = np.concatenate([eda_v, hr_v], axis=1)
                periph = np.nan_to_num(periph, nan=0.0, posinf=0.0, neginf=0.0)
                sample_id = f"{sid}__s{k}_w{i}"
                samples.append(ObservedDualStreamSample(
                    sample_id=sample_id, group_id=sid, schema=SCHEMA,
                    physiology_values=central.astype(np.float32),
                    physiology_timestamps_s=((q - lo).astype(np.float32)),
                    physiology_feature_mask=np.ones_like(central, dtype=bool),
                    vehicle_values=periph.astype(np.float32),
                    vehicle_timestamps_s=((q - lo).astype(np.float32)),
                    vehicle_feature_mask=np.ones_like(periph, dtype=bool),
                    source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest()))
                labels.append(int(labval)); groups.append(sid)
    return samples, np.array(labels), np.array(groups)


def _export(adapter, batch):
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.no_grad():
        o = adapter(normalized)
    return o.pooled_embedding.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True); HEAVY.mkdir(parents=True, exist_ok=True)
    print("[clare] building samples...", flush=True)
    samples, labels, groups = build_samples()
    print(f"[clare] samples={len(samples)} subjects={len(set(groups))} label_dist={np.bincount(labels)[1:].tolist() if labels.max()>=1 else []}", flush=True)
    uniq = sorted(set(groups)); test_subs = set(uniq[::4])
    train_idx = [i for i, g in enumerate(groups) if g not in test_subs]
    test_idx = [i for i, g in enumerate(groups) if g in test_subs]
    print(f"[clare] train={len(train_idx)} test={len(test_idx)}", flush=True)

    batch = collate_observation_samples(samples)
    fold = FoldLineage(fold_id="clare_loso", train_sample_ids=tuple(samples[i].sample_id for i in train_idx),
                       validation_sample_ids=(), held_out_sample_ids=tuple(samples[i].sample_id for i in test_idx))
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids, held_out_sample_ids=fold.held_out_sample_ids)
    veh_labels = tuple((n, n) for n in PERIPH_NAMES)

    # binary: low (<=6) vs high (>=7) for a balanced LOSO classification
    ybin = (labels >= 7).astype(int)
    runs = [("fusion_safe_lag", "chronaris", "safe_lag"),
            ("fusion_multiscale", "chronaris", "multiscale"),
            ("central_only", "physiology_only", "multiscale"),
            ("peripheral_only", "vehicle_only", "multiscale")]
    results = []
    for label, method, fk in runs:
        t0 = time.perf_counter()
        print(f"[clare] training {label} ...", flush=True)
        res = train_common_pretext_method(
            method, batch=batch, fold=fold, physiology_feature_names=CENTRAL_NAMES,
            vehicle_feature_names=PERIPH_NAMES, vehicle_field_labels=veh_labels,
            normalizer=normalizer, output_root=str(HEAVY / "checkpoints" / f"seed{SEED}" / label),
            config=CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED),
            augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk, resume=False)
        enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
        adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="clare", checkpoint_sha256="0" * 64)
        emb = _export(adapter, batch)
        Xtr = StandardScaler().fit_transform(emb[train_idx]); Xte = StandardScaler().fit_transform(emb[test_idx])
        # regression (Spearman) on raw 1-9
        rg = Ridge(alpha=10.0).fit(Xtr, labels[train_idx])
        pred_r = rg.predict(Xte)
        rho = float(spearmanr(labels[test_idx], pred_r).correlation)
        # binary low/high
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED).fit(Xtr, ybin[train_idx])
        pred_b = clf.predict(Xte)
        f1 = f1_score(ybin[test_idx], pred_b, average="macro")
        ba = balanced_accuracy_score(ybin[test_idx], pred_b)
        rec = {"label": label, "spearman": round(rho, 3), "macro_f1": round(f1, 3), "bal_acc": round(ba, 3), "elapsed_s": round(time.perf_counter() - t0, 1)}
        results.append(rec); print(f"  -> {rec}", flush=True)

    print("\n===== CLARE cognitive load (LOSO, public auxiliary) =====")
    print(f"{'method':<22}{'spearman':>10}{'macro_f1':>10}{'bal_acc':>9}")
    for r in results:
        print(f"{r['label']:<22}{r['spearman']:>10}{r['macro_f1']:>10}{r['bal_acc']:>9}")
    (RUN_DIR / f"clare_metrics_seed{SEED}.json").write_text(json.dumps({
        "n_samples": len(samples), "n_subjects": len(set(groups)), "test_subjects": sorted(test_subs),
        "seed": SEED, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / f'clare_metrics_seed{SEED}.json'}")


if __name__ == "__main__":
    sys.exit(main())
