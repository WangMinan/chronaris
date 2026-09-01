"""CogPilot event->response: the genuinely cross-modal real-data task.

Trigger (aircraft maneuver) lives in the vehicle stream; response (physiology change
after the event) lives in the physiology stream. Predicting the response needs BOTH the
trigger (vehicle) and the baseline (physiology). vehicle_only has the trigger but no
physiology baseline; physiology_only has the baseline but not the trigger; fusion has
both -> should beat each single stream. This is the non-persistent cross-modal task that
vehicle-dominated difficulty/maneuver tasks are not.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from chronaris.dataset.group_splits import split_group_train_validation
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
CP_ROOT = Path("/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-ils")
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-24_cogpilot-event-response"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-24_cogpilot-event-response"

CTX_S = 12.0          # pre-event context length
RESP_PRE_S = 2.0      # baseline window before event
RESP_POST_S = 8.0     # response window after event
MIN_GAP_S = 15.0      # min spacing between events in a run
FS = 10
N_PT = int(CTX_S * FS)
N_SUBJECTS = 20
EPOCHS = 8
BATCH = 8
SEED = 17

VEH_FILE = "lslxp11xpcac"
VEH_COLS = ("aircraft_indicated_airspeed_kias", "aircraft_pitch_deg", "aircraft_roll_deg",
            "aircraft_agl_altitude_m", "aircraft_climb_rate_mps",
            "aircraft_ils_deflection_gs", "aircraft_ils_deflection_h", "aircraft_velocity_u_mps")
PHYS_FILES = {"lslshimmereda": ("ppg_finger_mV", "eda_hand_l_kOhms"),
              "lslshimmerresp": ("respiration_trace_mV",)}
ECG_FILE, ECG_COL = "lslshimmerecg", "ecg_projection_ll_ra_mV"
PHYS_NAMES = ("physiology.ppg", "physiology.eda", "physiology.resp", "physiology.hr")
VEH_NAMES = tuple(f"vehicle.{c.replace('aircraft_', '')}" for c in VEH_COLS)


def _series(df, cols):
    t = (df.iloc[:, 0].to_numpy(float) - df.iloc[0, 0]) * 86400.0
    return t, df[list(cols)].to_numpy(float)


def _interp(t, x, q):
    out = np.zeros((len(q), x.shape[1]))
    for c in range(x.shape[1]):
        m = np.isfinite(x[:, c])
        out[:, c] = np.interp(q, t[m], x[m, c]) if m.sum() > 1 else 0.0
    return out


def _hr(t, ecg):
    thr = np.nanpercentile(ecg, 90)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1e-3
    pk, _ = find_peaks(ecg, height=thr, distance=max(1, int(0.4 / max(dt, 1e-6))))
    if len(pk) < 3:
        return None
    rr_t = 0.5 * (t[pk][1:] + t[pk][:-1]); rr = np.diff(t[pk])
    return rr_t, np.clip(60.0 / np.maximum(rr, 1e-3), 30.0, 200.0)


def _resample_grid(lo, hi):
    return np.linspace(lo, hi, int((hi - lo) * FS) + 1)


def build_events():
    """Return list of (sample, response_delta, subject_id). Response = EDA rise after event."""
    schema = ObservationSchema(schema_id="cogpilot_event.v1", source_kind="cogpilot_public",
        physiology_feature_names=PHYS_NAMES, vehicle_feature_names=VEH_NAMES,
        physiology_feature_roles=tuple("observed" for _ in PHYS_NAMES),
        vehicle_feature_roles=tuple("observed" for _ in VEH_NAMES))
    out = []
    subs = sorted(CP_ROOT.glob("sub-cp*"))[:N_SUBJECTS]
    for sub in subs:
        for run in sorted(sub.glob("ses-*/level-*_run-*")):
            try:
                d_veh = pd.read_csv(next(run.glob(f"*stream-{VEH_FILE}*_dat.csv")))
                d_ecg = pd.read_csv(next(run.glob(f"*stream-{ECG_FILE}*_dat.csv")))
                phys_arrs = []
                for tok, cols in PHYS_FILES.items():
                    dfp = pd.read_csv(next(run.glob(f"*stream-{tok}*_dat.csv")))
                    phys_arrs.append(_series(dfp, cols))
                tv, veh = _series(d_veh, VEH_COLS)
                te, ecg = _series(d_ecg, (ECG_COL,))
                # events: roll-rate spikes (maneuvers)
                roll = veh[:, VEH_COLS.index("aircraft_roll_deg")]
                m = np.isfinite(roll) & np.isfinite(tv)
                tv2, roll2 = tv[m], roll[m]
                rrate = np.abs(np.gradient(roll2, tv2))
                thr = np.nanpercentile(rrate, 85)
                pk, _ = find_peaks(rrate, height=thr, distance=int(MIN_GAP_S / np.median(np.diff(tv2))))
                # cap to top-3 strongest events per run (tractability + diversity)
                if len(pk) > 3:
                    top = np.argsort(rrate[pk])[-3:]
                    pk = np.sort(pk[top])
                evt_t = tv2[pk]
                hr = _hr(te, ecg[:, 0])
                teda, eda = phys_arrs[0][0], phys_arrs[0][1][:, list(PHYS_FILES["lslshimmereda"]).index("eda_hand_l_kOhms")]
                for j, t_e in enumerate(evt_t):
                    if t_e < CTX_S + 2 or t_e > tv[-1] - (RESP_POST_S + 2):
                        continue
                    q_ctx = _resample_grid(t_e - CTX_S, t_e)
                    pchunks = [_interp(a[0], a[1], q_ctx) for a in phys_arrs]
                    if hr:
                        pchunks.append(np.interp(q_ctx, hr[0], hr[1]).reshape(-1, 1))
                    else:
                        pchunks.append(np.full((len(q_ctx), 1), 80.0))
                    phys_ctx = np.concatenate(pchunks, axis=1)
                    veh_ctx = _interp(tv, veh, q_ctx)
                    # response: EDA and HR change after vs before event
                    pre = np.interp([t_e - RESP_PRE_S, t_e], teda, eda).mean()
                    post = np.interp([t_e + RESP_POST_S], teda, eda)[0]
                    eda_delta = float(post - pre)
                    sid = f"{sub.name}::{run.name}::ev{j}"
                    sample = ObservedDualStreamSample(
                        sample_id=sid, group_id=sub.name, schema=schema,
                        physiology_values=phys_ctx.astype(np.float32),
                        physiology_timestamps_s=(q_ctx - q_ctx[0]).astype(np.float32),
                        physiology_feature_mask=np.ones_like(phys_ctx, dtype=bool),
                        vehicle_values=veh_ctx.astype(np.float32),
                        vehicle_timestamps_s=(q_ctx - q_ctx[0]).astype(np.float32),
                        vehicle_feature_mask=np.ones_like(veh_ctx, dtype=bool),
                        source_sample_hash=hashlib.sha256(sid.encode()).hexdigest(),
                        context_duration_s=CTX_S)
                    out.append((sample, eda_delta, sub.name))
            except (StopIteration, ValueError):
                continue
    return out, schema


def _export(adapter, batch):
    adapter.encoder.eval()
    with torch.no_grad():
        o = adapter(batch)
    return o.pooled_embedding.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True); HEAVY.mkdir(parents=True, exist_ok=True)
    print("[event] building events...", flush=True)
    events, schema = build_events()
    samples = [e[0] for e in events]; y = np.array([e[1] for e in events]); groups = [e[2] for e in events]
    print(f"[event] events={len(samples)} subjects={len(set(groups))} y_range=[{y.min():.2f},{y.max():.2f}]", flush=True)
    # clip extreme responses, standardize target later
    lo, hi = np.percentile(y, [2, 98]); y = np.clip(y, lo, hi)

    uniq = sorted(set(groups)); test_subs = set(uniq[::4])
    train_idx = [i for i, g in enumerate(groups) if g not in test_subs]
    test_idx = [i for i, g in enumerate(groups) if g in test_subs]
    print(f"[event] train={len(train_idx)} test={len(test_idx)}", flush=True)

    batch = collate_observation_samples(samples)
    pretrain_idx, validation_idx = split_group_train_validation(
        train_idx,
        groups,
        seed=SEED,
    )
    fold = FoldLineage(
        fold_id="ev_loso",
        train_sample_ids=tuple(samples[i].sample_id for i in pretrain_idx),
        validation_sample_ids=tuple(samples[i].sample_id for i in validation_idx),
        held_out_sample_ids=tuple(samples[i].sample_id for i in test_idx),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    veh_labels = tuple((n, n) for n in schema.vehicle_feature_names)

    ytr_mean = y[train_idx].mean()
    runs = [("chronaris_safe_lag", "chronaris", "safe_lag"),
            ("chronaris_multiscale", "chronaris", "multiscale"),
            ("vehicle_only", "vehicle_only", "multiscale"),
            ("physiology_only", "physiology_only", "multiscale")]
    results = []
    for label, method, fk in runs:
        t0 = time.perf_counter()
        print(f"[event] training {label} ...", flush=True)
        res = train_common_pretext_method(
            method, batch=batch, fold=fold, physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=veh_labels,
            normalizer=normalizer, output_root=str(HEAVY / "checkpoints" / label),
            config=CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED),
            augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk, resume=False)
        enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
        adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="ev", checkpoint_sha256="0" * 64)
        emb = _export(adapter, batch)
        fx = StandardScaler().fit(emb[train_idx])
        model = Ridge(alpha=10.0).fit(fx.transform(emb[train_idx]), y[train_idx])
        pred = model.predict(fx.transform(emb[test_idx]))
        # skill vs predicting-train-mean baseline
        baseline_mae = mean_absolute_error(y[test_idx], np.full_like(y[test_idx], ytr_mean))
        model_mae = mean_absolute_error(y[test_idx], pred)
        r2 = r2_score(y[test_idx], pred)
        rec = {"label": label, "r2": round(float(r2), 3), "mae": round(float(model_mae), 4),
               "skill_vs_mean": round(float(1 - model_mae / baseline_mae), 3), "elapsed_s": round(time.perf_counter() - t0, 1)}
        results.append(rec); print(f"  -> {rec}", flush=True)

    print("\n===== CogPilot event->EDA-response (cross-modal, public) =====")
    print(f"{'method':<24}{'R2':>8}{'MAE':>9}{'skill':>9}")
    for r in results:
        print(f"{r['label']:<24}{r['r2']:>8}{r['mae']:>9}{r['skill_vs_mean']:>9}")
    (RUN_DIR / "event_response_metrics.json").write_text(json.dumps({
        "n_events": len(samples), "n_subjects": len(set(groups)), "test_subjects": sorted(test_subs),
        "seed": SEED, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / 'event_response_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
