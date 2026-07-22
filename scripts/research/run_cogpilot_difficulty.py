"""CogPilot difficulty classification (public-data gate-3 candidate).

Builds a real dual-stream dataset from the CogPilot/PhysioNet VR-piloting data:
physiology = slow autonomic signals (EDA, PPG, respiration); vehicle = X-Plane
aircraft state (airspeed, attitude, altitude, ILS deflection, climb rate). Each run is
resampled to a common 10 Hz grid over a 30 s window. Task: 4-class flight difficulty,
grouped by subject (LOSO), so physiology+aircraft fusion should beat either single stream
where task difficulty drives both arousal and control activity.

Trains chronaris (safe_lag), chronaris (multiscale), vehicle_only, physiology_only with
the same unlabeled budget, then a Logistic head on the frozen window representation, and
reports LOSO macro-F1. This is a real public-data downstream result.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score

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

REPO = Path(__file__).resolve().parents[2]
CP_ROOT = Path("/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-ils")
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-23_cogpilot-difficulty"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-23_cogpilot-difficulty"

WINDOW_START_S = 60.0
WINDOW_DUR_S = 30.0
FS_HZ = 10
N_PTS = int(WINDOW_DUR_S * FS_HZ)
N_SUBJECTS = 10  # tractable subset
EPOCHS = 12
BATCH = 8


def _seed() -> int:
    return int(sys.argv[1]) if len(sys.argv) > 1 else 17


SEED = _seed()

PHYS_FILES = {  # stream token -> columns to use
    "lslshimmereda": ("ppg_finger_mV", "eda_hand_l_kOhms"),
    "lslshimmerresp": ("respiration_trace_mV",),
}
VEH_FILE = "lslxp11xpcac"
VEH_COLS = (
    "aircraft_indicated_airspeed_kias", "aircraft_pitch_deg", "aircraft_roll_deg",
    "aircraft_agl_altitude_m", "aircraft_climb_rate_mps",
    "aircraft_ils_deflection_gs", "aircraft_ils_deflection_h", "aircraft_velocity_u_mps",
)
PHYS_NAMES = ("physiology.ppg", "physiology.eda", "physiology.resp")
VEH_NAMES = tuple(f"vehicle.{c.replace('aircraft_','')}" for c in VEH_COLS)


def _resample(t_s, vals, cols, query):
    """Resample selected columns onto the query grid (seconds) by per-bin mean/interp."""
    out = np.full((len(query), len(cols)), np.nan, dtype=np.float64)
    for j, c in enumerate(cols):
        x = vals[c].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & (t_s >= query.min() - 1) & (t_s <= query.max() + 1)
        if m.sum() < 2:
            continue
        tm, xm = t_s[m], x[m]
        order = np.argsort(tm); tm, xm = tm[order], xm[order]
        out[:, j] = np.interp(query, tm, xm)
    # forward/back fill any edge NaNs per column
    for j in range(len(cols)):
        col = out[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            out[:, j] = 0.0
        elif bad.any():
            idx = np.arange(len(col)); out[bad, j] = np.interp(idx[bad], idx[~bad], col[~bad])
    return out


def _load_run(run_dir, query):
    """Load physiology + vehicle windows for one run; return (phys[N,3], veh[N,8]) or None."""
    phys_chunks = []
    for token, cols in PHYS_FILES.items():
        f = list(Path(run_dir).glob(f"*stream-{token}*_dat.csv"))
        if not f:
            return None
        df = pd.read_csv(f[0])
        t = (df.iloc[:, 0].to_numpy(dtype=np.float64) - df.iloc[0, 0]) * 86400.0
        phys_chunks.append(_resample(t, df, cols, query))
    phys = np.concatenate(phys_chunks, axis=1)
    f = list(Path(run_dir).glob(f"*stream-{VEH_FILE}*_dat.csv"))
    if not f:
        return None
    df = pd.read_csv(f[0])
    t = (df.iloc[:, 0].to_numpy(dtype=np.float64) - df.iloc[0, 0]) * 86400.0
    veh = _resample(t, df, VEH_COLS, query)
    return phys, veh


def build_samples():
    query = WINDOW_START_S + np.arange(N_PTS) / FS_HZ
    subs = sorted(CP_ROOT.glob("sub-cp*"))[:N_SUBJECTS]
    schema = ObservationSchema(
        schema_id="cogpilot_difficulty.v1", source_kind="cogpilot_public",
        physiology_feature_names=PHYS_NAMES, vehicle_feature_names=VEH_NAMES,
        physiology_feature_roles=tuple("observed" for _ in PHYS_NAMES),
        vehicle_feature_roles=tuple("observed" for _ in VEH_NAMES),
    )
    samples, labels, groups = [], [], []
    for sub in subs:
        sub_id = sub.name
        for run in sorted(sub.glob("ses-*/level-*_run-*")):
            level = run.name.split("_")[0]  # level-01B..04B
            diff = int(level.split("-")[1][:2]) - 1  # 0..3
            loaded = _load_run(run, query)
            if loaded is None:
                continue
            phys, veh = loaded
            sid = f"{sub_id}::{run.name}"
            samples.append(ObservedDualStreamSample(
                sample_id=sid, group_id=sub_id, schema=schema,
                physiology_values=phys.astype(np.float32),
                physiology_timestamps_s=(query - query[0]).astype(np.float32),
                physiology_feature_mask=np.ones_like(phys, dtype=bool),
                vehicle_values=veh.astype(np.float32),
                vehicle_timestamps_s=(query - query[0]).astype(np.float32),
                vehicle_feature_mask=np.ones_like(veh, dtype=bool),
                source_sample_hash=hashlib.sha256(f"{sub_id}:{run.name}".encode()).hexdigest(),
            ))
            labels.append(diff); groups.append(sub_id)
    return samples, np.array(labels), np.array(groups), schema


def _export_pooled(adapter, batch):
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.no_grad():
        out = adapter(normalized)
    return out.pooled_embedding.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True); HEAVY.mkdir(parents=True, exist_ok=True)
    print("[cogpilot] building samples...", flush=True)
    samples, labels, groups, schema = build_samples()
    print(f"[cogpilot] samples={len(samples)} subjects={len(set(groups))} label_dist={np.bincount(labels).tolist()}", flush=True)
    config = CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED)

    # LOSO: hold out ~25% of subjects as test (deterministic, by sorted subject)
    uniq = sorted(set(groups)); n_test = max(1, len(uniq)//4)
    test_subs = set(uniq[::4])  # every 4th subject -> ~25%, spread across the cohort
    train_idx = [i for i, g in enumerate(groups) if g not in test_subs]
    test_idx = [i for i, g in enumerate(groups) if g in test_subs]
    print(f"[cogpilot] train={len(train_idx)} test={len(test_idx)} test_subs={sorted(test_subs)}", flush=True)

    fold_train = tuple(samples[i].sample_id for i in train_idx)
    fold_test = tuple(samples[i].sample_id for i in test_idx)
    from chronaris.representation.lineage import FoldLineage
    fold = FoldLineage(fold_id="cogpilot_loso", train_sample_ids=fold_train,
                       validation_sample_ids=(),
                       held_out_sample_ids=fold_test)
    batch = collate_observation_samples(samples)
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold_train, held_out_sample_ids=fold_test)
    veh_labels = tuple((n, n) for n in schema.vehicle_feature_names)

    runs = [("chronaris_safe_lag", "chronaris", "safe_lag"),
            ("chronaris_multiscale", "chronaris", "multiscale"),
            ("vehicle_only", "vehicle_only", "multiscale"),
            ("physiology_only", "physiology_only", "multiscale")]
    results = []
    for label, method, fk in runs:
        t0 = time.perf_counter()
        print(f"[cogpilot] training {label} ...", flush=True)
        res = train_common_pretext_method(
            method, batch=batch, fold=fold,
            physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,
            vehicle_field_labels=veh_labels, normalizer=normalizer,
            output_root=str(HEAVY / "checkpoints" / f"seed{SEED}" / label), config=config,
            augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk, resume=False,
        )
        enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
        adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="cogpilot", checkpoint_sha256="0"*64)
        emb = _export_pooled(adapter, batch)
        Xtr = StandardScaler().fit_transform(emb[train_idx]); Xte = StandardScaler().fit_transform(emb[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED).fit(Xtr, labels[train_idx])
        pred = clf.predict(Xte)
        f1 = f1_score(labels[test_idx], pred, average="macro")
        ba = balanced_accuracy_score(labels[test_idx], pred)
        rec = {"label": label, "macro_f1": round(f1, 4), "balanced_accuracy": round(ba, 4), "elapsed_s": round(time.perf_counter()-t0, 1)}
        results.append(rec); print(f"  -> {rec}", flush=True)

    print("\n===== CogPilot difficulty 4-class (LOSO, public data) =====")
    print(f"{'method':<24}{'macro_f1':>10}{'bal_acc':>10}")
    for r in results:
        print(f"{r['label']:<24}{r['macro_f1']:>10}{r['balanced_accuracy']:>10}")
    (RUN_DIR / f"difficulty_metrics_seed{SEED}.json").write_text(json.dumps({
        "n_subjects": len(set(groups)), "n_samples": len(samples), "test_subjects": sorted(test_subs),
        "seed": SEED, "epochs": EPOCHS, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / f'difficulty_metrics_seed{SEED}.json'}")


if __name__ == "__main__":
    sys.exit(main())
