"""CogPilot event->response: the genuinely cross-modal real-data task.

Trigger (aircraft maneuver) lives in the vehicle stream; response (physiology change
after the event) lives in the physiology stream. Predicting the response needs BOTH the
trigger (vehicle) and the baseline (physiology). vehicle_only has the trigger but no
physiology baseline; physiology_only has the baseline but not the trigger; fusion has
both -> should beat each single stream. This is the non-persistent cross-modal task that
vehicle-dominated difficulty/maneuver tasks are not.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from chronaris.dataset.cogpilot_native import (
    build_cogpilot_event_response_dataset,
)
from chronaris.dataset.group_splits import split_group_train_validation
from chronaris.modeling.training.common_pretraining import (
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import TrainOnlyRobustNormalizer
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
N_SUBJECTS = 20
EPOCHS = 8
BATCH = 8
SEED = 17

def _export(adapter, dataset):
    adapter.encoder.eval()
    rows = []
    for sample_ids in dataset.batch_ids(BATCH):
        with torch.no_grad():
            output = adapter(dataset.batch_provider(sample_ids))
        rows.append(output.pooled_embedding.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(rows, axis=0)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    HEAVY.mkdir(parents=True, exist_ok=True)
    print("[event] building events...", flush=True)
    dataset = build_cogpilot_event_response_dataset(
        CP_ROOT,
        subject_limit=N_SUBJECTS,
        context_duration_s=CTX_S,
        response_pre_s=RESP_PRE_S,
        response_post_s=RESP_POST_S,
        minimum_event_gap_s=MIN_GAP_S,
        cache_root=HEAVY / "native_cache",
    )
    schema = dataset.schema
    sample_ids = dataset.sample_ids
    y = np.asarray(dataset.labels, dtype=float)
    groups = dataset.group_ids
    print(f"[event] events={len(dataset.records)} subjects={len(set(groups))} y_range=[{y.min():.2f},{y.max():.2f}]", flush=True)
    # clip extreme responses, standardize target later
    lo, hi = np.percentile(y, [2, 98])
    y = np.clip(y, lo, hi)

    uniq = sorted(set(groups))
    test_subs = set(uniq[::4])
    train_idx = [i for i, g in enumerate(groups) if g not in test_subs]
    test_idx = [i for i, g in enumerate(groups) if g in test_subs]
    print(f"[event] train={len(train_idx)} test={len(test_idx)}", flush=True)

    pretrain_idx, validation_idx = split_group_train_validation(
        train_idx,
        groups,
        seed=SEED,
    )
    fold = FoldLineage(
        fold_id="ev_loso",
        train_sample_ids=tuple(sample_ids[i] for i in pretrain_idx),
        validation_sample_ids=tuple(sample_ids[i] for i in validation_idx),
        held_out_sample_ids=tuple(sample_ids[i] for i in test_idx),
    )
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        dataset.batch_provider,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
        batch_size=BATCH,
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
            method, batch=None, batch_provider=dataset.batch_provider, fold=fold, physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=veh_labels,
            normalizer=normalizer, output_root=str(HEAVY / "checkpoints" / label),
            config=CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED),
            augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk, resume=False)
        enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
        adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="ev", checkpoint_sha256="0" * 64)
        emb = _export(adapter, dataset)
        fx = StandardScaler().fit(emb[train_idx])
        model = Ridge(alpha=10.0).fit(fx.transform(emb[train_idx]), y[train_idx])
        pred = model.predict(fx.transform(emb[test_idx]))
        # skill vs predicting-train-mean baseline
        baseline_mae = mean_absolute_error(y[test_idx], np.full_like(y[test_idx], ytr_mean))
        model_mae = mean_absolute_error(y[test_idx], pred)
        r2 = r2_score(y[test_idx], pred)
        rec = {"label": label, "r2": round(float(r2), 3), "mae": round(float(model_mae), 4),
               "skill_vs_mean": round(float(1 - model_mae / baseline_mae), 3), "elapsed_s": round(time.perf_counter() - t0, 1)}
        results.append(rec)
        print(f"  -> {rec}", flush=True)

    print("\n===== CogPilot event->EDA-response (cross-modal, public) =====")
    print(f"{'method':<24}{'R2':>8}{'MAE':>9}{'skill':>9}")
    for r in results:
        print(f"{r['label']:<24}{r['r2']:>8}{r['mae']:>9}{r['skill_vs_mean']:>9}")
    (RUN_DIR / "event_response_metrics.json").write_text(json.dumps({
        "n_events": len(dataset.records), "n_subjects": len(set(groups)), "test_subjects": sorted(test_subs),
        "seed": SEED, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / 'event_response_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
