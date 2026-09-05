"""CogPilot difficulty classification (public-data gate-3 candidate).

Builds a real dual-stream dataset from the CogPilot/PhysioNet VR-piloting data:
physiology = slow autonomic signals (EDA, PPG, respiration); vehicle = X-Plane
aircraft state (airspeed, attitude, altitude, ILS deflection, climb rate). Each stream
retains its native timestamps over a 30 s window. Task: 4-class flight difficulty,
grouped by subject (LOSO), so physiology+aircraft fusion should beat either single stream
where task difficulty drives both arousal and control activity.

Trains chronaris (safe_lag), chronaris (multiscale), vehicle_only, physiology_only with
the same unlabeled budget, then a Logistic head on the frozen window representation, and
reports LOSO macro-F1. This is a real public-data downstream result.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score

from chronaris.dataset.cogpilot_native import build_cogpilot_difficulty_dataset
from chronaris.dataset.group_splits import split_group_train_validation
from chronaris.modeling.training.common_pretraining import (
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import TrainOnlyRobustNormalizer
from chronaris.representation.lineage import FoldLineage
from chronaris.representation.augmentation import AugmentationPolicy

REPO = Path(__file__).resolve().parents[2]
CP_ROOT = Path("/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-ils")
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-23_cogpilot-difficulty"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-23_cogpilot-difficulty"

WINDOW_START_S = 60.0
WINDOW_DUR_S = 30.0
N_SUBJECTS = 20  # stronger cohort
EPOCHS = 12
BATCH = 8


def _seed() -> int:
    return int(sys.argv[1]) if len(sys.argv) > 1 else 17


SEED = _seed()

def _export_pooled(adapter, dataset):
    adapter.encoder.eval()
    rows = []
    for sample_ids in dataset.batch_ids(BATCH):
        with torch.no_grad():
            out = adapter(dataset.batch_provider(sample_ids))
        rows.append(out.pooled_embedding.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(rows, axis=0)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    HEAVY.mkdir(parents=True, exist_ok=True)
    print("[cogpilot] building samples...", flush=True)
    dataset = build_cogpilot_difficulty_dataset(
        CP_ROOT,
        subject_limit=N_SUBJECTS,
        window_start_s=WINDOW_START_S,
        context_duration_s=WINDOW_DUR_S,
        cache_root=HEAVY / "native_cache",
    )
    labels = np.asarray(dataset.labels, dtype=int)
    groups = np.asarray(dataset.group_ids)
    sample_ids = dataset.sample_ids
    schema = dataset.schema
    print(f"[cogpilot] samples={len(dataset.records)} subjects={len(set(groups))} label_dist={np.bincount(labels).tolist()}", flush=True)
    config = CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED)

    # LOSO: hold out ~25% of subjects as test (deterministic, by sorted subject)
    uniq = sorted(set(groups))
    test_subs = set(uniq[::4])  # every 4th subject -> ~25%, spread across the cohort
    train_idx = [i for i, g in enumerate(groups) if g not in test_subs]
    test_idx = [i for i, g in enumerate(groups) if g in test_subs]
    print(f"[cogpilot] train={len(train_idx)} test={len(test_idx)} test_subs={sorted(test_subs)}", flush=True)

    pretrain_idx, validation_idx = split_group_train_validation(
        train_idx,
        groups,
        seed=SEED,
    )
    fold_train = tuple(sample_ids[i] for i in pretrain_idx)
    fold_validation = tuple(sample_ids[i] for i in validation_idx)
    fold_test = tuple(sample_ids[i] for i in test_idx)
    fold = FoldLineage(fold_id="cogpilot_loso", train_sample_ids=fold_train,
                       validation_sample_ids=fold_validation,
                       held_out_sample_ids=fold_test)
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        dataset.batch_provider,
        train_sample_ids=fold_train,
        held_out_sample_ids=fold_validation + fold_test,
        batch_size=BATCH,
    )
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
            method, batch=None, batch_provider=dataset.batch_provider, fold=fold,
            physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,
            vehicle_field_labels=veh_labels, normalizer=normalizer,
            output_root=str(HEAVY / "checkpoints" / f"seed{SEED}" / label), config=config,
            augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk, resume=False,
        )
        enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
        adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="cogpilot", checkpoint_sha256="0"*64)
        emb = _export_pooled(adapter, dataset)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight="balanced",
                random_state=SEED,
            ),
        ).fit(emb[train_idx], labels[train_idx])
        pred = clf.predict(emb[test_idx])
        f1 = f1_score(labels[test_idx], pred, average="macro")
        ba = balanced_accuracy_score(labels[test_idx], pred)
        rec = {"label": label, "macro_f1": round(f1, 4), "balanced_accuracy": round(ba, 4), "elapsed_s": round(time.perf_counter()-t0, 1)}
        results.append(rec)
        print(f"  -> {rec}", flush=True)

    print("\n===== CogPilot difficulty 4-class (LOSO, public data) =====")
    print(f"{'method':<24}{'macro_f1':>10}{'bal_acc':>10}")
    for r in results:
        print(f"{r['label']:<24}{r['macro_f1']:>10}{r['balanced_accuracy']:>10}")
    (RUN_DIR / f"difficulty_metrics_{N_SUBJECTS}subj_seed{SEED}.json").write_text(json.dumps({
        "n_subjects": len(set(groups)), "n_samples": len(dataset.records), "test_subjects": sorted(test_subs),
        "seed": SEED, "epochs": EPOCHS, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / f'difficulty_metrics_{N_SUBJECTS}subj_seed{SEED}.json'}")


if __name__ == "__main__":
    sys.exit(main())
