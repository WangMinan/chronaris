"""CLARE cognitive-load: central-EEG vs peripheral fusion (auxiliary public dataset).

Cognitive load reflects BOTH central EEG activity (alpha/theta) AND peripheral arousal
(EDA/HR). The genuinely cross-modal test: does fusing central + peripheral beat either
single stream? Streams: central = EEG amplitude envelope (4 ch), peripheral = EDA
conductance + ECG-derived HR (2 ch). LOSO by subject. Clocks aligned (EDA/ECG clock P
offset from EEG clock N).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.stats import spearmanr

from chronaris.dataset.clare_native import (
    CENTRAL_NAMES,
    PERIPH_NAMES,
    build_clare_native_dataset,
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
CLARE = Path("/home/wangminan/dataset/chronaris/clare")
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-24_clare-cognitive-load"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-24_clare-cognitive-load"

WIN_S = 10.0
N_SUBJECTS = 16
EPOCHS = 8
BATCH = 8


def _seed() -> int:
    return int(sys.argv[1]) if len(sys.argv) > 1 else 17


SEED = _seed()
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
    print("[clare] building samples...", flush=True)
    dataset = build_clare_native_dataset(
        CLARE,
        subject_limit=N_SUBJECTS,
        context_duration_s=WIN_S,
        cache_root=HEAVY / "native_cache",
    )
    labels = np.asarray(dataset.labels, dtype=int)
    groups = np.asarray(dataset.group_ids)
    sample_ids = dataset.sample_ids
    print(f"[clare] samples={len(dataset.records)} subjects={len(set(groups))} label_dist={np.bincount(labels)[1:].tolist() if labels.max()>=1 else []}", flush=True)
    uniq = sorted(set(groups))
    test_subs = set(uniq[::4])
    train_idx = [i for i, g in enumerate(groups) if g not in test_subs]
    test_idx = [i for i, g in enumerate(groups) if g in test_subs]
    print(f"[clare] train={len(train_idx)} test={len(test_idx)}", flush=True)

    pretrain_idx, validation_idx = split_group_train_validation(
        train_idx,
        groups,
        seed=SEED,
    )
    fold = FoldLineage(
        fold_id="clare_loso",
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
            method, batch=None, batch_provider=dataset.batch_provider, fold=fold, physiology_feature_names=CENTRAL_NAMES,
            vehicle_feature_names=PERIPH_NAMES, vehicle_field_labels=veh_labels,
            normalizer=normalizer, output_root=str(HEAVY / "checkpoints" / f"seed{SEED}" / label),
            config=CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED),
            augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk, resume=False)
        enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
        adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id="clare", checkpoint_sha256="0" * 64)
        emb = _export(adapter, dataset)
        # regression (Spearman) on raw 1-9
        rg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
            emb[train_idx], labels[train_idx]
        )
        pred_r = rg.predict(emb[test_idx])
        rho = float(spearmanr(labels[test_idx], pred_r).correlation)
        # binary low/high
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight="balanced",
                random_state=SEED,
            ),
        ).fit(emb[train_idx], ybin[train_idx])
        pred_b = clf.predict(emb[test_idx])
        f1 = f1_score(ybin[test_idx], pred_b, average="macro")
        ba = balanced_accuracy_score(ybin[test_idx], pred_b)
        rec = {"label": label, "spearman": round(rho, 3), "macro_f1": round(f1, 3), "bal_acc": round(ba, 3), "elapsed_s": round(time.perf_counter() - t0, 1)}
        results.append(rec)
        print(f"  -> {rec}", flush=True)

    print("\n===== CLARE cognitive load (LOSO, public auxiliary) =====")
    print(f"{'method':<22}{'spearman':>10}{'macro_f1':>10}{'bal_acc':>9}")
    for r in results:
        print(f"{r['label']:<22}{r['spearman']:>10}{r['macro_f1']:>10}{r['bal_acc']:>9}")
    (RUN_DIR / f"clare_metrics_seed{SEED}.json").write_text(json.dumps({
        "n_samples": len(dataset.records), "n_subjects": len(set(groups)), "test_subjects": sorted(test_subs),
        "seed": SEED, "results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / f'clare_metrics_seed{SEED}.json'}")


if __name__ == "__main__":
    sys.exit(main())
