"""CLARE GroupKFold-5: the definitive low-variance cross-modal test.

Single 4-subject LOSO splits are high-variance (seed-dependent flips). GroupKFold-5 puts
every subject in a test fold exactly once and averages -> the rigorous, low-variance
answer to "does central+peripheral fusion beat either single stream on cognitive load?".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.dataset.group_splits import split_group_train_validation
from chronaris.modeling.training.common_pretraining import (
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import TrainOnlyRobustNormalizer, collate_observation_samples
from chronaris.representation.augmentation import AugmentationPolicy
from chronaris.representation.lineage import FoldLineage
from run_clare_cognitive_load import build_samples, CENTRAL_NAMES, PERIPH_NAMES

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-24_clare-groupkfold"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-24_clare-groupkfold"
EPOCHS = 6
SEED = 17


def _export(adapter, batch):
    adapter.encoder.eval()
    with torch.no_grad():
        o = adapter(batch)
    return o.pooled_embedding.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    HEAVY.mkdir(parents=True, exist_ok=True)
    print("[gkf] building samples...", flush=True)
    samples, labels, groups = build_samples()
    ybin = (labels >= 7).astype(int)
    print(f"[gkf] samples={len(samples)} subjects={len(set(groups))}", flush=True)
    gkf = GroupKFold(n_splits=5)
    runs = [("fusion_safe_lag", "chronaris", "safe_lag"),
            ("fusion_multiscale", "chronaris", "multiscale"),
            ("central_only", "physiology_only", "multiscale"),
            ("peripheral_only", "vehicle_only", "multiscale")]
    fold_scores = {r[0]: [] for r in runs}
    sample_ids = [s.sample_id for s in samples]
    for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(sample_ids, ybin, groups), 1):
        print(f"[gkf] fold {fold_id}/5: train={len(tr_idx)} test={len(te_idx)}", flush=True)
        batch = collate_observation_samples(samples)
        inner_train, inner_validation = split_group_train_validation(
            tr_idx,
            groups,
            seed=SEED + fold_id,
        )
        fold_train = tuple(sample_ids[i] for i in inner_train)
        fold_validation = tuple(sample_ids[i] for i in inner_validation)
        fold_test = tuple(sample_ids[i] for i in te_idx)
        fold_obj = FoldLineage(
            fold_id=f"f{fold_id}",
            train_sample_ids=fold_train,
            validation_sample_ids=fold_validation,
            held_out_sample_ids=fold_test,
        )
        normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold_obj.train_sample_ids,
                                                     held_out_sample_ids=fold_validation + fold_test)
        veh_labels = tuple((n, n) for n in PERIPH_NAMES)
        for label, method, fk in runs:
            t0 = time.perf_counter()
            res = train_common_pretext_method(
                method, batch=batch, fold=fold_obj, physiology_feature_names=CENTRAL_NAMES,
                vehicle_feature_names=PERIPH_NAMES, vehicle_field_labels=veh_labels,
                normalizer=normalizer, output_root=str(HEAVY / f"f{fold_id}" / label),
                config=CommonPretrainingConfig(epochs=EPOCHS, batch_size=8, seed=SEED),
                augmentation_policy=AugmentationPolicy(), chronaris_fusion_kind=fk, resume=False)
            enc, _h, ln, _p = load_common_pretraining_checkpoint(res.best_checkpoint_path)
            adapter = TrainedFusionAdapter(encoder=enc, normalizer=ln, fold_id=f"f{fold_id}", checkpoint_sha256="0" * 64)
            emb = _export(adapter, batch)
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED),
            ).fit(emb[tr_idx], ybin[tr_idx])
            predictions = clf.predict(emb[te_idx])
            ba = balanced_accuracy_score(ybin[te_idx], predictions)
            fold_scores[label].append(ba)
            print(f"    {label}: bal_acc={ba:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n===== CLARE GroupKFold-5 (definitive low-variance) =====")
    print(f"{'method':<22}{'mean_bal_acc':>13}{'std':>8}")
    summary = {}
    for label, scores in fold_scores.items():
        m, sd = float(np.mean(scores)), float(np.std(scores))
        summary[label] = {"mean_bal_acc": round(m, 3), "std": round(sd, 3), "folds": [round(s, 3) for s in scores]}
        print(f"{label:<22}{m:>13.3f}{sd:>8.3f}")
    (RUN_DIR / "groupkfold_metrics.json").write_text(json.dumps({
        "n_samples": len(samples), "n_folds": 5, "epochs": EPOCHS, "seed": SEED, "summary": summary}, indent=2))
    print(f"\nwrote {RUN_DIR / 'groupkfold_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
