"""Add MulT + ContiFormer + naive-sync to the Dingxin maneuver comparison.

Same fold01/seed17/30ep setup as run_dingxin_safe_lag_maneuver.py; trains the remaining
fusion baselines so gate-4 (Chronaris beats ALL fusion baselines) can be assessed on a
single same-run table. Reuses the exact simple-downstream target + consumer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    load_dingxin_target_source_data,
)
from chronaris.evaluation.dingxin.simple_downstream_consumers import (
    SimpleConsumerConfig,
    fit_simple_downstream_consumer,
    maneuver_metric_summary,
)
from chronaris.evaluation.dingxin.simple_downstream_protocol import (
    extract_simple_raw_targets,
    fit_simple_loso_targets,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.common_pretraining import (
    CommonPretrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import TrainOnlyRobustNormalizer
from chronaris.representation.augmentation import AugmentationPolicy

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
FIXED_AUDIT = REPO / "docs/artifacts/runs/2026-07-10_fixed-data-audit"
INNER_SPLIT = REPO / "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-22_dingxin-safe-lag-maneuver"
HEAVY = REPO / "artifacts/application_evaluation/2026-07-22_dingxin-safe-lag-maneuver"

FOLD_ID = "leave_one_sortie_out__fold01"
SEED = 17
EPOCHS = 30
BATCH = 4


def _export(adapter, provider, sample_ids):
    batch = provider(list(sample_ids))
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.inference_mode():
        out = adapter(normalized)
    pooled = out.pooled_embedding.detach().cpu().numpy().astype(np.float64)
    return {str(sid): pooled[i] for i, sid in enumerate(batch.sample_ids)}


def main() -> None:
    source = load_dingxin_target_source_data(fixed_audit_root=FIXED_AUDIT, snapshot_root=SNAPSHOT)
    raw = extract_simple_raw_targets(source, snapshot_root=SNAPSHOT)
    bundle = fit_simple_loso_targets(raw)
    m_fold = bundle.maneuver_targets[bundle.maneuver_targets["fold_id"].astype(str) == FOLD_ID].copy()
    p_fold = bundle.physiology_targets[bundle.physiology_targets["fold_id"].astype(str) == FOLD_ID].copy()
    train_ctx = tuple(m_fold[m_fold["split_role"] == "train"]["context_id"].astype(str))
    held_ctx = tuple(m_fold[m_fold["split_role"] == "held_out"]["context_id"].astype(str))

    data = load_dingxin_fold_pretraining_data(
        fold_id=FOLD_ID, snapshot_root=SNAPSHOT, fixed_audit_root=FIXED_AUDIT, inner_split_root=INNER_SPLIT,
    )
    schema = data.index.plan.schema
    fold = data.fold
    provider = data.load_batch
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        provider, train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids, batch_size=2,
    )
    vehicle_labels = tuple((name, name) for name in schema.vehicle_feature_names)
    config = CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED)
    policy = AugmentationPolicy()

    results = []
    for label, method in (("mult", "mult"), ("contiformer", "contiformer")):
        t0 = time.perf_counter()
        print(f"[dingxin-baselines] training {label} ...", flush=True)
        result = train_common_pretext_method(
            method, batch=None, batch_provider=provider, fold=fold,
            physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,
            vehicle_field_labels=vehicle_labels, normalizer=normalizer,
            output_root=str(HEAVY / "checkpoints" / label), config=config,
            augmentation_policy=policy, resume=False,
        )
        encoder, _h, loaded_norm, _p = load_common_pretraining_checkpoint(result.best_checkpoint_path)
        adapter = TrainedFusionAdapter(encoder=encoder, normalizer=loaded_norm, fold_id=FOLD_ID, checkpoint_sha256="0" * 64)
        train_reps = _export(adapter, provider, train_ctx)
        held_reps = _export(adapter, provider, held_ctx)
        train_ids = tuple(train_reps.keys()); held_ids = tuple(held_reps.keys())
        consumer = fit_simple_downstream_consumer(
            pooled_embedding=np.stack([train_reps[i] for i in train_ids]), sample_ids=train_ids,
            maneuver_targets=m_fold, physiology_targets=p_fold, config=SimpleConsumerConfig(random_state=SEED),
        )
        pred = consumer.predict(np.stack([held_reps[i] for i in held_ids]))
        metrics, _ = maneuver_metric_summary(m_fold, sample_ids=held_ids, score_prediction=pred["maneuver_score"], class_probability=pred["maneuver_probability"])
        record = {"label": label, "macro_f1": round(metrics.get("macro_f1"), 4), "balanced_accuracy": round(metrics.get("balanced_accuracy"), 4), "elapsed_s": round(time.perf_counter() - t0, 1)}
        results.append(record)
        print(f"  -> {record}", flush=True)

    # merge with the prior run's metrics
    prior_path = RUN_DIR / "maneuver_metrics.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {"results": []}
    combined = {r["label"]: r for r in prior.get("results", [])}
    for r in results:
        combined[r["label"]] = r
    print("\n===== Dingxin future-maneuver 3-class — full fusion-baseline table =====")
    print(f"{'method':<26}{'macro_f1':>10}{'bal_acc':>10}")
    for label in ("chronaris_safe_lag", "chronaris_multiscale", "mult", "contiformer", "vehicle_only"):
        if label in combined:
            r = combined[label]
            print(f"{label:<26}{r.get('macro_f1')!s:>10}{r.get('balanced_accuracy')!s:>10}")
    (RUN_DIR / "maneuver_metrics_baselines.json").write_text(json.dumps({"results": results}, indent=2))
    print(f"\nwrote {RUN_DIR / 'maneuver_metrics_baselines.json'}")


if __name__ == "__main__":
    sys.exit(main())
