"""Test whether the lag-aware loss adds cross-stream increment on Dingxin.

Trains chronaris safe_lag WITH lag_aware_alignment_loss injected (weight 0.1) on fold01,
seed 17, 30 epochs, and evaluates both future-maneuver and future-physiology with the
exact simple-downstream consumer. Compares to the no-lag-aware safe_lag (seed 17) to
isolate the lag-aware contribution (wave-B core).
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
    physiology_metric_summary,
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
LABEL = "chronaris_safe_lag_lagaware"


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
    bundle = fit_simple_loso_targets(extract_simple_raw_targets(source, snapshot_root=SNAPSHOT))
    m_fold = bundle.maneuver_targets[bundle.maneuver_targets["fold_id"].astype(str) == FOLD_ID].copy()
    p_fold = bundle.physiology_targets[bundle.physiology_targets["fold_id"].astype(str) == FOLD_ID].copy()
    train_ctx = tuple(m_fold[m_fold["split_role"] == "train"]["context_id"].astype(str))
    held_ctx = tuple(m_fold[m_fold["split_role"] == "held_out"]["context_id"].astype(str))
    data = load_dingxin_fold_pretraining_data(
        fold_id=FOLD_ID, snapshot_root=SNAPSHOT, fixed_audit_root=FIXED_AUDIT, inner_split_root=INNER_SPLIT,
    )
    schema = data.index.plan.schema; fold = data.fold; provider = data.load_batch
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        provider, train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids, batch_size=2,
    )
    vehicle_labels = tuple((name, name) for name in schema.vehicle_feature_names)

    print(f"[lag-aware] training {LABEL} (fusion=safe_lag, lag_aware_weight=0.1) ...", flush=True)
    t0 = time.perf_counter()
    result = train_common_pretext_method(
        "chronaris", batch=None, batch_provider=provider, fold=fold,
        physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
        vehicle_field_labels=vehicle_labels, normalizer=normalizer,
        output_root=str(HEAVY / "checkpoints" / LABEL),
        config=CommonPretrainingConfig(epochs=30, batch_size=4, seed=SEED),
        augmentation_policy=AugmentationPolicy(),
        chronaris_fusion_kind="safe_lag",
        chronaris_lag_aware_weight=0.1,
        resume=False,
    )
    print(f"  trained in {time.perf_counter()-t0:.1f}s", flush=True)
    encoder, _h, loaded_norm, _p = load_common_pretraining_checkpoint(result.best_checkpoint_path)
    adapter = TrainedFusionAdapter(encoder=encoder, normalizer=loaded_norm, fold_id=FOLD_ID, checkpoint_sha256="0" * 64)

    train_reps = _export(adapter, provider, train_ctx); held_reps = _export(adapter, provider, held_ctx)
    train_ids = tuple(train_reps.keys()); held_ids = tuple(held_reps.keys())
    consumer = fit_simple_downstream_consumer(
        pooled_embedding=np.stack([train_reps[i] for i in train_ids]), sample_ids=train_ids,
        maneuver_targets=m_fold, physiology_targets=p_fold, config=SimpleConsumerConfig(random_state=SEED),
    )
    pred = consumer.predict(np.stack([held_reps[i] for i in held_ids]))
    m_metrics, _ = maneuver_metric_summary(m_fold, sample_ids=held_ids, score_prediction=pred["maneuver_score"], class_probability=pred["maneuver_probability"])
    p_metrics, _ = physiology_metric_summary(p_fold, sample_ids=held_ids, standardized_prediction=pred["physiology_standardized"], fields=consumer.physiology_fields)
    record = {
        "label": LABEL,
        "maneuver_macro_f1": round(m_metrics.get("macro_f1"), 4),
        "phys_std_rmse": round(p_metrics.get("standardized_rmse_macro"), 4),
        "phys_skill": round(p_metrics.get("skill_vs_persistence"), 4),
        "phys_pos_ratio": round(p_metrics.get("positive_skill_field_ratio"), 4),
        "phys_spo2_rmse": round(p_metrics.get("spo2_rmse_macro"), 4),
    }
    print(json.dumps(record, indent=2))
    (RUN_DIR / "lagaware_test.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {RUN_DIR / 'lagaware_test.json'}")


if __name__ == "__main__":
    sys.exit(main())
