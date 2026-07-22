"""Dingxin future-physiology evaluation (gate-5 dual-stream incremental test).

Reuses the wave-A Dingxin checkpoints (safe_lag/multiscale/vehicle/mult/contiformer) and
adds physiology_only, then evaluates the future-physiology field task with the exact
simple-downstream consumer. Reports per-field RMSE, MAE, Spearman, relative-persistence
skill, and the positive-skill field ratio — the dual-stream-increment metric where fusion
should beat the best single stream (gate 5).
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
HEAVY = REPO / "artifacts/application_evaluation/2026-07-22_dingxin-safe-lag-maneuver"
RUN_DIR = REPO / "docs/artifacts/runs/2026-07-22_dingxin-safe-lag-maneuver"

FOLD_ID = "leave_one_sortie_out__fold01"
SEED = 17


def _export(adapter, provider, sample_ids):
    batch = provider(list(sample_ids))
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.inference_mode():
        out = adapter(normalized)
    pooled = out.pooled_embedding.detach().cpu().numpy().astype(np.float64)
    return {str(sid): pooled[i] for i, sid in enumerate(batch.sample_ids)}


def _eval_method(adapter, provider, train_ctx, held_ctx, m_fold, p_fold):
    train_reps = _export(adapter, provider, train_ctx)
    held_reps = _export(adapter, provider, held_ctx)
    train_ids = tuple(train_reps.keys()); held_ids = tuple(held_reps.keys())
    consumer = fit_simple_downstream_consumer(
        pooled_embedding=np.stack([train_reps[i] for i in train_ids]), sample_ids=train_ids,
        maneuver_targets=m_fold, physiology_targets=p_fold, config=SimpleConsumerConfig(random_state=SEED),
    )
    fields = consumer.physiology_fields
    pred = consumer.predict(np.stack([held_reps[i] for i in held_ids]))
    metrics, _ = physiology_metric_summary(
        p_fold, sample_ids=held_ids, standardized_prediction=pred["physiology_standardized"], fields=fields,
    )
    return metrics, fields


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

    # train physiology_only (single stream), reuse the rest from disk
    print("[phys] training physiology_only ...", flush=True)
    t0 = time.perf_counter()
    res = train_common_pretext_method(
        "physiology_only", batch=None, batch_provider=provider, fold=fold,
        physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
        vehicle_field_labels=vehicle_labels, normalizer=normalizer,
        output_root=str(HEAVY / "checkpoints" / "physiology_only"),
        config=CommonPretrainingConfig(epochs=30, batch_size=4, seed=SEED),
        augmentation_policy=AugmentationPolicy(), resume=False,
    )
    print(f"  physiology_only trained in {time.perf_counter()-t0:.1f}s", flush=True)

    ckpts = {
        "chronaris_safe_lag": (HEAVY / "checkpoints/chronaris_safe_lag/chronaris/best.pt", "chronaris"),
        "chronaris_multiscale": (HEAVY / "checkpoints/chronaris_multiscale/chronaris/best.pt", "chronaris"),
        "mult": (HEAVY / "checkpoints/mult/mult/best.pt", "mult"),
        "contiformer": (HEAVY / "checkpoints/contiformer/contiformer/best.pt", "contiformer"),
        "vehicle_only": (HEAVY / "checkpoints/vehicle_only/vehicle_only/best.pt", "vehicle_only"),
        "physiology_only": (res.best_checkpoint_path, "physiology_only"),
    }
    results = []
    fields = None
    for label, (ckpt, _method) in ckpts.items():
        if not Path(ckpt).exists():
            print(f"  skip {label}: {ckpt} missing"); continue
        encoder, _h, loaded_norm, _p = load_common_pretraining_checkpoint(ckpt)
        adapter = TrainedFusionAdapter(encoder=encoder, normalizer=loaded_norm, fold_id=FOLD_ID, checkpoint_sha256="0" * 64)
        metrics, fields = _eval_method(adapter, provider, train_ctx, held_ctx, m_fold, p_fold)
        record = {"label": label, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}}
        results.append(record)
        print(f"  {label}: rmse={record.get('rmse')} skill={record.get('mean_relative_persistence_skill')} pos_fields={record.get('positive_skill_field_count')}/{record.get('field_count')}", flush=True)

    print("\n===== Dingxin future-physiology (fold01, seed 17) =====")
    print(f"{'method':<24}{'rmse':>9}{'skill':>9}{'pos_fields':>12}")
    for r in results:
        print(f"{r['label']:<24}{r.get('rmse')!s:>9}{r.get('mean_relative_persistence_skill')!s:>9}"
              f"{str(r.get('positive_skill_field_count'))+'/'+str(r.get('field_count')):>12}")
    (RUN_DIR / "physiology_metrics.json").write_text(
        json.dumps({"fold_id": FOLD_ID, "seed": SEED, "fields": list(fields) if fields else [], "results": results}, indent=2)
    )
    print(f"\nwrote {RUN_DIR / 'physiology_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
