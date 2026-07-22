"""Decisive Dingxin maneuver comparison: safe_lag vs multiscale vs vehicle_only.

Runs ONE leave-one-sortie-out fold (fold01), ONE seed, on real Dingxin data, using the
exact simple-downstream maneuver target + consumer, so the macro-F1 is directly
comparable to the closed result (vehicle_only 0.808, old Chronaris 0.195).

Trains chronaris (safe_lag), chronaris (multiscale) and vehicle_only with the same
unlabeled budget, exports frozen representations, fits the Ridge/Logistic consumer on
the train-sortie contexts and evaluates future-maneuver 3-class macro-F1 on the held-out
sortie. This is a real Dingxin downstream result (single fold, single seed).
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


def _export_reps(adapter: TrainedFusionAdapter, provider, sample_ids) -> dict[str, np.ndarray]:
    batch = provider(list(sample_ids))
    device = next(adapter.encoder.parameters()).device
    normalized = adapter.normalizer.transform(move_observation_batch(batch, device=device))
    adapter.encoder.eval()
    with torch.inference_mode():
        out = adapter(normalized)
    pooled = out.pooled_embedding.detach().cpu().numpy().astype(np.float64)
    return {str(sid): pooled[i] for i, sid in enumerate(batch.sample_ids)}


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    HEAVY.mkdir(parents=True, exist_ok=True)

    source = load_dingxin_target_source_data(
        fixed_audit_root=FIXED_AUDIT, snapshot_root=SNAPSHOT
    )
    raw = extract_simple_raw_targets(source, snapshot_root=SNAPSHOT)
    bundle = fit_simple_loso_targets(raw)
    maneuver = bundle.maneuver_targets
    physiology = bundle.physiology_targets
    m_fold = maneuver[maneuver["fold_id"].astype(str) == FOLD_ID].copy()
    p_fold = physiology[physiology["fold_id"].astype(str) == FOLD_ID].copy()
    train_ctx = tuple(m_fold[m_fold["split_role"] == "train"]["context_id"].astype(str))
    held_ctx = tuple(m_fold[m_fold["split_role"] == "held_out"]["context_id"].astype(str))
    print(f"[dingxin] fold={FOLD_ID} train_ctx={len(train_ctx)} held_ctx={len(held_ctx)}", flush=True)

    data = load_dingxin_fold_pretraining_data(
        fold_id=FOLD_ID,
        snapshot_root=SNAPSHOT,
        fixed_audit_root=FIXED_AUDIT,
        inner_split_root=INNER_SPLIT,
    )
    provider = data.load_batch
    schema = data.index.plan.schema
    fold = data.fold
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        provider,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
        batch_size=2,
    )
    vehicle_labels = tuple((name, name) for name in schema.vehicle_feature_names)
    config = CommonPretrainingConfig(epochs=EPOCHS, batch_size=BATCH, seed=SEED)
    policy = AugmentationPolicy()

    runs = [
        ("chronaris_safe_lag", "chronaris", "safe_lag"),
        ("chronaris_multiscale", "chronaris", "multiscale"),
        ("vehicle_only", "vehicle_only", "multiscale"),
    ]
    results = []
    for label, method, fusion_kind in runs:
        t0 = time.perf_counter()
        print(f"[dingxin] training {label} ...", flush=True)
        result = train_common_pretext_method(
            method,
            batch=None,
            batch_provider=provider,
            fold=fold,
            physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,
            vehicle_field_labels=vehicle_labels,
            normalizer=normalizer,
            output_root=str(HEAVY / "checkpoints" / label),
            config=config,
            augmentation_policy=policy,
            chronaris_fusion_kind=fusion_kind,
            resume=False,
        )
        encoder, _h, loaded_norm, _p = load_common_pretraining_checkpoint(result.best_checkpoint_path)
        adapter = TrainedFusionAdapter(
            encoder=encoder, normalizer=loaded_norm,
            fold_id=FOLD_ID, checkpoint_sha256="0" * 64,
        )
        train_reps = _export_reps(adapter, provider, train_ctx)
        held_reps = _export_reps(adapter, provider, held_ctx)
        train_ids = tuple(train_reps.keys())
        held_ids = tuple(held_reps.keys())
        train_emb = np.stack([train_reps[i] for i in train_ids])
        held_emb = np.stack([held_reps[i] for i in held_ids])
        consumer = fit_simple_downstream_consumer(
            pooled_embedding=train_emb,
            sample_ids=train_ids,
            maneuver_targets=m_fold,
            physiology_targets=p_fold,
            config=SimpleConsumerConfig(random_state=SEED),
        )
        pred = consumer.predict(held_emb)
        metrics, _ = maneuver_metric_summary(
            m_fold,
            sample_ids=held_ids,
            score_prediction=pred["maneuver_score"],
            class_probability=pred["maneuver_probability"],
        )
        record = {
            "label": label,
            "method": method,
            "fusion_kind": fusion_kind,
            "step_count": result.step_count,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        }
        results.append(record)
        print(f"  -> macro_f1={record.get('macro_f1')} balanced_acc={record.get('balanced_accuracy')} ({record['elapsed_s']}s)", flush=True)

    print("\n===== Dingxin future-maneuver 3-class (fold01, seed 17) =====")
    print(f"{'method':<26}{'macro_f1':>10}{'bal_acc':>10}{'spearman':>10}{'skill':>10}")
    for r in results:
        print(
            f"{r['label']:<26}{r.get('macro_f1')!s:>10}{r.get('balanced_accuracy')!s:>10}"
            f"{r.get('spearman')!s:>10}{r.get('relative_current_state_skill')!s:>10}"
        )
    (RUN_DIR / "maneuver_metrics.json").write_text(
        json.dumps({"fold_id": FOLD_ID, "seed": SEED, "epochs": EPOCHS, "results": results}, indent=2)
    )
    print(f"\nwrote {RUN_DIR / 'maneuver_metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
