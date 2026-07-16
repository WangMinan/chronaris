from __future__ import annotations

import numpy as np
import pandas as pd

from chronaris.evaluation.dingxin.simple_downstream_consumers import (
    SimpleConsumerConfig,
    fit_simple_downstream_consumer,
    maneuver_metric_summary,
    physiology_metric_summary,
)
from chronaris.evaluation.dingxin.simple_downstream_protocol import (
    SimpleRawTargetBundle,
    fit_simple_loso_targets,
)


def test_fold_targets_use_90_views_and_60_independent_vehicle_contexts() -> None:
    raw = _synthetic_raw_bundle()
    bundle = fit_simple_loso_targets(raw)

    assert len(bundle.fold_manifest) == 2
    assert len(bundle.maneuver_targets) == 180
    for fold_id in bundle.fold_manifest["fold_id"]:
        maneuver = bundle.maneuver_targets[
            bundle.maneuver_targets["fold_id"] == fold_id
        ]
        assert len(maneuver) == 90
        assert maneuver["vehicle_context_id"].nunique() == 60
        assert np.allclose(
            maneuver.groupby(["split_role", "vehicle_context_id"])[
                "sample_weight"
            ].sum(),
            1.0,
        )
        selected = bundle.physiology_targets[
            (bundle.physiology_targets["fold_id"] == fold_id)
            & bundle.physiology_targets["selected"]
        ]
        assert selected["context_id"].nunique() == 90
        assert selected["field_name"].nunique() == 3


def test_fixed_consumers_fit_once_and_aggregate_shared_vehicle_contexts() -> None:
    bundle = fit_simple_loso_targets(_synthetic_raw_bundle())
    fold_id = str(bundle.fold_manifest.iloc[0]["fold_id"])
    maneuver = bundle.maneuver_targets[
        bundle.maneuver_targets["fold_id"] == fold_id
    ]
    physiology = bundle.physiology_targets[
        bundle.physiology_targets["fold_id"] == fold_id
    ]
    sample_ids = tuple(maneuver["context_id"].astype(str))
    rng = np.random.default_rng(17)
    embeddings = rng.normal(size=(len(sample_ids), 64)).astype(np.float32)
    consumer = fit_simple_downstream_consumer(
        pooled_embedding=embeddings,
        sample_ids=sample_ids,
        maneuver_targets=maneuver,
        physiology_targets=physiology,
        config=SimpleConsumerConfig(random_state=17),
    )

    held_out = maneuver[maneuver["split_role"] == "held_out"]
    position = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    held_out_ids = tuple(held_out["context_id"].astype(str))
    held_out_values = embeddings[[position[value] for value in held_out_ids]]
    predictions = consumer.predict(held_out_values)
    maneuver_summary, aggregated = maneuver_metric_summary(
        held_out,
        sample_ids=held_out_ids,
        score_prediction=predictions["maneuver_score"],
        class_probability=predictions["maneuver_probability"],
    )
    physiology_summary, per_field = physiology_metric_summary(
        physiology[physiology["split_role"] == "held_out"],
        sample_ids=held_out_ids,
        standardized_prediction=predictions["physiology_standardized"],
        fields=consumer.physiology_fields,
    )

    assert maneuver_summary["independent_vehicle_context_count"] == 30
    assert len(aggregated) == 30
    assert aggregated["view_count"].isin((1, 2)).all()
    assert physiology_summary["view_context_count"] == len(held_out_ids)
    assert physiology_summary["field_count"] == 3
    assert len(per_field) == 3


def _synthetic_raw_bundle() -> SimpleRawTargetBundle:
    context_rows = []
    maneuver_rows = []
    physiology_rows = []
    sorties = (("sortie_a", ("view_a1", "view_a2")), ("sortie_b", ("view_b",)))
    semantic_keys = tuple(f"maneuver_{index}" for index in range(5))
    physiology_fields = (
        ("eeg_alpha", "eeg"),
        ("eeg_beta", "eeg"),
        ("spo2", "spo2"),
    )
    for sortie_index, (sortie_id, views) in enumerate(sorties):
        for index in range(30):
            target_start = (index + 6) * 5_000
            vehicle_context_id = f"{sortie_id}::target_{target_start:09d}"
            for semantic_index, semantic_key in enumerate(semantic_keys):
                phase = index + sortie_index * 0.35 + semantic_index * 0.1
                maneuver_rows.append(
                    {
                        "vehicle_context_id": vehicle_context_id,
                        "sortie_id": sortie_id,
                        "target_start_offset_ms": target_start,
                        "semantic_key": semantic_key,
                        "current_count": 5,
                        "current_std": 1.0 + 0.04 * max(phase - 1.0, 0.0),
                        "current_abs_delta": 0.5 + 0.03 * max(phase - 1.0, 0.0),
                        "future_count": 5,
                        "future_std": 1.0 + 0.04 * phase,
                        "future_abs_delta": 0.5 + 0.03 * phase,
                        "snapshot_sha256": "vehicle-hash",
                    }
                )
            for view_index, view_id in enumerate(views):
                context_id = f"{view_id}::context_{index:02d}"
                context_rows.append(
                    {
                        "context_id": context_id,
                        "sortie_id": sortie_id,
                        "view_id": view_id,
                        "pilot_id": view_index + 1,
                        "input_start_offset_ms": target_start - 30_000,
                        "input_end_exclusive_ms": target_start,
                        "target_start_offset_ms": target_start,
                        "target_end_exclusive_ms": target_start + 5_000,
                        "vehicle_context_id": vehicle_context_id,
                        "snapshot_stop_offset_ms": 181_000,
                    }
                )
                for field_index, (field_name, category) in enumerate(
                    physiology_fields
                ):
                    level = index + sortie_index * 0.5 + view_index * 0.2
                    physiology_rows.append(
                        {
                            "context_id": context_id,
                            "sortie_id": sortie_id,
                            "view_id": view_id,
                            "field_name": field_name,
                            "semantic_category": category,
                            "current_count": 5,
                            "current_median": level + field_index,
                            "future_count": 5,
                            "future_median": level + field_index + 0.25,
                            "snapshot_sha256": "physiology-hash",
                        }
                    )
    return SimpleRawTargetBundle(
        contexts=pd.DataFrame(context_rows),
        maneuver_statistics=pd.DataFrame(maneuver_rows),
        physiology_statistics=pd.DataFrame(physiology_rows),
        source_hashes={"synthetic": "hash"},
    )
