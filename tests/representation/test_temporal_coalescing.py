from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    ensure_dingxin_model_input_contract,
)

from chronaris.representation import (
    DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
    ObservationSchema,
    ObservedDualStreamSample,
    coalesce_observation_batch,
    collate_observation_samples,
)


def _batch():
    schema = ObservationSchema(
        schema_id="temporal_coalescing.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a", "physiology.b"),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed", "observed"),
        vehicle_feature_roles=("observed",),
    )
    sample = ObservedDualStreamSample(
        sample_id="sample",
        group_id="group",
        schema=schema,
        physiology_values=np.asarray(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float32
        ),
        physiology_timestamps_s=np.asarray([0.01, 0.08, 0.16]),
        physiology_feature_mask=np.asarray(
            [[True, False], [False, True], [True, False]], dtype=bool
        ),
        vehicle_values=np.asarray([[4.0], [6.0]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([0.02, 0.07]),
        vehicle_feature_mask=np.ones((2, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(b"sample").hexdigest(),
    )
    return collate_observation_samples([sample])


def test_temporal_coalescing_uses_last_event_time_without_future_advance() -> None:
    batch = _batch()

    output = coalesce_observation_batch(batch, bin_width_s=0.1)

    assert output.source_sample_hashes == batch.source_sample_hashes
    assert output.query_timestamps_s.equal(batch.query_timestamps_s)
    assert output.physiology_point_mask.sum().item() == 2
    assert torch.allclose(
        output.physiology_timestamps_s[0, :2],
        torch.tensor([0.08, 0.16], dtype=torch.float64),
    )
    assert output.physiology_values[0, 0].tolist() == [1.0, 2.0]
    assert output.physiology_feature_mask[0, 0].tolist() == [True, True]
    assert output.vehicle_point_mask.sum().item() == 1
    assert output.vehicle_timestamps_s[0, 0].item() == 0.07
    assert output.vehicle_values[0, 0, 0].item() == 5.0


def test_temporal_coalescing_preserves_missing_modality_as_one_padded_row() -> None:
    batch = _batch()
    missing = replace(
        batch,
        physiology_values=torch.zeros_like(batch.physiology_values),
        physiology_point_mask=torch.zeros_like(batch.physiology_point_mask),
        physiology_feature_mask=torch.zeros_like(batch.physiology_feature_mask),
        physiology_observation_age_s=torch.full_like(
            batch.physiology_observation_age_s, torch.inf
        ),
    )

    output = coalesce_observation_batch(missing, bin_width_s=0.1)

    assert output.physiology_values.shape[1] == 1
    assert not output.physiology_point_mask.any()
    assert not output.physiology_feature_mask.any()
    assert torch.isinf(output.physiology_observation_age_s).all()


def test_dingxin_model_input_contract_is_created_and_reused(tmp_path) -> None:
    path = ensure_dingxin_model_input_contract(tmp_path)

    assert ensure_dingxin_model_input_contract(tmp_path) == path
    assert json.loads(path.read_text(encoding="utf-8"))[
        "model_input_bin_width_s"
    ] == 0.1


def test_dingxin_model_input_contract_rejects_legacy_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 1}, checkpoint)

    with pytest.raises(ValueError, match="predate"):
        ensure_dingxin_model_input_contract(tmp_path)


def test_future_prediction_input_contract_records_history_policy(tmp_path) -> None:
    path = ensure_dingxin_model_input_contract(
        tmp_path,
        maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "chronaris.dingxin_model_input_contract.v2"
    assert payload["maneuver_history_policy"] == (
        DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY
    )
