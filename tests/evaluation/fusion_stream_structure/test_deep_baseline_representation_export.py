from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch

from chronaris.evaluation.fusion_stream_structure.contracts import ContractError
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_export import (
    DeepBaselineRepresentationExportConfig,
    extract_pooled_embeddings_for_indices,
    load_deep_baseline_representation_long_table,
    rows_from_pooled_embeddings,
    validate_checkpoint_manifest_frame,
    validate_deep_baseline_representation_frame,
    write_deep_baseline_representation_long_table,
)
from chronaris.modeling.common.gpu_runtime import prepare_fold_tensors


def _held_out_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["view_a::w001", "view_a::w002"],
            "sortie_id": ["sortie_1", "sortie_1"],
            "view_id": ["view_a", "view_a"],
            "window_index": [1, 2],
            "start_offset_ms": [1000, 2000],
            "pilot_id": [10033, 10033],
            "sample_partition": ["analysis", "analysis"],
            "maneuver_proxy_label": ["low", "high"],
            "physio_fluctuation_interval": [False, True],
        },
    )


def _valid_rows() -> list[dict[str, object]]:
    config = DeepBaselineRepresentationExportConfig(epochs=1, require_cuda=False)
    return rows_from_pooled_embeddings(
        _held_out_frame(),
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        config=config,
        model_name="mult",
        fold_index=1,
        fold_group="view_a",
        checkpoint_path="docs/artifacts/runs/example/checkpoint.pt",
        train_sample_count=4,
        test_sample_count=2,
    )


def test_oof_embedding_long_table_schema_round_trips(tmp_path):
    path = tmp_path / "deep_baseline_oof_embeddings_long.csv"
    write_deep_baseline_representation_long_table(_valid_rows(), path)

    frame = load_deep_baseline_representation_long_table(path)

    assert set(frame["method_name"]) == {"mult"}
    assert "fusion_feature_1" in frame.columns
    assert "fusion_feature_2" in frame.columns
    assert set(frame["representation_family"]) == {"T2_response_lovo_seed17_pooled_embedding"}
    assert frame["source_fold_group"].tolist() == ["view_a", "view_a"]
    assert frame["window_id"].tolist() == ["view_a::w001", "view_a::w002"]
    assert frame["time"].tolist() == [1000.0, 2000.0]


@pytest.mark.parametrize("column", ["logit_1", "prediction_value", "rank_score", "embedding_norm", "y_true", "loss"])
def test_forbidden_prediction_or_diagnostic_columns_are_rejected(column):
    frame = pd.DataFrame(_valid_rows())
    frame[column] = 1.0

    with pytest.raises(ContractError):
        validate_deep_baseline_representation_frame(frame)


def test_checkpoint_manifest_schema():
    frame = pd.DataFrame(
        [
            {
                "model_name": "mult",
                "task_name": "T2_next_window_physiology_response",
                "task_type": "regression",
                "split_strategy": "leave_one_view_out",
                "seed": 17,
                "fold_index": 1,
                "fold_group": "view_a",
                "checkpoint_path": "docs/artifacts/runs/example/checkpoint.pt",
                "representation_family": "T2_response_lovo_seed17_pooled_embedding",
                "source_e_run_manifest_path": "e.json",
                "source_f_run_manifest_path": "f.json",
                "training_invoked": True,
                "confirmed_metrics_changed": False,
            }
        ]
    )

    validate_checkpoint_manifest_frame(frame)


def test_representation_family_is_required():
    frame = pd.DataFrame(_valid_rows())
    frame.loc[0, "representation_family"] = ""

    with pytest.raises(ContractError):
        validate_deep_baseline_representation_frame(frame)


def test_fold_metadata_aligns_to_sample_id_window_id_and_time():
    rows = _valid_rows()

    assert rows[0]["window_id"] == "view_a::w001"
    assert rows[0]["time"] == 1000.0
    assert rows[0]["source_fold_index"] == 1
    assert rows[0]["source_test_sample_count"] == 2
    assert rows[1]["fusion_feature_2"] == pytest.approx(0.4)


def test_partial_fold_failure_status_can_be_preserved():
    status_row = {
        "model_name": "mult",
        "fold_index": 1,
        "fold_group": "view_a",
        "status": "failed",
        "reason": "RuntimeError:oom",
        "training_invoked": True,
        "confirmed_metrics_changed": False,
    }

    assert status_row["status"] == "failed"
    assert "oom" in status_row["reason"]
    assert status_row["confirmed_metrics_changed"] is False


def test_no_confirmed_metrics_mutation_flags_on_rows():
    rows = _valid_rows()

    assert "confirmed_metrics_changed" not in rows[0]
    assert rows[0]["source_task_name"] == "T2_next_window_physiology_response"


@dataclass(frozen=True)
class _MockOutput:
    pooled_embedding: torch.Tensor
    logits: torch.Tensor


class _MockPooledModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, modality_arrays, *, time_axis, modality_masks):
        del time_axis, modality_masks
        phys = modality_arrays["physiology"].mean(dim=(1, 2), keepdim=False).view(-1, 1)
        vehicle = modality_arrays["vehicle"].mean(dim=(1, 2), keepdim=False).view(-1, 1)
        pooled = torch.cat([phys, vehicle], dim=1) * self.weight
        return _MockOutput(pooled_embedding=pooled, logits=pooled[:, :1])


def test_mock_model_can_export_pooled_embedding():
    arrays = {
        "physiology": np.ones((3, 4, 2), dtype=np.float32),
        "vehicle": np.full((3, 4, 1), 2.0, dtype=np.float32),
    }
    masks = {
        "physiology": np.ones((3, 4), dtype=np.uint8),
        "vehicle": np.ones((3, 4), dtype=np.uint8),
    }
    prepared = prepare_fold_tensors(
        modality_arrays=arrays,
        modality_masks=masks,
        time_axis=np.tile(np.arange(4, dtype=np.float32), (3, 1)),
        ordered_modalities=("physiology", "vehicle"),
        train_indices=np.asarray([0, 1], dtype=int),
        targets=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        requested_mode="cpu",
        device="cpu",
    )

    embeddings = extract_pooled_embeddings_for_indices(
        _MockPooledModel(),
        prepared,
        np.asarray([1, 2], dtype=int),
        batch_size=1,
        amp_mode="off",
    )

    assert embeddings.shape == (2, 2)
    assert np.isfinite(embeddings).all()
