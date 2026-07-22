from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from chronaris.evaluation.application_tasks.task_aware_partial_unfreeze import (
    configure_partial_unfreeze,
)
from chronaris.evaluation.application_tasks.task_aware_safe_residual_model import (
    TaskSafeResidualHead,
    gate_is_non_degenerate,
    safe_against_anchor,
    summarize_sequence_tensor,
)
from chronaris.evaluation.application_tasks.task_aware_safe_residual_run import (
    CANDIDATES,
    DingxinSafeResidualConfig,
    _align_metric_values,
)
from chronaris.evaluation.application_tasks.task_aware_run_utils import preserved_elapsed


def test_zero_residual_projection_exactly_preserves_task_anchors() -> None:
    model = TaskSafeResidualHead(10, dropout=0.0)
    features = torch.randn(4, 10)
    maneuver_logits = torch.randn(4, 3)
    maneuver_score = torch.randn(4)
    response = torch.randn(4)
    risk_logit = torch.randn(4)

    maneuver = model.forward_maneuver(
        features,
        maneuver_logits,
        maneuver_score,
        score_scale=2.0,
    )
    physiology = model.forward_response(
        features,
        response,
        risk_logit,
        response_scale=3.0,
    )

    torch.testing.assert_close(maneuver["logits"], maneuver_logits)
    torch.testing.assert_close(maneuver["score"], maneuver_score)
    torch.testing.assert_close(physiology["response"], response)
    torch.testing.assert_close(physiology["risk_logit"], risk_logit)
    assert gate_is_non_degenerate(model.gate_values())


def test_sequence_summary_respects_non_contiguous_valid_mask() -> None:
    sequence = torch.tensor(
        [[[1.0, 10.0], [99.0, 99.0], [3.0, 30.0], [77.0, 77.0]]]
    )
    mask = torch.tensor([[True, False, True, False]])

    summary = summarize_sequence_tensor(sequence, mask)

    np.testing.assert_allclose(summary[0, :2], [2.0, 20.0])
    np.testing.assert_allclose(summary[0, 4:6], [3.0, 30.0])
    np.testing.assert_allclose(summary[0, 6:8], [2.0, 20.0])
    np.testing.assert_allclose(summary[0, 8:10], [2.0, 20.0])


def test_safe_gate_uses_all_three_preregistered_tolerances() -> None:
    direct = {
        "maneuver": {"macro_f1": 0.80},
        "response": {"rmse": 0.50},
        "high_response": {"auprc": 0.70},
    }
    safe = {
        "maneuver": {"macro_f1": 0.795},
        "response": {"rmse": 0.505},
        "high_response": {"auprc": 0.695},
    }
    unsafe = {
        **safe,
        "response": {"rmse": 0.506},
    }

    assert safe_against_anchor(safe, direct)
    assert not safe_against_anchor(unsafe, direct)


def test_metric_alignment_is_by_split_identity() -> None:
    full = pd.DataFrame({"split_id": ["b", "a"], "value": [2.0, 1.0]})
    direct = pd.DataFrame({"split_id": ["a", "b"], "value": [10.0, 20.0]})

    aligned = _align_metric_values(full, direct).set_index("split_id")

    assert aligned.loc["a", "full"] == 1.0
    assert aligned.loc["a", "direct"] == 10.0
    assert aligned.loc["b", "full"] == 2.0
    assert aligned.loc["b", "direct"] == 20.0


def test_candidate_budget_and_access_contract_are_bounded() -> None:
    config = DingxinSafeResidualConfig(device="cpu")

    assert len(CANDIDATES) <= 12
    assert config.run_id == "2026-07-15_dingxin-task-aware-safe-residual"
    assert "outer" not in config.__dataclass_fields__


class _PartialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.continuous_backbone = nn.Module()
        self.continuous_backbone.physiology_stream = nn.Module()
        self.continuous_backbone.physiology_stream.encoder = nn.Linear(2, 2)
        self.continuous_backbone.physiology_stream.decoder = nn.Linear(2, 2)
        self.continuous_backbone.vehicle_stream = nn.Module()
        self.continuous_backbone.vehicle_stream.encoder = nn.Linear(2, 2)
        self.causal_fusion = nn.Module()
        self.causal_fusion.scale_gate = nn.Linear(2, 2)
        self.causal_fusion.other = nn.Linear(2, 2)


def test_partial_unfreeze_scope_is_fail_closed() -> None:
    encoder = _PartialEncoder()

    selected = configure_partial_unfreeze(encoder)
    flags = {name: parameter.requires_grad for name, parameter in encoder.named_parameters()}

    assert selected
    assert flags["continuous_backbone.physiology_stream.encoder.weight"]
    assert flags["continuous_backbone.vehicle_stream.encoder.weight"]
    assert flags["causal_fusion.scale_gate.weight"]
    assert not flags["continuous_backbone.physiology_stream.decoder.weight"]
    assert not flags["causal_fusion.other.weight"]


def test_preserved_elapsed_keeps_completed_runtime(tmp_path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        '{"status":"task_aware_safe_residual_complete","elapsed_s":12.5}',
        encoding="utf-8",
    )

    assert preserved_elapsed(progress, 99.0) == 12.5
    assert preserved_elapsed(tmp_path / "missing.json", 99.0) == 99.0
