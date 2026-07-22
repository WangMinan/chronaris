from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from chronaris.evaluation.application_tasks.core_feasibility_models import FoldTargets
from chronaris.evaluation.application_tasks.residual_activation_contracts import (
    CANDIDATES,
    build_selective_teacher_targets,
)
from chronaris.evaluation.application_tasks.residual_activation_model import (
    ActivatedResidualHead,
)
from chronaris.evaluation.application_tasks.residual_activation_optimization import (
    compose_task_checkpoint,
)
from chronaris.evaluation.application_tasks.residual_activation_run import (
    aggregate_activation_candidates,
)
from chronaris.evaluation.application_tasks.task_aware_safe_residual_model import (
    PredictionPair,
    SafeAnchorPredictions,
)


def test_candidate_grid_is_bounded_and_contains_teacher_assisted_paths() -> None:
    assert len(CANDIDATES) == 8
    assert len({candidate.candidate_id for candidate in CANDIDATES}) == 8
    assert any(candidate.use_distillation for candidate in CANDIDATES)
    assert any(candidate.gate_mode == "conditional" for candidate in CANDIDATES)
    assert any(candidate.adapter_mode == "task_specific" for candidate in CANDIDATES)


def test_fixed_gate_activates_small_random_residual_without_leaving_bound() -> None:
    candidate = next(
        value for value in CANDIDATES if value.candidate_id == "fixed025_residual_task"
    )
    torch.manual_seed(4)
    model = ActivatedResidualHead(
        10,
        candidate=candidate,
        hidden_dim=8,
        dropout=0.0,
        maximum_gate=0.5,
    )
    features = torch.randn(6, 10)
    logits = torch.randn(6, 3)
    score = torch.randn(6)

    output = model.forward_maneuver(
        features,
        logits,
        score,
        score_scale=2.0,
        release_gate=True,
    )

    torch.testing.assert_close(output["maneuver_gate"], torch.full((6,), 0.25))
    assert float(output["maneuver_correction"].detach().abs().max()) > 0
    assert float(output["maneuver_gate"].detach().max()) <= 0.5


def test_conditional_gate_uses_sample_features_and_stays_bounded() -> None:
    candidate = next(
        value
        for value in CANDIDATES
        if value.candidate_id == "conditional015_residual_task"
    )
    model = ActivatedResidualHead(
        5,
        candidate=candidate,
        hidden_dim=6,
        dropout=0.0,
        maximum_gate=0.5,
    )
    gate = model.forward_response(
        torch.randn(7, 5),
        torch.randn(7),
        torch.randn(7),
        response_scale=1.5,
        release_gate=True,
    )["response_gate"]

    assert gate.shape == (7,)
    assert bool(torch.all(gate > 0))
    assert bool(torch.all(gate < 0.5))


def test_task_checkpoint_composition_keeps_one_model_with_independent_heads() -> None:
    base = {
        "adapters.maneuver.network.1.weight": torch.tensor([0.0]),
        "adapters.response.network.1.weight": torch.tensor([0.0]),
        "delta_layers.high_response.weight": torch.tensor([0.0]),
    }
    states = {
        "maneuver": {
            **base,
            "adapters.maneuver.network.1.weight": torch.tensor([1.0]),
        },
        "response": {
            **base,
            "adapters.response.network.1.weight": torch.tensor([2.0]),
        },
        "high_response": {
            **base,
            "delta_layers.high_response.weight": torch.tensor([3.0]),
        },
    }

    result = compose_task_checkpoint(base_state=base, task_states=states)

    assert float(result["adapters.maneuver.network.1.weight"]) == 1.0
    assert float(result["adapters.response.network.1.weight"]) == 2.0
    assert float(result["delta_layers.high_response.weight"]) == 3.0


def test_selective_teacher_targets_use_only_train_prediction_shapes() -> None:
    targets = _targets()
    anchor = _anchor(offset=0.0)
    teachers = {
        "vehicle_only": _anchor(offset=0.2),
        "contiformer": _anchor(offset=0.3),
        "mult": _anchor(offset=-0.1),
        "naive_time_sync": _anchor(offset=0.1),
    }

    result = build_selective_teacher_targets(
        method_anchors=teachers,
        safe_anchor=anchor,
        targets=targets,
    )

    assert result.maneuver_probability.shape == (6, 3)
    assert result.response.shape == (6,)
    assert result.high_response_probability.shape == (6,)
    assert np.all((result.maneuver_weight >= 0) & (result.maneuver_weight <= 1))
    assert np.all(
        (result.high_response_probability >= 0)
        & (result.high_response_probability <= 1)
    )


def test_aggregate_requires_activation_safety_and_two_task_improvement() -> None:
    metrics = []
    activations = []
    gates = []
    passing_id = "conditional015_residual_task_distill"
    for candidate in CANDIDATES:
        for split_index in range(6):
            split_id = f"split_{split_index}"
            passing = candidate.candidate_id == passing_id
            direct_f1 = 0.90
            full_f1 = 0.91 if passing else 0.90
            direct_rmse = 1.0
            full_rmse = 0.90 if passing else 1.0
            for variant, f1, rmse, ratio, skill, ap, normalized in (
                ("direct_only", direct_f1, direct_rmse, 1.0, 0.0, 0.70, 0.20),
                (
                    "full",
                    full_f1,
                    full_rmse,
                    0.90 if passing else 1.0,
                    0.10 if passing else 0.0,
                    0.75 if passing else 0.70,
                    0.45 if passing else 0.20,
                ),
            ):
                metrics.extend(
                    _metric_rows(
                        split_id,
                        candidate.candidate_id,
                        variant,
                        f1,
                        rmse,
                        ratio,
                        skill,
                        ap,
                        normalized,
                    )
                )
            activations.append(
                {
                    "split_id": split_id,
                    "candidate_id": candidate.candidate_id,
                    "best_epoch": 4 if passing else 0,
                    "maximum_prediction_difference": 0.2 if passing else 0.0,
                    "corrected_sample_fraction": 0.60 if passing else 0.0,
                    "median_contribution_ratio": 0.10 if passing else 0.0,
                }
            )
            for task in ("maneuver", "maneuver_score", "response", "high_response"):
                gates.append(
                    {
                        "split_id": split_id,
                        "candidate_id": candidate.candidate_id,
                        "task": task,
                        "lower_saturation_fraction": 0.0,
                        "upper_saturation_fraction": 0.0,
                    }
                )

    rows = aggregate_activation_candidates(
        pd.DataFrame(metrics), pd.DataFrame(activations), pd.DataFrame(gates)
    )
    selected = next(row for row in rows if row["candidate_id"] == passing_id)

    assert selected["activation_gate_passed"] is True
    assert selected["safety_gate_passed"] is True
    assert selected["research_gate_passed"] is True
    assert selected["improved_task_count"] >= 2


def _targets() -> FoldTargets:
    return FoldTargets(
        train_maneuver=np.array([0, 1, 2, 0, 1, 2]),
        validation_maneuver=np.array([0, 1, 2]),
        train_maneuver_score=np.linspace(0.0, 1.0, 6),
        validation_maneuver_score=np.linspace(0.1, 0.9, 3),
        train_response=np.linspace(0.2, 1.2, 6),
        validation_response=np.linspace(0.3, 1.0, 3),
        train_high_response=np.array([0, 0, 1, 0, 1, 1]),
        validation_high_response=np.array([0, 1, 1]),
    )


def _anchor(offset: float) -> SafeAnchorPredictions:
    logits = np.array(
        [
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [-1.0, 0.0, 2.0],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [-1.0, 0.0, 2.0],
        ]
    ) + offset
    return SafeAnchorPredictions(
        maneuver_logits=PredictionPair(logits, logits[:3]),
        maneuver_score=PredictionPair(
            np.linspace(0.0, 1.0, 6) + offset,
            np.linspace(0.1, 0.9, 3) + offset,
        ),
        response=PredictionPair(
            np.linspace(0.2, 1.2, 6) + offset,
            np.linspace(0.3, 1.0, 3) + offset,
        ),
        high_response_logit=PredictionPair(
            np.array([-2.0, -1.0, 1.0, -1.0, 1.0, 2.0]) + offset,
            np.array([-1.0, 1.0, 2.0]) + offset,
        ),
    )


def _metric_rows(
    split_id,
    candidate_id,
    variant,
    f1,
    rmse,
    ratio,
    skill,
    auprc,
    normalized_ap,
):
    return [
        {
            "split_id": split_id,
            "candidate_id": candidate_id,
            "variant": variant,
            "task": "maneuver",
            "metric": "macro_f1",
            "value": f1,
        },
        {
            "split_id": split_id,
            "candidate_id": candidate_id,
            "variant": variant,
            "task": "response",
            "metric": "rmse",
            "value": rmse,
        },
        {
            "split_id": split_id,
            "candidate_id": candidate_id,
            "variant": variant,
            "task": "response",
            "metric": "rmse_ratio",
            "value": ratio,
        },
        {
            "split_id": split_id,
            "candidate_id": candidate_id,
            "variant": variant,
            "task": "response",
            "metric": "response_skill",
            "value": skill,
        },
        {
            "split_id": split_id,
            "candidate_id": candidate_id,
            "variant": variant,
            "task": "high_response",
            "metric": "auprc",
            "value": auprc,
        },
        {
            "split_id": split_id,
            "candidate_id": candidate_id,
            "variant": variant,
            "task": "high_response",
            "metric": "normalized_ap",
            "value": normalized_ap,
        },
    ]
