from __future__ import annotations

import json

import pytest
import torch

from chronaris.evaluation.application_tasks.chronaris_v2_locked_dingxin_pretraining import (
    _load_locked_candidate,
)
from chronaris.evaluation.application_tasks.dingxin_locked_representation_run import (
    _locked_v2_candidate_id,
)
from chronaris.evaluation.application_tasks.simulation_locked_representation_run import (
    _locked_v2_candidate_id as _simulation_locked_v2_candidate_id,
)
from chronaris.evaluation.application_tasks.simulation_locked_consumer_run import (
    _checkpoint_paths as _simulation_consumer_checkpoint_paths,
)
from chronaris.evaluation.application_tasks.simulation_stress_consumer_run import (
    _checkpoint_paths as _simulation_stress_checkpoint_paths,
)
from chronaris.modeling.training import TRAINABLE_FUSION_METHODS
from chronaris.evaluation.application_tasks.chronaris_v2_simulation_ablation_pretraining import (
    ChronarisV2SimulationAblationConfig,
)
from chronaris.evaluation.application_tasks.chronaris_v2_simulation_ablation_representations import (
    ChronarisV2SimulationAblationRepresentationConfig,
    _require_checkpoints as _require_v2_ablation_checkpoints,
)


def test_locked_candidate_requires_clean_task_independent_lock(tmp_path) -> None:
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "format": "chronaris.v2_locked_configuration.v1",
                "configuration_locked": True,
                "selection_uses_downstream_labels": False,
                "outer_test_opened": False,
                "candidate": {
                    "candidate_id": "locked",
                    "internal_hidden_dim": 96,
                    "lag_mode": "continuous_basis",
                    "ode_method": "euler",
                    "learning_rate": 1e-3,
                    "physiology_residual_mode": "direct_causal_query",
                },
            }
        ),
        encoding="utf-8",
    )

    candidate, _payload = _load_locked_candidate(path)

    assert candidate.candidate_id == "locked"
    assert candidate.physiology_residual_mode == "direct_causal_query"
    assert _locked_v2_candidate_id(path) == "locked"
    assert _simulation_locked_v2_candidate_id(path) == "locked"


def test_locked_candidate_rejects_outer_test_opened_lock(tmp_path) -> None:
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "format": "chronaris.v2_locked_configuration.v1",
                "configuration_locked": True,
                "selection_uses_downstream_labels": False,
                "outer_test_opened": True,
                "candidate": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="clean locked configuration"):
        _load_locked_candidate(path)


def test_simulation_consumer_checkpoint_set_can_mix_v1_baselines_and_v2(tmp_path) -> None:
    v2_root = tmp_path / "v2"
    baseline_root = tmp_path / "baseline"
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(
        json.dumps(
            {method: {"candidate_id": f"{method}_selected"}
             for method in TRAINABLE_FUSION_METHODS}
        ),
        encoding="utf-8",
    )
    seed = 17
    for method in TRAINABLE_FUSION_METHODS:
        if method == "chronaris":
            path = v2_root / "checkpoints" / f"seed_{seed}" / method / "v2_locked" / "last.pt"
            payload = {"training_status": "completed", "config": {"seed": seed}}
        else:
            path = baseline_root / "checkpoints" / f"seed_{seed}" / method / f"{method}_selected" / "best.pt"
            payload = {"training_status": "completed", "seed": seed}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    paths = _simulation_consumer_checkpoint_paths(
        v2_root,
        selected_path,
        (seed,),
        baseline_root=baseline_root,
        locked_candidate_id="v2_locked",
    )

    assert paths[(seed, "chronaris")].name == "last.pt"
    assert paths[(seed, "mult")].is_relative_to(baseline_root)
    stress_paths = _simulation_stress_checkpoint_paths(
        v2_root,
        selected_path,
        (seed,),
        baseline_root=baseline_root,
        locked_candidate_id="v2_locked",
    )
    assert stress_paths == paths


def test_v2_formal_ablation_checkpoint_is_structure_bound(tmp_path) -> None:
    variant = "no_corrected_physics"
    root = tmp_path / "run"
    path = (
        root / "checkpoints" / "seed_17" / variant / "chronaris"
        / f"locked__{variant}" / "last.pt"
    )
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "training_status": "completed",
            "config": {"seed": 17},
            "candidate_config": {
                "candidate_id": f"locked__{variant}",
                "structure_candidate_id": "structure_07_missing_curriculum",
            },
        },
        path,
    )
    config = ChronarisV2SimulationAblationRepresentationConfig(
        seeds=(17,), variants=(variant,)
    )

    checkpoints = _require_v2_ablation_checkpoints(root, config)

    assert checkpoints[(17, variant)] == path
    with pytest.raises(ValueError, match="unsupported"):
        ChronarisV2SimulationAblationConfig(variants=("invented",))
