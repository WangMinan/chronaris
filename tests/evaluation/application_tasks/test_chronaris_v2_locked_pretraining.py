from __future__ import annotations

import json

import pytest

from chronaris.evaluation.application_tasks.chronaris_v2_locked_dingxin_pretraining import (
    _load_locked_candidate,
)
from chronaris.evaluation.application_tasks.dingxin_locked_representation_run import (
    _locked_v2_candidate_id,
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
