from __future__ import annotations

import hashlib
import json

import pytest

from chronaris.evaluation.application_tasks.chronaris_v2_protocol import (
    PRIMARY_METRIC_THRESHOLDS,
    SealedConfirmationManifest,
    authorize_sealed_confirmation,
    audit_v2_promotion,
)
from chronaris.evaluation.application_tasks.chronaris_v2_hyperparameter_screen import (
    ChronarisV2HyperparameterScreenConfig,
    _assert_complete_v2_passed_structure_gate,
)
from chronaris.evaluation.application_tasks.chronaris_v2_structure_diagnostics_run import (
    _final_training_checkpoint,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_sealed_confirmation_denies_development_access() -> None:
    manifest = SealedConfirmationManifest(
        family_id="confirmation_family_2026_07_12",
        generator_protocol_sha256=_sha("protocol"),
        payload_sha256=_sha("payload"),
        sealed_before_configuration_lock=True,
    )
    with pytest.raises(PermissionError, match="unavailable"):
        manifest.assert_access_allowed(
            phase="development",
            configuration_locked=False,
        )


def test_promotion_is_fail_closed_when_one_metric_misses_threshold() -> None:
    rows = []
    for metric, (direction, threshold) in PRIMARY_METRIC_THRESHOLDS.items():
        passing = threshold + 0.01 if direction == "higher" else threshold - 0.01
        if metric == "simulation_load_rmse":
            passing = threshold + 0.01
        for seed in (17, 29, 43):
            rows.append(
                {
                    "metric_name": metric,
                    "seed": seed,
                    "value": passing,
                    "rank_first": True,
                }
            )
    gates = {
        "single_stream_no_harm": True,
        "class_recall_gap_within_0_05": True,
        "time_mechanism_within_10_percent": True,
        "missingness_slopes_top_three": True,
        "causal_future_invariance": True,
        "invalid_queries_excluded_from_pooling": True,
    }
    audit = audit_v2_promotion(rows, additional_gates=gates)
    assert not audit["promoted"]
    assert audit["paper_main_model"] == "chronaris_v1"
    assert not audit["confirmed_v1_evidence_changed"]


def test_hyperparameter_screen_cannot_start_before_structure_gate(tmp_path) -> None:
    root = tmp_path / "structure"
    root.mkdir()
    (root / "evidence_manifest.json").write_text(
        json.dumps({"status": "gates_failed"}),
        encoding="utf-8",
    )
    (root / "task_independent_ranking.csv").write_text(
        "candidate_id,gate_passed\nstructure_08_complete_v2,False\n",
        encoding="utf-8",
    )
    config = ChronarisV2HyperparameterScreenConfig(
        compact_output_root=str(tmp_path),
        structure_diagnostics_run_id="structure",
        device="cpu",
        max_candidates=1,
        max_epochs=1,
    )

    with pytest.raises(PermissionError, match="closed"):
        _assert_complete_v2_passed_structure_gate(config)


def test_confirmation_unlock_requires_a_task_independent_locked_config(tmp_path) -> None:
    sealed = SealedConfirmationManifest(
        family_id="family",
        generator_protocol_sha256=_sha("protocol"),
        payload_sha256=_sha("payload"),
        sealed_before_configuration_lock=True,
    )
    sealed_path = tmp_path / "sealed.json"
    sealed_path.write_text(json.dumps(sealed.to_dict()), encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "format": "chronaris.v2_locked_configuration.v1",
                "configuration_locked": False,
                "selection_uses_downstream_labels": False,
                "outer_test_opened": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="locked configuration"):
        authorize_sealed_confirmation(
            sealed_manifest_path=sealed_path,
            locked_configuration_path=lock_path,
            output_path=tmp_path / "access.json",
        )


def test_structure_gate_uses_final_training_state_not_public_loss_best(tmp_path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    best = root / "best.pt"
    last = root / "last.pt"
    best.write_bytes(b"public-best")
    last.write_bytes(b"final-state")

    assert _final_training_checkpoint(best) == last
