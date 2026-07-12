from __future__ import annotations

import hashlib
import json

import pytest

from chronaris.evaluation.application_tasks.chronaris_v2_protocol import (
    PRIMARY_METRIC_THRESHOLDS,
    SealedConfirmationManifest,
    audit_v2_promotion,
)
from chronaris.evaluation.application_tasks.chronaris_v2_hyperparameter_screen import (
    ChronarisV2HyperparameterScreenConfig,
    _assert_complete_v2_passed_structure_gate,
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
