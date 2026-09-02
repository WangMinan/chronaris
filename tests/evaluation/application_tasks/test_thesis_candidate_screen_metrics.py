from pathlib import Path

from chronaris.evaluation.application_tasks.thesis_candidate_screen_metrics import (
    audit_thesis_candidate_screen,
)


def test_candidate_screen_audit_selects_only_candidate_passing_both_gates(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    rows = []
    stage_folds = {"simulation": 1, "dingxin": 5, "cogpilot": 1, "clare": 5}
    for stage, folds in stage_folds.items():
        for fold in range(folds):
            for seed_index, seed in enumerate((17, 29, 43)):
                for candidate in (
                    "base",
                    "explicit_shift",
                    "semantic_pair",
                    "both_objectives",
                ):
                    semantic = candidate in ("semantic_pair", "both_objectives")
                    physical_active = stage != "clare"
                    rows.append(
                        {
                            "unit": f"{stage}:{fold}:{seed}:{candidate}",
                            "stage": stage,
                            "fold": fold,
                            "seed": seed,
                            "candidate": candidate,
                            "balanced_shift_accuracy": (
                                0.2 + 0.1 * (seed_index > 0)
                                if candidate == "both_objectives"
                                else 0.2
                            ),
                            "pair_positive_similarity": 0.8 if semantic else None,
                            "pair_negative_similarity": 0.6 if semantic else None,
                            "pair_recall_at_1": 0.9 if semantic else None,
                            "validation_self_supervised_loss": 1.0,
                            "parameter_count": 100 + int(semantic),
                            "training_elapsed_s": 2.0 + int(semantic),
                            "checkpoint_path": str(checkpoint),
                            "mechanism_terms": [
                                {
                                    "term_name": "chronaris_physical_consistency",
                                    "status": (
                                        "active" if physical_active else "unavailable"
                                    ),
                                    "raw_loss": 0.1 if physical_active else None,
                                }
                            ],
                            "validation_maneuver_macro_f1": 0.5,
                            "validation_response_rmse": 1.0,
                            "validation_response_spearman": 0.2,
                            "validation_macro_f1": 0.4,
                            "validation_balanced_accuracy": 0.5,
                            "validation_score_rmse": 1.2,
                            "validation_score_spearman": 0.1,
                        }
                    )
    state = {
        "protocol_version": "v3.2",
        "source_commit": "a" * 40,
        "source_code_sha256": "b" * 64,
        "runner_sha256": "c" * 64,
        "outer_results_opened": False,
        "completed_for_requested_stages": True,
        "rows": rows,
    }

    audit = audit_thesis_candidate_screen(state)

    assert audit["protocol_gate_passed"] is True
    assert audit["physical_scope_gate_passed"] is True
    assert audit["explicit_shift_gate"]["passed"] is True
    assert audit["event_pair_gate"]["passed"] is True
    assert audit["selected_candidate"] == "both_objectives"
    assert len(audit["application_metric_rows"]) == 36
