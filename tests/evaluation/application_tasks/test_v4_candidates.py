import pytest

from chronaris.evaluation.application_tasks.v4_candidates import CANDIDATE_CHANGES, BASELINE_CANDIDATES, candidate_options


def test_cohort_has_nine_single_changes_and_baseline_common_opportunities():
    assert len(CANDIDATE_CHANGES) == 10
    reference = candidate_options("chronaris", "reference")
    assert reference["hidden_dim"] == 32 and not reference["training"] and not reference["missingness_mixture"]
    for name in CANDIDATE_CHANGES:
        options = candidate_options("chronaris", name)
        groups_changed = sum((options["hidden_dim"] != 32, bool(options["training"]), options["missingness_mixture"]))
        assert groups_changed == (0 if name == "reference" else 1)
    for method in ("physiology_only", "vehicle_only", "mult", "contiformer"):
        for name in BASELINE_CANDIDATES:
            assert candidate_options(method, name) == candidate_options("chronaris", name)
        with pytest.raises(ValueError, match="structural"):
            candidate_options(method, "quality_gate")
    with pytest.raises(ValueError, match="outside"):
        candidate_options("chronaris", "unapproved_combination")
