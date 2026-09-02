from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    SimulationLockedPretrainingConfig,
    _auxiliary_schedule_matches_config,
)


def test_locked_schedule_follows_objective_configuration_from_first_epoch():
    config = SimulationLockedPretrainingConfig(
        chronaris_explicit_shift_weight=0.1,
        chronaris_event_pair_weight=0.1,
    )
    rows = [
        {"term_name": name, "weight": weight}
        for name, weight in (
            ("chronaris_continuous_alignment", 0.04),
            ("chronaris_physical_consistency", 0.02),
            ("chronaris_causal_direction", 0.02),
            ("explicit_time_shift", 0.02),
            ("event_response_pairing", 0.02),
        )
    ]

    assert _auxiliary_schedule_matches_config(config, rows)
    assert not _auxiliary_schedule_matches_config(
        config,
        [row for row in rows if row["term_name"] != "event_response_pairing"],
    )
