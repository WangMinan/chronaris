from chronaris.evaluation.application_tasks.core_recovery_confirmation import (
    _aggregate_metrics,
)


def test_aggregate_metrics_ranks_each_task_in_the_declared_direction():
    rows = []
    values = {
        ("chronaris", "maneuver", "higher"): 0.8,
        ("baseline", "maneuver", "higher"): 0.7,
        ("chronaris", "response", "lower"): 0.3,
        ("baseline", "response", "lower"): 0.4,
    }
    for (method, task, direction), value in values.items():
        for seed in (17, 29):
            rows.append(
                {
                    "seed": seed,
                    "fold_id": "fold01",
                    "method_name": method,
                    "task": task,
                    "metric": "score",
                    "direction": direction,
                    "value": value,
                }
            )

    _, overall = _aggregate_metrics(rows)

    ranks = {(row["method_name"], row["task"]): row["rank"] for row in overall}
    assert ranks[("chronaris", "maneuver")] == 1
    assert ranks[("baseline", "maneuver")] == 2
    assert ranks[("chronaris", "response")] == 1
    assert ranks[("baseline", "response")] == 2
