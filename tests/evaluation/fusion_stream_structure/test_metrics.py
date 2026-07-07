from __future__ import annotations

from chronaris.evaluation.fusion_stream_structure.contracts import FusionStreamRunConfig
from chronaris.evaluation.fusion_stream_structure.metrics import (
    compute_composite_scores,
    cp_tolerance_hit_rate,
    cross_view_segment_stability,
    discord_interval_overlap,
    motif_event_consistency,
    segment_event_purity,
)


def test_change_point_tolerance_hit_rate() -> None:
    boundaries = [False] * 50
    boundaries[20] = True
    boundaries[35] = True
    result = cp_tolerance_hit_rate([19, 36], boundaries, tol=1)
    assert result["status"] == "completed"
    assert result["value"] == 1.0


def test_segment_event_purity() -> None:
    labels = ["low"] * 10 + ["high"] * 10
    result = segment_event_purity([10], labels)
    assert result["status"] == "completed"
    assert result["value"] == 1.0


def test_cross_view_segment_stability_identical_is_one() -> None:
    result = cross_view_segment_stability(
        {"v1": [10, 20], "v2": [10, 20]},
        T=40,
    )
    assert result["status"] == "completed"
    assert result["value"] == 1.0


def test_motif_event_consistency_same_label_segments() -> None:
    labels = ["low"] * 10 + ["high"] * 10 + ["low"] * 10
    motif = {
        "left": {"start_index": 0, "end_index": 5},
        "right": {"start_index": 20, "end_index": 25},
    }
    result = motif_event_consistency(motif, labels)
    assert result["status"] == "completed"
    assert result["value"] == 1.0


def test_discord_overlap() -> None:
    interval = [False] * 20
    for index in range(8, 13):
        interval[index] = True
    result = discord_interval_overlap(
        {"start_index": 10, "end_index": 15},
        interval,
        metric_name="discord_maneuver_overlap",
    )
    assert result["status"] == "completed"
    assert result["value"] == 3 / 7


def test_composite_score_uses_fixed_weights() -> None:
    rows = [
        {"method_name": "chronaris", "metric": "cp_tolerance_hit_rate", "value": 1.0, "status": "completed"},
        {"method_name": "chronaris", "metric": "segment_event_purity", "value": 0.5, "status": "completed"},
        {"method_name": "chronaris", "metric": "motif_event_consistency", "value": 0.25, "status": "completed"},
        {"method_name": "chronaris", "metric": "discord_maneuver_overlap", "value": 0.75, "status": "completed"},
        {"method_name": "chronaris", "metric": "cross_view_segment_stability", "value": 0.8, "status": "completed"},
        {"method_name": "chronaris", "metric": "nn_segment_cross_view_consistency", "value": 0.6, "status": "completed"},
        {"method_name": "chronaris", "metric": "clap_state_replay_consistency", "value": 0.4, "status": "completed"},
    ]
    composites = compute_composite_scores(rows, config=FusionStreamRunConfig())
    event_score = next(row for row in composites if row["metric"] == "event_alignment_score")
    assert event_score["status"] == "completed"
    assert event_score["value"] == (1.0 + 0.5 + 0.25 + 0.75) / 4
    assert event_score["details"]["weights"]["cp_tolerance_hit_rate"] == 0.25
