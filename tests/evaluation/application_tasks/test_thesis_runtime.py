import pytest

from chronaris.evaluation.application_tasks.thesis_runtime import (
    summarize_runtime_timings,
)


def test_runtime_timing_summary_uses_batch_throughput():
    row = summarize_runtime_timings((0.01, 0.02, 0.03), batch_size=8)

    assert row["latency_p50_ms"] == pytest.approx(20.0)
    assert row["throughput_samples_per_s"] == pytest.approx(400.0)
