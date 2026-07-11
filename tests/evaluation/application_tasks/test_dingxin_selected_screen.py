from __future__ import annotations

import pytest

from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    _build_guarded_cached_provider,
)


def test_guarded_cached_provider_reuses_allowed_and_rejects_outer_test() -> None:
    calls = []

    def base(sample_ids):
        calls.append(tuple(sample_ids))
        return {"sample_ids": tuple(sample_ids)}

    provider, audit = _build_guarded_cached_provider(
        base,
        allowed_sample_ids=("train", "validation"),
        forbidden_sample_ids=("outer_test",),
    )

    assert provider(("train",)) is provider(("train",))
    assert calls == [("train",)]
    assert audit["cache_hit_count"] == 1
    with pytest.raises(ValueError, match="outer-test"):
        provider(("outer_test",))
    assert audit["forbidden_request_count"] == 1
