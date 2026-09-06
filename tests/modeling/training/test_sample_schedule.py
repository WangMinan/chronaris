from collections import Counter

from chronaris.modeling.training.sample_schedule import BalancedSampleSchedule


def test_subjects_then_recordings_are_balanced_and_cursor_replay_is_exact():
    hierarchy = {"a0": ("a", "r0"), "a1": ("a", "r1"), "b0": ("b", "r0")}
    hierarchy.update({f"b{i}": ("b", "r1") for i in range(1, 6)})
    schedule = BalancedSampleSchedule(tuple(hierarchy), hierarchy)
    draws = sum((schedule.draw(start, 4) for start in range(0, 600, 4)), ())
    assert Counter(hierarchy[value][0] for value in draws) == {"a": 300, "b": 300}
    assert Counter(hierarchy[value] for value in draws) == {("a", "r0"): 150, ("a", "r1"): 150,
                                                         ("b", "r0"): 150, ("b", "r1"): 150}
    restored = BalancedSampleSchedule(tuple(reversed(hierarchy)), hierarchy)
    assert restored.sha256 == schedule.sha256
    assert restored.draw(164, 4) == draws[164:168]
