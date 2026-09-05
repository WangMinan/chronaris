from chronaris.dataset.group_splits import split_group_train_validation


def test_group_train_validation_is_deterministic_and_disjoint() -> None:
    groups = ("a", "a", "b", "b", "c", "c", "d", "d")
    first = split_group_train_validation(range(len(groups)), groups, seed=17)
    second = split_group_train_validation(range(len(groups)), groups, seed=17)
    train, validation = first

    assert first == second
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(len(groups)))
    assert {groups[index] for index in train}.isdisjoint(
        {groups[index] for index in validation}
    )
