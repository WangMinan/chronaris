import torch

from chronaris.evaluation.application_tasks.checkpoint_performance import compare_values


def test_comparison_rejects_missing_nonfinite_or_changed_training_state():
    original = {'optimizer': [torch.tensor([1., 2.])], 'cursor': 200}
    assert compare_values(original, original)['bitwise_equal']
    changed = {'optimizer': [torch.tensor([1., 2.+1e-6])], 'cursor': 200}
    report = compare_values(original, changed)
    assert report['close'] and not report['bitwise_equal']
    for actual in ({}, {'optimizer': [torch.tensor([1., float('nan')])], 'cursor': 200},
                   {'optimizer': [torch.tensor([1., 2.])], 'cursor': 201}):
        assert not compare_values(original, actual)['close']
