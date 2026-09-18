import torch

from chronaris.evaluation.application_tasks.execution_equivalence import compare_runtime


def test_runtime_policy_bounds_sparse_and_distributed_drift_and_keeps_masks_exact():
    x = torch.ones(10000)
    sparse = x.clone(); sparse[0] += .001
    assert compare_runtime(x, sparse, representation=True)['close']
    assert not compare_runtime(x, x + .001, representation=True)['close']
    sparse[0] = 1.1
    assert not compare_runtime(x, sparse, representation=True)['close']
    assert not compare_runtime(torch.zeros(100), torch.full((100,), 1e-5), representation=True)['close']
    for changed in (torch.ones(100), torch.full_like(x, float('nan')), torch.ones_like(x, dtype=torch.float64)):
        assert not compare_runtime(x, changed, representation=True)['close']
    assert not compare_runtime({'mask': torch.tensor([True])}, {'mask': torch.tensor([False])}, representation=True)['close']
