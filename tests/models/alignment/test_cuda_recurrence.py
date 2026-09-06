from copy import deepcopy

import pytest
import torch

from chronaris.models.alignment.cuda_recurrence import CUDAGraphRecurrence, _CellChunk
from chronaris.models.alignment.ode_cells import ODERNNCell


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
def test_chunk_replay_preserves_outputs_gradients_accumulation_and_resume():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    eager = ODERNNCell(8, hidden_dim=8, dynamics_hidden_dim=16, ode_method="euler").cuda()
    fast = deepcopy(eager)
    replay = CUDAGraphRecurrence(fast)
    optimizer = torch.optim.AdamW(fast.parameters(), lr=3e-4)
    expected_optimizer = torch.optim.AdamW(eager.parameters(), lr=3e-4)
    h = torch.randn(2, 8, device="cuda", requires_grad=True)
    x = torch.randn(2, 1027, 8, device="cuda", requires_grad=True)
    dt = torch.full((2, 1027), .005, dtype=torch.float64, device="cuda", requires_grad=True)
    mask = torch.ones(2, 1027, dtype=torch.bool, device="cuda")
    mask[0] = False
    mask[1, 200:400] = False
    inputs = h, x, dt
    reference_outputs, reference_gradients = [], []
    for cell, scan, opt in ((eager, _CellChunk(eager), expected_optimizer), (fast, replay, optimizer)):
        opt.zero_grad(set_to_none=True)
        for value in inputs:
            value.grad = None
        for micro in range(2):
            # Multiple outstanding forwards and unequal sequence lengths share a graph.
            first = scan(h, x + micro * .1, dt, mask)
            second = scan(h, x[:, :613] - .1, dt[:, :613], mask[:, :613])
            sum(value.square().mean() for value in (*first, *second)).backward()
            reference_outputs.append(tuple(value.detach().clone() for value in (*first, *second)))
        reference_gradients.append(tuple(value.grad.clone() for value in (*cell.parameters(), *inputs)))
        opt.step()
    for left, right in zip(reference_outputs[:2], reference_outputs[2:], strict=True):
        for expected, actual in zip(left, right, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    for expected, actual in zip(*reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    restored = deepcopy(eager)
    restored.load_state_dict(fast.state_dict())
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for cell, scan, opt in ((fast, replay, optimizer), (restored, CUDAGraphRecurrence(restored), resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        sum(value.square().mean() for value in scan(h, x, dt, mask)).backward()
        opt.step()
    for expected, actual in zip(fast.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(optimizer.state.values(), resumed_optimizer.state.values(), strict=True):
        for key in expected:
            assert torch.equal(expected[key], actual[key])

    # A fresh cache must also work while the encoder is frozen for head warmup.
    fast.requires_grad_(False)
    with torch.inference_mode():
        expected = _CellChunk(fast)(h, x, dt, mask)
        actual = CUDAGraphRecurrence(fast)(h, x, dt, mask)
        for left, right in zip(expected, actual, strict=True):
            assert torch.equal(left, right)
