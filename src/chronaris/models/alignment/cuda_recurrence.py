"""Replay the existing Euler cell in fixed chunks without retaining graph buffers.

The observation recurrence stays sequential. Backward recomputes each chunk and
copies its gradients before another replay can overwrite CUDA's static storage.
"""
import torch
from torch import nn
from torch.nn import functional as F


class _CellChunk(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, hidden, embeddings, deltas, mask):
        evolved_rows, updated_rows = [], []
        for index in range(embeddings.shape[1]):
            evolved, hidden = self.cell(hidden, deltas[:, index], embeddings[:, index], mask[:, index])
            evolved_rows.append(evolved)
            updated_rows.append(hidden)
        return torch.stack(evolved_rows, 1), torch.stack(updated_rows, 1), hidden.clone()


class _RecomputedReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, replay, hidden, embeddings, deltas, mask, *parameters):
        ctx.replay = replay
        ctx.save_for_backward(hidden, embeddings, deltas, mask, *parameters)
        return tuple(value.clone() for value in replay(hidden, embeddings, deltas, mask))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *output_gradients):
        hidden, embeddings, deltas, mask, *parameters = ctx.saved_tensors
        inputs = tuple(value.detach().requires_grad_() for value in (hidden, embeddings, deltas))
        with torch.enable_grad():
            outputs = ctx.replay(*inputs, mask)
            gradients = torch.autograd.grad(outputs, (*inputs, *parameters), output_gradients)
        # Returning static graph gradients aliases AccumulateGrad's first buffer;
        # the next replay would silently overwrite previously accumulated work.
        gradients = tuple(value.clone() for value in gradients)
        return (None, *gradients[:3], None, *gradients[3:])


class CUDAGraphRecurrence:
    """An execution cache, deliberately absent from the model's state_dict."""
    chunk_size = 512

    def __init__(self, cell):
        if cell.ode_method != "euler" or cell.max_ode_step_s is not None:
            raise ValueError("CUDA recurrence requires the original single-step Euler cell")
        self.cell = cell
        self.graphs = {}

    def __call__(self, hidden, embeddings, deltas, mask):
        if embeddings.device.type != "cuda" or embeddings.dtype != torch.float32 or torch.is_autocast_enabled("cuda"):
            raise ValueError("CUDA recurrence requires FP32 without autocast")
        parameters = tuple(value for value in self.cell.parameters() if value.requires_grad)
        key = (embeddings.device, embeddings.shape[0], deltas.dtype,
               torch.cuda.current_stream().cuda_stream,
               tuple((value.data_ptr(), value.requires_grad) for value in self.cell.parameters()))
        padding = (-embeddings.shape[1]) % self.chunk_size
        point_count = embeddings.shape[1]
        embeddings = F.pad(embeddings, (0, 0, 0, padding))
        deltas, mask = F.pad(deltas, (0, padding)), F.pad(mask, (0, padding))
        if key not in self.graphs:
            # Capture also supports an enclosing inference_mode/no_grad context.
            with torch.inference_mode(False), torch.enable_grad():
                sample = tuple(value.detach().clone().requires_grad_() for value in
                               (hidden, embeddings[:, :self.chunk_size], deltas[:, :self.chunk_size]))
                self.graphs[key] = torch.cuda.make_graphed_callables(
                    _CellChunk(self.cell), (*sample, mask[:, :self.chunk_size].clone()), num_warmup_iters=1)
        replay = self.graphs[key]
        evolved_rows, updated_rows = [], []
        for start in range(0, embeddings.shape[1], self.chunk_size):
            args = (hidden, embeddings[:, start:start + self.chunk_size],
                    deltas[:, start:start + self.chunk_size], mask[:, start:start + self.chunk_size])
            if torch.is_grad_enabled():
                evolved, updated, hidden = _RecomputedReplay.apply(replay, *args, *parameters)
            else:
                evolved, updated, hidden = (value.clone() for value in replay(*args))
            evolved_rows.append(evolved)
            updated_rows.append(updated)
        return torch.cat(evolved_rows, 1)[:, :point_count], torch.cat(updated_rows, 1)[:, :point_count], hidden
