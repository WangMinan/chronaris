"""CPU-only execution comparison; does not load observations or checkpoints."""
from dataclasses import fields
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import torch

from chronaris.modeling.fusion_encoders.chronaris_continuous import _causal_semantic_sequence
from chronaris.models.fusion.semantic_event import CausalEventFusion, CausalEventFusionConfig


def main():
    project = Path(__file__).resolve().parents[4]
    revision = "4c723e460c919545e1aa5997b9db6c26c58e4b6e"
    source = subprocess.check_output([
        "git", "show", revision + ":src/chronaris/models/fusion/semantic_event.py",
    ], cwd=project, text=True)
    legacy = types.ModuleType("chronaris_semantic_before")
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, "frozen_semantic_event.py", "exec"), legacy.__dict__)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    batch, points, hidden = 4, 96, 32
    config = CausalEventFusionConfig(state_dim=hidden, learnable_queries=True)
    current = CausalEventFusion(config)
    old = CausalEventFusion(config)
    old.load_state_dict(current.state_dict())
    old.extractor = legacy.EventTokenExtractor(config)
    base = {
        "physiology_states": torch.randn(batch, points, hidden),
        "vehicle_states": torch.randn(batch, points, hidden),
        "attention_weights": torch.rand(batch, points, points),
    }
    valid = torch.rand(batch, points) > .2
    valid[0] = False
    valid[-1] = True
    fixed = dict(physiology_valid=valid, vehicle_valid=valid,
                 query_times=torch.linspace(0., 30., points).expand(batch, -1))

    def run(model):
        model.zero_grad(set_to_none=True)
        inputs = {k: v.clone().requires_grad_() for k, v in base.items()}
        output, context = _causal_semantic_sequence(model, **inputs, **fixed)
        context.square().sum().backward()
        return output, context, inputs

    before, before_context, before_inputs = run(old)
    after, after_context, after_inputs = run(current)
    assert torch.equal(before_context, after_context)
    for field in fields(before):
        left, right = getattr(before, field.name), getattr(after, field.name)
        assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    for name in base:
        assert torch.equal(before_inputs[name].grad, after_inputs[name].grad)
    assert torch.equal(old.query_bank.query_residual.grad, current.query_bank.query_residual.grad)
    measurements = {}
    for name, model in (("dense", old), ("segment_max", current)):
        for _ in range(3):
            run(model)
        times = []
        for _ in range(15):
            start = time.perf_counter()
            run(model)
            times.append(time.perf_counter() - start)
        measurements[name] = dict(seconds=times, median_seconds=statistics.median(times))
    result = dict(
        scope="synthetic_cpu_semantic_prefix_forward_backward_only",
        reference_revision=revision, device="cpu", threads=1, torch_version=str(torch.__version__),
        batch=batch, reference_points=points, hidden_dim=hidden,
        warmups=3, repetitions=15, outputs_bitwise_equal=True, gradients_bitwise_equal=True,
        measurements=measurements,
        median_speedup=measurements["dense"]["median_seconds"] / measurements["segment_max"]["median_seconds"],
        gpu_verified=False, full_training_verified=False, production_adopted=False,
    )
    output = Path(__file__).with_name("cpu_semantic_benchmark.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
