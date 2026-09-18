"""Versioned runtime equivalence, separate from scientific selection thresholds."""
import torch

from chronaris.evaluation.application_tasks.checkpoint_performance import compare_values

POLICY = {
    'version': 'stage45_execution_v2',
    'state_atol': 1e-5, 'state_rtol': 1e-4,
    'representation_relative_rms': 1e-4,
    'representation_scaled_max': 1e-2,
    'representation_scale_floor': 1e-3,
    'prediction_relative_rms': 1e-4,
    'prediction_atol': 1e-5,
    'classification_changes': 0,
    'resume_exact': True,
}


def compare_runtime(expected, actual, *, representation=False):
    """Keep structure/masks exact; bound both distributed drift and sparse tails."""
    if not representation:
        return compare_values(expected, actual, atol=POLICY['state_atol'], rtol=POLICY['state_rtol'])
    report = dict(close=True, bitwise_equal=True, max_abs=0., failures=[], tensors={})

    def visit(left, right, path):
        if isinstance(left, torch.Tensor) and left.is_floating_point():
            if (not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype
                or not torch.isfinite(left).all() or not torch.isfinite(right).all()):
                report['failures'].append(path)
                report['bitwise_equal'] = False
                return
            if not left.numel():
                return
            a, b = left.double(), right.double()
            error = (a-b).abs()
            scale = max(float(a.square().mean().sqrt()), POLICY['representation_scale_floor'])
            rms, maximum = float(error.square().mean().sqrt())/scale, float(error.max())
            outliers = float((error > POLICY['state_atol'] + POLICY['state_rtol']*a.abs()).double().mean())
            report['tensors'][path] = dict(relative_rms=rms, max_abs=maximum, scale=scale, outlier_fraction=outliers)
            report['max_abs'] = max(report['max_abs'], maximum)
            report['bitwise_equal'] &= torch.equal(left, right)
            if rms > POLICY['representation_relative_rms'] or maximum/scale > POLICY['representation_scaled_max']:
                report['failures'].append(path)
        elif isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
            for key in left:
                visit(left[key], right[key], f'{path}/{key}')
        elif isinstance(left, (list, tuple)) and isinstance(right, type(left)) and len(left) == len(right):
            for i, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, f'{path}/{i}')
        else:
            check = compare_values(left, right, atol=0., rtol=0.)
            report['bitwise_equal'] &= check['bitwise_equal']
            if not check['close']:
                report['failures'].append(path)

    visit(expected, actual, '')
    report['close'] = not report['failures']
    return report
