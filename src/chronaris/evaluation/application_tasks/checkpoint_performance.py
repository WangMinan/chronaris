"""Bounded execution trials from a preserved checkpoint; no production migration."""
from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import torch

from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.chronaris_continuous import (
    ChronarisContinuousEncoderConfig, ChronarisContinuousFusionEncoder,
)
from chronaris.modeling.training.candidate_config import CandidateScreenConfig
from chronaris.modeling.training.candidate_checkpoint import candidate_data_sha256, candidate_source_code_sha256
from chronaris.modeling.training.candidate_step import pretext_micro_step
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.pretext import CommonPretextHeadBundle
from chronaris.modeling.training.pretraining_encoders import TrainableFusionEncoder
from chronaris.modeling.training.rng import capture_rng_state, restore_rng_state, isolated_training_rng, canonical_training_state_sha256
from chronaris.modeling.training.sample_schedule import training_sample_schedule
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def compare_values(expected, actual, *, atol=1e-6, rtol=1e-5):
    """Report every failed tensor path, including nonfinite and missing state."""
    result = dict(bitwise_equal=True, close=True, max_abs=0., failures=[])

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            same = isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
            exact = same and torch.equal(left, right)
            close = same and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            if close:
                if left.numel():
                    result['max_abs'] = max(result['max_abs'], float((left.double()-right.double()).abs().max()))
                close = torch.allclose(left, right, atol=atol, rtol=rtol) if left.is_floating_point() else exact
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                exact = close = False
            else:
                for key in left:
                    visit(left[key], right[key], f'{path}/{key}')
                return
        elif isinstance(left, (list, tuple)):
            if not isinstance(right, type(left)) or len(left) != len(right):
                exact = close = False
            else:
                for i, (a, b) in enumerate(zip(left, right, strict=True)):
                    visit(a, b, f'{path}/{i}')
                return
        else:
            exact = close = left == right
        result['bitwise_equal'] &= bool(exact)
        result['close'] &= bool(close)
        if not close:
            result['failures'].append(path)

    visit(expected, actual, '')
    return result


def compare_trials(expected_root, actual_root, *, exact=False):
    expected_root, actual_root = Path(expected_root), Path(actual_root)
    result = {}
    paths = sorted(actual_root.glob('state_*.pt'))
    if not paths:
        raise ValueError('trial produced no state')
    for path in paths:
        update = int(path.stem.split('_')[-1])
        left, right = (torch.load(root/path.name, map_location='cpu', weights_only=True)
                       for root in (expected_root, actual_root))
        if left['parent_sha256'] != right['parent_sha256']:
            raise ValueError('trial checkpoint lineage differs')
        checks = {}
        for key in ('encoder_state_dict', 'head_state_dict', 'optimizer_state_dict', 'rng_state', 'data_cursor'):
            checks[key] = compare_values(left[key], right[key], atol=0 if exact else 1e-6, rtol=0 if exact else 1e-5)
        left, right = (torch.load(root/f'observations_{update}.pt', map_location='cpu', weights_only=True)
                       for root in (expected_root, actual_root))
        for key in left:
            checks[key] = compare_values(left[key], right[key], atol=0 if exact else 1e-6,
                rtol=0 if exact or key == 'outputs' else 1e-5)
        result[str(update)] = checks
    return dict(passed=all(c['close'] for checks in result.values() for c in checks.values()),
        exact_required=exact, comparisons=result)


def _restore(payload, graph):
    config = ChronarisContinuousEncoderConfig.from_checkpoint_dict(payload['encoder_manifest']['backbone_config'])
    encoder = TrainableFusionEncoder(method_name='chronaris', backbone=ChronarisContinuousFusionEncoder(
        replace(config, cuda_graph_recurrence=graph))).cuda().train()
    encoder.load_state_dict(payload['encoder_state_dict'], strict=True)
    heads = CommonPretextHeadBundle(representation_dim=64,
        target_feature_count=len(payload['physiology_feature_names'])+len(payload['vehicle_feature_names']),
        **payload['pretext_head_config']).cuda().train()
    heads.load_state_dict(payload['head_state_dict'], strict=True)
    parameters = tuple(encoder.parameters())+tuple(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=payload['candidate_config']['learning_rate'],
        weight_decay=payload['config']['weight_decay'])
    optimizer.load_state_dict(deepcopy(payload['optimizer_state_dict']))
    restore_rng_state(payload['rng_state'])
    return encoder, heads, parameters, optimizer


def trial(*, checkpoint, pipeline_config, output_root, graph, threads=1, updates=2, resume_state=None):
    if updates not in (1, 2) or threads not in (1, 4, 8):
        raise ValueError('performance trials require 1-2 updates and 1/4/8 CPU threads')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    original_hash = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state_hash = canonical_training_state_sha256(*(payload[k] for k in ('encoder_state_dict',
        'head_state_dict', 'explicit_time_shift_head_state_dict', 'optimizer_state_dict', 'rng_state')))
    if state_hash != payload['canonical_training_state_sha256']:
        raise ValueError('preserved checkpoint training state changed')
    if (payload['optimizer_updates'] != 200 or payload['method_name'] != 'chronaris'
        or payload['source_code_sha256'] != candidate_source_code_sha256()
        or payload['chronaris_mechanism_enabled'] or payload['chronaris_lag_aware_weight']
        or payload['explicit_time_shift_head_state_dict'] is not None):
        raise ValueError('trial requires the unchanged-source stage4 update-200 checkpoint')
    config = CandidateScreenConfig(**payload['config'])
    if config.batch_size != 4 or config.effective_batch_size != 16 or config.cuda_graph_recurrence:
        raise ValueError('unexpected preserved execution configuration')
    pipeline = json.loads(Path(pipeline_config).read_text())
    torch.set_num_threads(threads)
    with development_gpu_lock() as acquired, _periodic_training_heartbeat('checkpoint_performance', 30., root=root) as progress:
        if not acquired:
            raise RuntimeError('production still holds the GPU lock')
        with isolated_training_rng(config.seed):
            progress.update(phase='verify_inputs', graph=graph, threads=threads)
            provider, _, fold, digest, _, _, _, raw = contract_development_inputs('cogpilot',
                data_root=pipeline['data_root'], registry_path=pipeline['registry_path'], full=True)
            if (fold.to_dict() != payload['fold'] or digest != config.data_manifest_sha256
                or candidate_data_sha256(None, provider, fold, 4) != payload['source_data_sha256']):
                raise ValueError('trial observations or roles differ from checkpoint')
            schedule = training_sample_schedule(None, provider, fold.train_sample_ids, 4)
            normalizer = TrainOnlyRobustNormalizer.from_manifest(payload['normalizer'])
            cursor = dict(payload['data_cursor'])
            state = payload
            if resume_state:
                state = torch.load(resume_state, map_location='cpu', weights_only=True)
                if (state['format'] != 'chronaris.performance_diagnostic.v1'
                    or state['parent_sha256'] != original_hash or state['graph'] != graph
                    or state['threads'] != threads):
                    raise ValueError('diagnostic resume lineage changed')
                cursor = state['data_cursor']
            if schedule.sha256 != cursor['sampling_order_sha256']:
                raise ValueError('trial sampling order differs from training')
            restored = payload | {k: state[k] for k in ('encoder_state_dict', 'head_state_dict', 'optimizer_state_dict', 'rng_state')}
            encoder, heads, parameters, optimizer = _restore(restored, graph)
            policy = AugmentationPolicy(**payload['augmentation_policy'])
            metadata = dict(parent_sha256=original_hash, source_code_sha256=candidate_source_code_sha256(),
                graph=graph, threads=threads, actual_batch=4, effective_batch=16,
                torch_version=str(torch.__version__), device=torch.cuda.get_device_name(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                training_role_shapes={role: {s: list(getattr(b, s+'_values').shape) for s in ('physiology', 'vehicle')} for role,b in raw.items()},
                resume_state=str(resume_state) if resume_state else None, measurements=[])
            (root/'metadata.json').write_text(json.dumps(metadata, indent=2))
            start_update = state['optimizer_updates']
            for update in range(start_update+1, start_update+updates+1):
                optimizer.zero_grad(set_to_none=True)
                outputs, losses, batches, timings = [], [], [], []
                def capture(_module, _args, output, sink=outputs):
                    sink.append((output.sequence_embedding.detach().clone(), output.modality_available_mask.detach().clone()))
                hook = encoder.register_forward_hook(capture)
                torch.cuda.reset_peak_memory_stats()
                for micro in range(4):
                    progress.update(phase='forward_backward', target_update=update, micro=micro)
                    ids = schedule.draw(cursor['samples_seen'], 4)
                    batches.append(ids)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output, mechanism, _, _ = pretext_micro_step(encoder=encoder, heads=heads, shift_head=None,
                        batch=None, batch_provider=provider, sample_ids=ids, normalizer=normalizer,
                        resolved=config, policy=policy, method_name='chronaris', epoch=cursor['micro_batches_seen']+1,
                        chronaris_lag_aware_weight=payload['chronaris_lag_aware_weight'],
                        chronaris_mechanism_enabled=payload['chronaris_mechanism_enabled'],
                        chronaris_explicit_shift_weight=payload['chronaris_explicit_shift_weight'],
                        chronaris_event_pair_weight=payload['chronaris_event_pair_weight'], optimizer_updates=update)
                    loss = output.total_loss+mechanism.additional_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite trial loss')
                    forward_done = time.perf_counter()
                    (loss/4).backward()
                    torch.cuda.synchronize()
                    timings.append(dict(forward_until_loss_sync_s=forward_done-started,
                        forward_backward_s=time.perf_counter()-started))
                    losses.append(loss.detach().cpu())
                    cursor['samples_seen'] += 4
                    cursor['micro_batches_seen'] += 1
                    print(json.dumps(dict(update=update, micro=micro, timing=timings[-1])), flush=True)
                hook.remove()
                gradients = {str(i): p.grad.detach().cpu().clone() for i,p in enumerate(parameters) if p.grad is not None}
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm, error_if_nonfinite=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.step()
                torch.cuda.synchronize()
                measurement = dict(update=update, micro_timings=timings, optimizer_s=time.perf_counter()-started,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(), gradient_norm=float(grad_norm))
                diagnostic = dict(format='chronaris.performance_diagnostic.v1', parent_sha256=original_hash,
                    graph=graph, threads=threads, optimizer_updates=update, data_cursor=dict(cursor),
                    encoder_state_dict={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()},
                    head_state_dict={k:v.detach().cpu().clone() for k,v in heads.state_dict().items()},
                    optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state())
                torch.save(diagnostic, root/f'state_{update}.pt')
                torch.save(dict(outputs=[tuple(t.cpu() for t in row) for row in outputs], losses=losses,
                    gradients=gradients, batches=batches), root/f'observations_{update}.pt')
                metadata['measurements'].append(measurement)
                progress.update(optimizer_updates=update, phase='diagnostic_state_saved')
                (root/'metadata.json').write_text(json.dumps(metadata, indent=2))
                del output, mechanism, loss, outputs, diagnostic, gradients
            assert sha256_file(checkpoint) == original_hash
            del encoder, heads, optimizer, parameters
            gc.collect()
            torch.cuda.empty_cache()
    return metadata
