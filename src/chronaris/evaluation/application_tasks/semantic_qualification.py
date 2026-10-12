"""Bounded, source-recorded semantic execution qualification on development data."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import torch

from chronaris.evaluation.application_tasks.checkpoint_performance import compare_values
from chronaris.evaluation.application_tasks.stage45_performance import performance_entry
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.models.fusion.semantic_event import EventTokenExtractor


@contextmanager
def semantic_execution(legacy):
    original = EventTokenExtractor.extract
    if legacy:
        project = Path(__file__).parents[4]
        source = subprocess.check_output(['git', 'show',
            '4c723e460c919545e1aa5997b9db6c26c58e4b6e:src/chronaris/models/fusion/semantic_event.py'],
            cwd=project, text=True)
        module = types.ModuleType('chronaris_qualification_legacy_events')
        sys.modules[module.__name__] = module
        exec(compile(source, 'frozen_event_extractor.py', 'exec'), module.__dict__)
        EventTokenExtractor.extract = module.EventTokenExtractor.extract
    try:
        yield
    finally:
        EventTokenExtractor.extract = original


def short_entry(*, root, domain, mode):
    root = Path(root)
    original = json.loads(Path('/mnt/e/chronaris-v4-results/2026-10-11-chapter5-reload/pipeline_config.json').read_text())
    side = 'legacy' if mode == 'legacy' else 'candidate'
    config = dict(root=str(root/side), data_root=original['data_root'], registry_path=original['registry_path'])
    started = time.perf_counter()
    with semantic_execution(mode == 'legacy'):
        if mode in ('legacy', 'candidate'):
            result = performance_entry(config, domain, 'thesis_reference', 'graph')
        else:
            with development_gpu_lock() as acquired:
                if not acquired:
                    raise RuntimeError('GPU belongs to another task')
                result = performance_entry(config, domain, 'thesis_reference', mode)
    return write_result(root/f'{domain}_{mode}.json', dict(status='completed', mode=mode,
        domain=domain, wall_seconds=time.perf_counter()-started, result=result, confirmation_opened=False))


def compare_entries(root, domain):
    root = Path(root)
    suffix = Path('performance')/domain/'thesis_reference'/'graph'/'trace'
    left, right = root/'legacy'/suffix, root/'candidate'/suffix
    names = sorted(p.name for p in left.glob('*.pt'))
    if not names or names != sorted(p.name for p in right.glob('*.pt')):
        raise ValueError('short-entry traces are incomplete')
    checks = {}
    for name in names:
        a, b = [torch.load(p/name, map_location='cpu', weights_only=True) for p in (left, right)]
        checks[name] = compare_values(a, b, atol=0., rtol=0.)
    resume = json.loads((root/'candidate/performance'/domain/'thesis_reference/resume_check.json').read_text())
    return write_result(root/f'{domain}_comparison.json', dict(status='completed',
        passed=all(c['close'] for c in checks.values()) and resume['passed'], checks=checks,
        independent_resume=resume, confirmation_opened=False))


def mature_probe(*, root, domain, mode, resume=None):
    """Use trained development weights and their real cursor with active mechanism losses."""
    from chronaris.evaluation.application_tasks.checkpoint_performance import _restore
    from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
    from chronaris.modeling.training.candidate_config import CandidateScreenConfig
    from chronaris.modeling.training.candidate_step import pretext_micro_step
    from chronaris.modeling.training.rng import capture_rng_state, isolated_training_rng
    from chronaris.modeling.training.sample_schedule import training_sample_schedule
    from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    checkpoints = {
        'clare': '/mnt/e/chronaris-v4-results/2026-09-14-stage45-recovery/units/screen/0/clare/chronaris/self_supervised/chronaris/C/best.pt',
        'cogpilot': '/mnt/e/chronaris-v4-results/2026-09-18-stage45-accelerated/units/screen/9/cogpilot/chronaris/self_supervised/chronaris/C/best.pt',
    }
    checkpoint = checkpoints[domain]
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    digest_before = sha256_file(checkpoint)
    if not (payload['chronaris_mechanism_enabled'] and payload['chronaris_explicit_shift_enabled']
            and payload['config']['semantic_event_enabled'] and payload['optimizer_updates'] >= 200):
        raise ValueError('mature probe requires a trained complete-mechanism checkpoint')
    source = json.loads(Path('/mnt/e/chronaris-v4-results/2026-10-11-chapter5-reload/pipeline_config.json').read_text())
    provider, _, fold, _, digest, _, _, _ = load_development_inputs(
        domain, source['data_root'], source['registry_path'])
    if fold.to_dict() != payload['fold'] or digest != payload['config']['data_manifest_sha256']:
        raise ValueError('mature probe data differs from checkpoint')
    torch.set_num_threads(1)
    schedule = training_sample_schedule(None, provider, fold.train_sample_ids, 4)
    if schedule.sha256 != payload['data_cursor']['sampling_order_sha256']:
        raise ValueError('mature probe sampling order differs')
    parent_update = payload['optimizer_updates']
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['parent_sha256'] != digest_before:
            raise ValueError('resume parent differs')
        payload = payload | saved['state']
    output = Path(root)/'mature'/domain/mode
    output.mkdir(parents=True, exist_ok=False)
    cursor = dict(payload['data_cursor'])
    normalizer = TrainOnlyRobustNormalizer.from_manifest(payload['normalizer'])
    config = CandidateScreenConfig(**payload['config'])
    policy = AugmentationPolicy(**payload['augmentation_policy'])
    micros = config.effective_batch_size // config.batch_size
    if config.batch_size != 4 or micros != 4:
        raise ValueError('mature probe expects native full-budget batch 4/16')
    measurements = []
    with development_gpu_lock() as acquired, semantic_execution(mode == 'legacy'), isolated_training_rng(payload['seed']):
        if not acquired:
            raise RuntimeError('GPU belongs to another task')
        encoder, heads, shift, parameters, optimizer = _restore(payload, True)
        for index in range(1 if resume else 2):
            update = payload['optimizer_updates'] + index + 1
            optimizer.zero_grad(set_to_none=True)
            outputs, losses, ids, terms, timings = [], [], [], [], []
            def capture(_module, _args, value):
                outputs.append((value.sequence_embedding.detach().cpu(), value.modality_available_mask.cpu()))
            hook = encoder.register_forward_hook(capture)
            for _ in range(micros):
                sample_ids = schedule.draw(cursor['samples_seen'], 4)
                data_started = time.perf_counter()
                raw = provider(sample_ids)
                data_seconds = time.perf_counter() - data_started
                torch.cuda.synchronize()
                started = time.perf_counter()
                public, mechanism, augmented, _ = pretext_micro_step(
                    encoder=encoder, heads=heads, shift_head=shift, batch=raw, batch_provider=None,
                    sample_ids=raw.sample_ids, normalizer=normalizer, resolved=config, policy=policy,
                    method_name='chronaris', epoch=cursor['micro_batches_seen']+1, optimizer_updates=update,
                    **{k:payload[k] for k in ('chronaris_lag_aware_weight','chronaris_mechanism_enabled',
                        'chronaris_explicit_shift_weight','chronaris_event_pair_weight')})
                loss = public.total_loss + mechanism.additional_loss
                assert torch.isfinite(loss)
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - started
                started = time.perf_counter()
                (loss/micros).backward()
                torch.cuda.synchronize()
                timings.append(dict(data_seconds=data_seconds, forward_seconds=forward_seconds,
                                    backward_seconds=time.perf_counter()-started))
                losses.append(loss.detach().cpu())
                terms.extend(dict(row) for row in mechanism.rows)
                ids.append((sample_ids, augmented.augmentation_ids))
                cursor['samples_seen'] += 4
                cursor['micro_batches_seen'] += 1
            hook.remove()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm, error_if_nonfinite=True)
            gradients = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters]
            torch.cuda.synchronize(); started = time.perf_counter()
            optimizer.step(); torch.cuda.synchronize()
            measurements.append(dict(update=update, micro_timings=timings, optimizer_seconds=time.perf_counter()-started))
            state = dict(encoder_state_dict=encoder.state_dict(), head_state_dict=heads.state_dict(),
                explicit_time_shift_head_state_dict=shift.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                rng_state=capture_rng_state(), data_cursor=dict(cursor), optimizer_updates=update)
            torch.save(dict(state=state, outputs=outputs, losses=losses, gradients=gradients, ids=ids,
                            parent_sha256=digest_before), output/f'update_{update}.pt')
            write_result(output/'mechanism_terms.json', dict(rows=terms))
            print(json.dumps(measurements[-1]), flush=True)
        encoder.eval(); heads.eval(); shift.eval()
        evaluation = []
        with torch.no_grad():
            for offset in range(0, min(16, len(fold.validation_sample_ids)), 4):
                raw = provider(fold.validation_sample_ids[offset:offset+4])
                torch.cuda.synchronize(); started = time.perf_counter()
                pretext_micro_step(encoder=encoder, heads=heads, shift_head=shift, batch=raw, batch_provider=None,
                    sample_ids=raw.sample_ids, normalizer=normalizer, resolved=config, policy=policy,
                    method_name='chronaris', epoch=1, optimizer_updates=parent_update+2,
                    **{k:payload[k] for k in ('chronaris_lag_aware_weight','chronaris_mechanism_enabled',
                        'chronaris_explicit_shift_weight','chronaris_event_pair_weight')})
                torch.cuda.synchronize()
                evaluation.append(time.perf_counter()-started)
    assert sha256_file(checkpoint) == digest_before
    return write_result(output/'summary.json', dict(status='completed', parent_checkpoint=checkpoint,
        parent_sha256=digest_before, domain=domain, actual_batch=4, effective_batch=16,
        measurements=measurements, validation_batch_seconds=evaluation, confirmation_opened=False))


def run_bounded(command, *, deadline_path, log_path, max_seconds=None):
    """Apply the shared wall-clock deadline to this process group, including children."""
    deadline = json.loads(Path(deadline_path).read_text())['deadline_unix_s']
    remaining = deadline - time.time()
    timeout = min(remaining, max_seconds) if max_seconds is not None else remaining
    if timeout <= 0:
        raise TimeoutError('qualification deadline reached')
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    record = dict(command=command, pid=process.pid, started_at_unix_s=started,
        seconds=time.time()-started, deadline_unix_s=deadline, exit_code=code, timed_out=timed_out)
    write_result(path.with_suffix('.receipt.json'), record)
    if code or timed_out:
        raise RuntimeError(f'qualification command stopped; see {path}')
    return record
