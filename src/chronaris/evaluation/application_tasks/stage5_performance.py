"""Bounded production-shape diagnostics; never resumes or replaces a formal unit."""
import json
from pathlib import Path
import time

import torch

from chronaris.evaluation.application_tasks.checkpoint_performance import _restore
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, v4_workflow_source_sha256
from chronaris.modeling.training.candidate_config import CandidateScreenConfig
from chronaris.modeling.training.candidate_step import pretext_micro_step
from chronaris.modeling.training.rng import capture_rng_state, isolated_training_rng
from chronaris.modeling.training.sample_schedule import BalancedSampleSchedule
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def production_probe(*, parent, output, resume=None):
    """Two true scheduled updates from a frozen update-200 state, with full state trace."""
    parent, output = Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = parent/'training/135/cogpilot/chronaris/self_supervised/chronaris/C/update_000200.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = json.loads((parent/'pipeline_config.json').read_text())
    torch.set_num_threads(1)
    provider, _, fold, hierarchy, digest, _, _, _ = load_development_inputs('cogpilot',
        config['confirmation_data_root'], config['registry_path'], subject_role='confirmation')
    if fold.to_dict() != payload['fold'] or digest != payload['config']['data_manifest_sha256']:
        raise ValueError('performance probe must retain the frozen formal fold')
    schedule = BalancedSampleSchedule(fold.train_sample_ids, {s: (hierarchy[s][0],) for s in fold.train_sample_ids})
    if schedule.sha256 != payload['data_cursor']['sampling_order_sha256']:
        raise ValueError('performance sampling order changed')
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['parent_sha256'] != sha256_file(checkpoint):
            raise ValueError('diagnostic resume parent changed')
        payload = payload | saved['state']
    cursor = dict(payload['data_cursor'])
    # Load ahead only; normalization and random augmentation remain inside timing.
    batches = [provider(schedule.draw(cursor['samples_seen'] + i*4, 4)) for i in range(8 if not resume else 4)]
    times = []
    with isolated_training_rng(17):
        encoder, heads, shift, parameters, optimizer = _restore(payload, True)
        normalizer = TrainOnlyRobustNormalizer.from_manifest(payload['normalizer'])
        resolved, policy = CandidateScreenConfig(**payload['config']), AugmentationPolicy(**payload['augmentation_policy'])
        for index in range(1 if resume else 2):
            outputs, losses, ids = [], [], []
            def capture(_module, _args, value):
                outputs.append(value.sequence_embedding.detach().cpu())
            handle = encoder.register_forward_hook(capture)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); started = time.perf_counter()
            for micro in range(4):
                raw = batches[index*4+micro]
                public, mechanism, augmented, _ = pretext_micro_step(
                    encoder=encoder, heads=heads, shift_head=shift, batch=raw, batch_provider=None,
                    sample_ids=raw.sample_ids, normalizer=normalizer, resolved=resolved, policy=policy,
                    method_name='chronaris', epoch=cursor['micro_batches_seen']+1,
                    optimizer_updates=payload['optimizer_updates']+index,
                    **{k:payload[k] for k in ('chronaris_lag_aware_weight', 'chronaris_mechanism_enabled',
                        'chronaris_explicit_shift_weight', 'chronaris_event_pair_weight')})
                loss = public.total_loss+mechanism.additional_loss
                (loss/4).backward()
                losses.append(loss.detach().cpu()); ids.append((raw.sample_ids, augmented.augmentation_ids))
                cursor['samples_seen'] += 4; cursor['micro_batches_seen'] += 1
            torch.nn.utils.clip_grad_norm_(parameters, resolved.gradient_clip_norm, error_if_nonfinite=True)
            gradients = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters]
            optimizer.step(); torch.cuda.synchronize(); times.append(time.perf_counter()-started)
            handle.remove()
            update = payload['optimizer_updates']+index+1
            state = dict(encoder_state_dict=encoder.state_dict(), head_state_dict=heads.state_dict(),
                explicit_time_shift_head_state_dict=shift.state_dict() if shift is not None else None,
                optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
                data_cursor=dict(cursor), optimizer_updates=update)
            torch.save(dict(state=state, outputs=outputs, losses=losses, gradients=gradients, ids=ids,
                parent_sha256=sha256_file(checkpoint)), output/f'update_{update}.pt')
    result = dict(status='completed', source_code_sha256=v4_workflow_source_sha256(),
        parent_checkpoint=str(checkpoint), parent_sha256=sha256_file(checkpoint), seconds=times,
        actual_batch=4, effective_batch=16, dtype='float32', confirmation_opened=False)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
