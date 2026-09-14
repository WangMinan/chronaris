"""Explicit development recipes; model changes are separate from execution choices."""
from dataclasses import asdict, replace

from chronaris.evaluation.application_tasks.application_finetuning import EndToEndFineTuningConfig
from chronaris.modeling.training import CandidateScreenConfig, EncoderCandidateConfig
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options

RECIPES = ('stage4_reference', 'thesis_reference', 'cosine_temperature',
           'pretraining_lr', 'finetuning_lr', 'single_stream_fidelity')


def training_recipe(name, *, method, full, seed, digest, train_count, graph=False,
                    calibration=None, micro_batch=4):
    if name not in RECIPES or seed not in (17, 29, 43):
        raise ValueError('unknown stage 4.5 recipe or seed')
    if method != 'chronaris' and name not in ('stage4_reference', 'pretraining_lr', 'finetuning_lr'):
        raise ValueError('structural recipe requires Chronaris')
    if graph and method != 'chronaris':
        raise ValueError('graph recurrence requires Chronaris')
    if micro_batch not in (4, 8, 16) or (not full and micro_batch != 4):
        raise ValueError('unsupported execution microbatch')
    effective = 16 if full else 4
    pre = CandidateScreenConfig(max_updates=max(300, (train_count+15)//16) if full else 2,
        batch_size=micro_batch, effective_batch_size=effective, device='cuda', seed=seed,
        validation_interval=50 if full else 2, early_stopping=False,
        retained_updates=(50, 200) if full else (), data_manifest_sha256=digest,
        cuda_graph_recurrence=graph)
    guided = EndToEndFineTuningConfig(max_updates=max(200, (train_count+15)//16) if full else 2,
        head_warmup_updates=50 if full else 2, batch_size=micro_batch, effective_batch_size=effective,
        device='cuda', seed=seed, validation_interval=50 if full else 2, early_stopping=False,
        record_gradient_groups=True, checkpoint_interval=25 if full else 1,
        retained_updates=(50, 200) if full else (), data_manifest_sha256=digest)
    candidate = EncoderCandidateConfig(candidate_id='C', hidden_dim=32,
        learning_rate=3e-4 if name == 'pretraining_lr' else 1e-3)
    mechanisms = name == 'thesis_reference'
    if mechanisms:
        if calibration is None:
            raise ValueError('mechanism recipe requires explicit train-only calibration, including empty relations')
        pre = replace(pre, semantic_event_enabled=True, learnable_semantic_queries=True,
                      physics_calibration=calibration, physics_weight=.05)
    if name in ('cosine_temperature', 'single_stream_fidelity'):
        pre = replace(pre, **candidate_options(method, name)['training'])
    if name == 'finetuning_lr':
        guided = replace(guided, learning_rate=3e-5)
    arguments = dict(chronaris_fusion_kind='safe_lag' if method == 'chronaris' else 'multiscale',
        chronaris_mechanism_enabled=mechanisms, chronaris_explicit_shift_enabled=mechanisms,
        chronaris_explicit_shift_weight=.1 if mechanisms else 0., chronaris_event_pair_weight=0.)
    record = dict(name=name, candidate=asdict(candidate), pretraining=asdict(pre), finetuning=asdict(guided),
        mechanism_arguments=arguments, execution=dict(cuda_graph_recurrence=graph, micro_batch=micro_batch,
        effective_batch=effective, dtype='float32', cpu_threads=1),
        physics_relations='not_applicable_no_calibrated_native_relations',
        observation_anchor_enabled=mechanisms,
        comparison='single change versus stage4_reference; thesis_reference is a declared mechanism bundle',
        confirmation_opened=False)
    return candidate, pre, guided, arguments, record
