"""Two-update, resumable adapter diagnostics using existing task losses and RNG state."""
from pathlib import Path
import time

import torch
from torch import nn

from chronaris.evaluation.application_tasks.application_task_heads import (
    application_task_losses, fit_application_task_parameters, select_application_targets)
from chronaris.modeling.training.rng import capture_rng_state, restore_rng_state, canonical_training_state_sha256
from chronaris.modeling.training.candidate_checkpoint import atomic_save_candidate


def encoder_checkpoint_state(encoder):
    return encoder.checkpoint_state() if hasattr(encoder, 'checkpoint_state') else encoder.state_dict()


def load_encoder_checkpoint(encoder, state):
    if hasattr(encoder, 'load_checkpoint_state'):
        encoder.load_checkpoint_state(state)
    else:
        encoder.load_state_dict(state)


def task_covered_positions(targets, definitions, train_ids, update, batch_size):
    """Choose by training-role task availability, never by label values or outcomes."""
    selected = select_application_targets(targets, train_ids, 'cpu')
    masks = {t.name: selected['valid_masks'][t.name].reshape(len(train_ids), -1).any(1) for t in definitions}
    positions = []
    for mask in masks.values():
        if positions and mask[positions].any():
            continue
        candidates = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not candidates:
            raise ValueError('declared training task has no available training labels')
        positions.append(candidates[(update*batch_size) % len(candidates)])
    if len(positions) > batch_size:
        raise ValueError('batch budget cannot cover declared training tasks')
    for offset in range(len(train_ids)):
        index = (update*batch_size+offset) % len(train_ids)
        if len(positions) == min(batch_size, len(train_ids)):
            break
        if index not in positions:
            positions.append(index)
    return positions


def short_fit(encoder, *, values, mask, prompts, targets, definitions, train_ids, metadata, root,
              stop_after=2, progress=None):
    """Engineering budget only. Save optimizer and RNG after each real CUDA update."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary_labels = metadata['target_supervision'] == 'summary_labels'
    heads = nn.ModuleDict({t.name: nn.Linear(encoder.feature_dim, t.output_dim) for t in definitions}).cuda() if summary_labels else nn.ModuleDict()
    params = fit_application_task_parameters(targets, definitions, train_ids) if summary_labels else None
    if hasattr(encoder, 'configure_training'):
        encoder.configure_training()
    else:
        encoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(heads.parameters()), lr=1e-4 if summary_labels else 1e-6)
    checkpoint = root/'training.pt'
    history, update = [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['metadata'] != metadata:
            raise ValueError('adapter training source/data changed; use a new root')
        load_encoder_checkpoint(encoder, saved['encoder'])
        heads.load_state_dict(saved['heads'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng_state(saved['rng'])
        history, update = saved['history'], saved['update']
    if progress is not None:
        progress['checkpoint_updates'] = update
    encoder.train()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    while update < stop_after:
        batch_size = 4 if summary_labels and metadata['method_name']=='timecma' else 1
        positions = (task_covered_positions(targets, definitions, train_ids, update, batch_size) if summary_labels else
            [(update*batch_size+j) % len(train_ids) for j in range(min(batch_size, len(train_ids)))])
        ids = tuple(train_ids[j] for j in positions)
        optimizer.zero_grad(set_to_none=True)
        if summary_labels:
            hidden = encoder(values[positions], mask[positions], None if prompts is None else prompts[positions])
            output = {'task_predictions': {name: head(hidden) for name, head in heads.items()}}
            task_losses = application_task_losses(output, select_application_targets(targets, ids, 'cuda'), definitions, params)
            loss = task_losses['total']
        else:
            loss = encoder.history_loss(values[positions], mask[positions])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite recent-model training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(heads.parameters()), 1., error_if_nonfinite=True)
        optimizer.step()
        update += 1
        if progress is not None:
            progress['optimizer_updates'] += 1
            progress.update(checkpoint_updates=update, last_loss=float(loss.detach()))
        history.append(dict(update=update, loss=float(loss.detach()), gradient_norm=float(norm), sample_ids=list(ids),
            task_counts=task_losses['counts'] if summary_labels else {},
            parameters_with_gradient=sum(p.numel() for p in encoder.parameters() if p.grad is not None)))
        saved = dict(metadata=metadata, encoder=encoder_checkpoint_state(encoder), heads=heads.state_dict(),
            optimizer=optimizer.state_dict(), rng=capture_rng_state(), update=update, history=history,
            task_parameters=params)
        atomic_save_candidate(checkpoint, saved)
    torch.cuda.synchronize()
    return dict(updates=update, history=history, seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        state_sha256=canonical_training_state_sha256(encoder_checkpoint_state(encoder), heads.state_dict(), optimizer.state_dict()))
