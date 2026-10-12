"""Preserve a training checkpoint and record an explicitly requested pause."""
import json
import os
from pathlib import Path
import shutil
import signal
import time

import torch

from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def paused_state(before, stopped, receipt_path):
    """Keep the raw termination record separately; never hide an actual failure."""
    if stopped['children'] or stopped['completed'] != before['completed']:
        raise ValueError('pause boundary changed; inspect the preserved raw state')
    old = before.get('failures', [])
    new = stopped.get('failures', [])[len(old):]
    if (stopped.get('failures', [])[:len(old)] != old or len(new) != 1
        or new[0]['stage'] != before['current_stage']
        or not new[0]['error'].rstrip().endswith('KeyboardInterrupt: pipeline termination requested')):
        raise ValueError('termination was not solely the requested pause')
    result = dict(stopped, status='paused_by_user', failures=old,
                  pause_receipt=str(receipt_path), updated_at_unix_s=time.time())
    result.pop('error', None)
    return result


def pause_pipeline(*, root, checkpoint, output_root, wait_seconds=600):
    root, checkpoint, output = Path(root).resolve(), Path(checkpoint).resolve(), Path(output_root).resolve()
    if root not in checkpoint.parents or root in output.parents or output == root:
        raise ValueError('checkpoint must belong to the run and receipt must use a separate root')
    if not 0 < wait_seconds <= 600:
        raise ValueError('pause wait must be bounded to at most ten minutes')
    state_path = root/'pipeline_state.json'
    before = json.loads(state_path.read_text())
    if before['status'] != 'running' or len(before['children']) != 1:
        raise ValueError('pause requires one active worker')
    controller = before['pid']
    worker = next(iter(before['children'].values()))
    for pid in (controller, worker):
        if str(root).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes():
            raise ValueError('recorded process belongs to another run')
    for receipt in before['completed'].values():
        if sha256_file(receipt['path']) != receipt['sha256']:
            raise ValueError('completed receipt changed before pause')
    output.mkdir(parents=True, exist_ok=False)
    write_result(output/'state_before.json', before)
    initial_mtime = checkpoint.stat().st_mtime_ns
    deadline = time.monotonic() + wait_seconds
    while checkpoint.stat().st_mtime_ns == initial_mtime:
        current = json.loads(state_path.read_text())
        if current['current_stage'] != before['current_stage'] or current['status'] != 'running':
            raise ValueError('unit boundary changed before pause; inspect without sending signals')
        if time.monotonic() >= deadline:
            raise TimeoutError('no new checkpoint within the bounded wait; original run remains active')
        time.sleep(.5)
    preserved = output/'preserved_last.pt'
    shutil.copyfile(checkpoint, preserved)
    payload = torch.load(preserved, map_location='cpu', weights_only=True)
    if not {'optimizer_state_dict', 'rng_state', 'data_cursor', 'optimizer_updates'} <= payload.keys():
        raise ValueError('new checkpoint is not resumable')
    saved_updates = payload['optimizer_updates']
    os.kill(controller, signal.SIGTERM)
    stop_deadline = time.monotonic() + 45
    for pid in (controller, worker):
        while Path(f'/proc/{pid}/cmdline').exists() and Path(f'/proc/{pid}/cmdline').read_bytes():
            if time.monotonic() >= stop_deadline:
                raise TimeoutError('process did not exit; preserved checkpoint is available')
            time.sleep(.2)
    stopped = json.loads(state_path.read_text())
    write_result(output/'raw_state_after_termination.json', stopped)
    receipt_path = output/'pause_receipt.json'
    updated = paused_state(before, stopped, receipt_path)
    final = torch.load(checkpoint, map_location='cpu', weights_only=True)
    receipt = dict(status='paused_by_user', requested_pause=True, stopped=True,
        parent_root=str(root), controller_pid=controller, children=before['children'],
        current_stage=before['current_stage'], completed_units=sum(k.startswith('chapter5_core__') for k in before['completed']),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_optimizer_updates=final['optimizer_updates'], preserved_checkpoint=str(preserved),
        preserved_sha256=sha256_file(preserved), preserved_optimizer_updates=saved_updates,
        interrupted_updates_after_checkpoint='not_observed; waited for a fresh checkpoint before signalling',
        completed_receipts=before['completed'], raw_state=str(output/'raw_state_after_termination.json'),
        stopped_at_unix_s=time.time(), confirmation_opened=False)
    write_result(receipt_path, receipt)
    write_result(state_path, updated)
    return receipt
