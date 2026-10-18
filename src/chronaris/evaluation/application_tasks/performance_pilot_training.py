"""One registered task/fold/seed panel; no architecture or task search."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows
from chronaris.evaluation.application_tasks.performance_pilot_data import future_query_variants
from chronaris.modeling.fusion_encoders.deep_baselines import (
    CausalContiFormerFusionEncoder, CausalMulTFusionEncoder, DeepBaselineEncoderConfig,
)
from chronaris.modeling.fusion_encoders.same_backbone import SameBackboneFusion
from chronaris.modeling.fusion_encoders.single_stream import ContinuousTimeSingleStreamEncoder, SingleStreamEncoderConfig
from chronaris.modeling.training.rng import capture_rng_state, isolated_training_rng


METHODS = ('vehicle_only', 'physiology_only', 'contiformer', 'mult', 'same_backbone')


def build_pilot_model(method, device='cpu'):
    if method in ('vehicle_only', 'physiology_only'):
        model = ContinuousTimeSingleStreamEncoder(SingleStreamEncoderConfig(
            'vehicle' if method == 'vehicle_only' else 'physiology', 8 if method == 'vehicle_only' else 4))
    elif method in ('mult', 'contiformer'):
        cls = CausalMulTFusionEncoder if method == 'mult' else CausalContiFormerFusionEncoder
        model = cls(DeepBaselineEncoderConfig(method, 4, 8))
    elif method == 'same_backbone':
        model = SameBackboneFusion(4, 8)
        # Identical branch initialization separates joint fitting from a lucky start.
        with isolated_training_rng(17):
            physiology = build_pilot_model('physiology_only')
        with isolated_training_rng(17):
            vehicle = build_pilot_model('vehicle_only')
        model.physiology.load_state_dict(physiology.state_dict())
        model.vehicle.load_state_dict(vehicle.state_dict())
    else:
        raise ValueError(method)
    return model.to(device)


def encode(model, method, p, v):
    if method in ('vehicle_only', 'physiology_only'):
        output = model.forward_queried(v if method == 'vehicle_only' else p)
    else:
        output = model.forward_queried(p, v)
    sequence = output.sequence_embedding
    valid = output.valid_mask if hasattr(output, 'valid_mask') else output.modality_available_mask
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid).masked_fill(~valid, -1).max(1).values
    if (positions < 0).any() or not torch.isfinite(sequence).all():
        raise ValueError('invalid or empty encoder output')
    return sequence, sequence[torch.arange(len(sequence), device=sequence.device), positions]


@torch.inference_mode()
def export_features(model, head, method, data, indices):
    model.eval(); head.eval()
    features, predictions = [], []
    for offset in range(0, len(indices), 32):
        p, v = data.queries(indices[offset:offset+32])
        _, pooled = encode(model, method, p, v)
        features.append(pooled.cpu().numpy())
        predictions.append(head(pooled).flatten().cpu().numpy())
    return np.concatenate(features), np.concatenate(predictions)


@torch.inference_mode()
def audit_history(model, method, data):
    model.eval()
    p, v = data.queries(data.indices('select')[:4])
    expected, _ = encode(model, method, p, v)
    past = p.timestamps_s <= 15.
    results = {}
    for name, cp, cv in zip(('changed', 'deleted'), future_query_variants(p), future_query_variants(v)):
        actual, _ = encode(model, method, cp, cv)
        torch.testing.assert_close(actual[past], expected[past], atol=1e-5, rtol=1e-5)
        results[name] = float((actual[past]-expected[past]).abs().max())
    return results


def atomic_checkpoint(path, payload):
    partial = path.with_suffix('.partial')
    torch.save(payload, partial)
    os.replace(partial, path)


def train_pilot_method(method, data, assets, report, *, deadline, binding):
    root = Path(assets)/method
    root.mkdir(exist_ok=False)
    start = time.monotonic()
    train, select = data.indices('fit'), data.indices('select')
    with isolated_training_rng(17):
        model = build_pilot_model(method, data.values.device)
        head = nn.Linear(64, 1).to(data.values.device)
        optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=.001, weight_decay=.0001)
        history = []; best = float('inf'); best_epoch = 0; updates = 0
        preflight = audit_history(model, method, data)
        status = 'budget_incomplete'
        for epoch in range(1, 101):
            if time.monotonic() >= min(deadline, start+3600):
                break
            model.train(); head.train(); losses = []
            order = np.random.default_rng(17+epoch).permutation(train)
            for offset in range(0, len(order), 32):
                indices = order[offset:offset+32]
                optimizer.zero_grad(set_to_none=True)
                p, v = data.queries(indices)
                _, pooled = encode(model, method, p, v)
                error = (head(pooled).flatten() - data.targets[indices])**2
                loss = (error*data.weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError(f'nonfinite loss: {method} epoch {epoch}')
                loss.backward()
                nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                losses.append((float(loss.detach()), len(indices)))
            _, prediction = export_features(model, head, method, data, select)
            expected = data.targets[select].cpu().numpy()
            validation = float(np.sqrt(np.mean((expected-prediction)**2)))
            improved = validation < best - 1e-6
            if improved:
                best, best_epoch = validation, epoch
            history.append(dict(method=method, epoch=epoch, updates=updates,
                                training_weighted_mse=sum(a*n for a, n in losses)/len(train),
                                selection_normalized_rmse=validation, best_epoch=best_epoch,
                                elapsed_s=time.monotonic()-start))
            payload = dict(format='chronaris.performance_pilot.v1', method=method, seed=17, epoch=epoch,
                           updates=updates, best_epoch=best_epoch, selection_normalized_rmse=validation,
                           encoder=model.state_dict(), head=head.state_dict(), optimizer=optimizer.state_dict(),
                           rng=capture_rng_state(), normalizer=data.normalizer, binding=binding)
            atomic_checkpoint(root/'last.pt', payload)
            if improved:
                atomic_checkpoint(root/'best.pt', payload)
            save_rows(Path(report)/f'{method}_history.csv', history)
            print(json.dumps(history[-1]), flush=True)
            if epoch >= 20 and epoch-best_epoch >= 15:
                status = 'early_stopped'; break
            if epoch == 100:
                status = 'max_epochs'
        if not history:
            raise TimeoutError('budget expired before first full epoch')
        best_state = torch.load(root/'best.pt', map_location=data.values.device, weights_only=True)
        model.load_state_dict(best_state['encoder']); head.load_state_dict(best_state['head'])
        features = {}
        for role in ('fit', 'select'):
            features[role] = export_features(model, head, method, data, data.indices(role))
        restored = build_pilot_model(method, data.values.device)
        restored.load_state_dict(best_state['encoder'])
        restored_head = nn.Linear(64, 1).to(data.values.device)
        restored_head.load_state_dict(best_state['head'])
        reload_features = export_features(restored, restored_head, method, data, select)
        for expected, actual in zip(features['select'], reload_features):
            np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)
        history_audit = audit_history(restored, method, data)
        receipt = dict(method=method, status=status, complete=status in ('early_stopped', 'max_epochs'),
                       epochs=history[-1]['epoch'], updates=updates, best_epoch=best_epoch,
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       elapsed_s=time.monotonic()-start, preflight_history=preflight,
                       trained_history=history_audit, checkpoint_sha256=file_digest(root/'best.pt'),
                       checkpoint_path=str((root/'best.pt').resolve()), reload_passed=True)
        save_json(Path(report)/f'{method}_receipt.json', receipt)
        del model, head, optimizer, restored, restored_head
        torch.cuda.empty_cache()
        return features, receipt
