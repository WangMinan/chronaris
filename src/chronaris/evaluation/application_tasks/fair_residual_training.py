"""Bounded fitting and strict replay for the preregistered residual models."""
from pathlib import Path
import json
import time

import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows
from chronaris.evaluation.application_tasks.performance_pilot_training import atomic_checkpoint
from chronaris.evaluation.application_tasks.performance_pilot_data import future_query_variants
from chronaris.modeling.fusion_encoders.fair_residual import FairResidualModel
from chronaris.modeling.training.rng import isolated_training_rng, capture_rng_state


@torch.inference_mode()
def predict_fair(model, data, ids):
    model.eval()
    features, predictions = [], []
    for offset in range(0, len(ids), 32):
        positions = ids[offset:offset+32]
        sequence, correction = model(*data.queries(positions))
        features.append(sequence[:, -1].cpu().numpy())
        predictions.append((data.anchor[positions]+data.target_scale*correction).cpu().numpy())
    return np.concatenate(features), np.concatenate(predictions)


@torch.inference_mode()
def check_history(model, data):
    model.eval()
    p, v = data.queries(data.indices('select')[:3])
    original, _ = model(p, v); past = p.timestamps_s <= 15
    deltas = {}
    for name, a, b in zip(('modified', 'deleted'), future_query_variants(p), future_query_variants(v)):
        actual, _ = model(a, b)
        torch.testing.assert_close(original[past], actual[past], atol=1e-5, rtol=1e-5)
        deltas[name] = float((original[past]-actual[past]).abs().max())
    return deltas


def load_fair_model(path, data):
    state = torch.load(path, map_location=data.device, weights_only=True)
    if state['normalizer'] != data.normalizer:
        raise ValueError('checkpoint normalization lineage mismatch')
    if abs(state['target_scale']-data.target_scale) > 1e-12:
        raise ValueError('checkpoint target scale mismatch')
    model = FairResidualModel(state['method']).to(data.device)
    model.load_state_dict(state['state'], strict=True)
    return model.eval()


def fit_fair_model(method, data, root, report, name, binding, deadline):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); fit, select = data.indices('fit'), data.indices('select')
    with isolated_training_rng(17):
        model = FairResidualModel(method).to(data.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        check_history(model, data)
        history = []; best = float('inf'); best_epoch = 0; updates = 0; status = 'budget_incomplete'
        for epoch in range(1, 101):
            if time.monotonic() >= min(deadline, start+3600):
                break
            model.train(); losses = []
            order = np.random.default_rng(17+epoch).permutation(fit)
            for offset in range(0, len(order), 32):
                ids = order[offset:offset+32]
                optimizer.zero_grad(set_to_none=True)
                _, correction = model(*data.queries(ids))
                loss = (((correction[:, -1]-data.targets[ids])**2)*data.weights[ids]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite residual training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                losses.append((float(loss.detach()), len(ids)))
            _, pred = predict_fair(model, data, select)
            rmse = float(np.sqrt(np.mean((pred[:, -1]-data.y[select])**2)))
            score = rmse/data.target_scale
            improved = score < best-1e-6
            if improved:
                best, best_epoch = score, epoch
            history.append(dict(model=name, epoch=epoch, updates=updates, training_mse=sum(v*n for v, n in losses)/len(fit),
                                selection_rmse=rmse, selection_scaled_rmse=score, best_epoch=best_epoch,
                                elapsed_s=time.monotonic()-start))
            payload = dict(format='chronaris.fair_residual.v1', method=method, name=name, seed=17,
                           epoch=epoch, best_epoch=best_epoch, updates=updates, target_scale=data.target_scale,
                           state=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng_state(),
                           normalizer=data.normalizer, fold=data.fold, binding=binding)
            atomic_checkpoint(root/'last.pt', payload)
            if improved:
                atomic_checkpoint(root/'best.pt', payload)
            save_rows(Path(report)/f'{name}_history.csv', history)
            print(json.dumps(history[-1]), flush=True)
            if epoch >= 20 and epoch-best_epoch >= 15:
                status = 'early_stopped'; break
            if epoch == 100:
                status = 'max_epochs'
        if not history:
            raise TimeoutError('no complete epoch within budget')
        state = torch.load(root/'best.pt', map_location=data.device, weights_only=True)
        model.load_state_dict(state['state'])
        restored = load_fair_model(root/'best.pt', data)
        ref = predict_fair(model, data, select)
        replay = predict_fair(restored, data, select)
        for a, b in zip(ref, replay):
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        receipt = dict(name=name, method=method, status=status, complete=status != 'budget_incomplete',
                       epochs=history[-1]['epoch'], best_epoch=best_epoch, updates=updates,
                       selection_rmse=float(np.sqrt(np.mean((ref[1][:, -1]-data.y[select])**2))),
                       checkpoint_path=str((root/'best.pt').resolve()), checkpoint_sha256=file_digest(root/'best.pt'),
                       parameter_count=sum(x.numel() for x in model.parameters()), elapsed_s=time.monotonic()-start,
                       history_check=check_history(restored, data), reload_max_delta=max(float(np.abs(a-b).max()) for a,b in zip(ref,replay)))
        save_json(Path(report)/f'{name}_receipt.json', receipt)
        del optimizer, model
        return restored, receipt
