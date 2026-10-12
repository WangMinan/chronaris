"""Check actual trained mechanisms before counting a subset unit as complete."""
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha


def validate_checkpoints(result, unit):
    import torch
    native = unit['domain'] != 'simulation'
    for route in unit['routes']:
        if native:
            row = next(r for r in result['results'] if r['route']==route)
            record = read(Path(row['consumers']['manifest_path']).parents[1]/'common_contract.json')
            checkpoint = record['declaration']['checkpoint_path']
            if result['files'].get(checkpoint) != record['declaration']['checkpoint_sha256']:
                raise ValueError('native checkpoint missing from subset evidence')
        else:
            checkpoint = result['training'][route+'_training']['best_checkpoint_path']
            if result['files'].get(checkpoint) != sha(checkpoint):
                raise ValueError('simulation checkpoint missing from subset evidence')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['training_status']!='completed' or saved['seed']!=17 or saved['config']['device']!='cuda':
            raise ValueError('subset requires completed seed-17 CUDA checkpoints')
        if native:
            expected = unit['recipe']['pretraining' if route=='self_supervised' else 'finetuning']
            if json.loads(json.dumps(saved['config'])) != json.loads(json.dumps(expected)):
                raise ValueError('trained checkpoint differs from the frozen recipe')
        if route == 'self_supervised':
            semantic = unit.get('recipe_name')=='thesis_reference' if native else unit['options'].get('semantic_event_enabled', True)
            if (saved['config']['semantic_event_enabled'] != semantic or saved['config']['learnable_semantic_queries'] != semantic
                or not saved['chronaris_mechanism_enabled'] or not saved['chronaris_explicit_shift_enabled']):
                raise ValueError('checkpoint did not train the declared thesis mechanisms')
            if not native and (saved['chronaris_variant'] != unit['options'].get('variant','full')
                or saved['config']['physics_weight'] != unit['options'].get('physics_weight',.05)):
                raise ValueError('simulation checkpoint changed the declared ablation')
        else:
            if result['files'].get(saved['source_checkpoint_path']) != saved['source_checkpoint_sha256']:
                raise ValueError('task-guided initialization is outside the frozen subset unit')
