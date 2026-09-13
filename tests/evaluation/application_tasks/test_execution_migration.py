from copy import deepcopy

import pytest
import torch

from chronaris.evaluation.application_tasks.execution_migration import STATE_KEYS, migrate_payload
from chronaris.modeling.training.candidate_checkpoint import candidate_protocol_hash, candidate_source_code_sha256
from chronaris.modeling.training.rng import canonical_training_state_sha256


def _payload():
    config = dict(cuda_graph_recurrence=False, ode_method='euler', max_ode_step_s=None,
                  batch_size=4, effective_batch_size=16, max_updates=300, seed=17, device='cuda')
    protocol = dict(source_data_sha256='data', source_code_sha256=candidate_source_code_sha256(),
        method_name='chronaris', config=config, candidate={}, augmentation_policy={}, fold={}, normalizer={},
        physiology_feature_names=['p'], vehicle_feature_names=['v'], vehicle_field_labels=[],
        data_access_mode='lazy_batch_provider', transfer_source=None, chronaris_fusion_kind='safe_lag',
        chronaris_variant='full', chronaris_lag_aware_weight=0., chronaris_mechanism_enabled=False,
        chronaris_explicit_shift_enabled=False, chronaris_explicit_shift_weight=0., chronaris_event_pair_weight=0.)
    payload = {k:v for k,v in protocol.items() if k not in ('candidate','data_access_mode')}
    payload.update(format='chronaris.common_pretraining_checkpoint.v2', optimizer_updates=200, step_count=200,
        candidate_config={}, protocol_sha256=candidate_protocol_hash(**protocol),
        encoder_manifest={'backbone_config':dict(config)}, training_elapsed_s=123.,
        data_cursor={'samples_seen':3200,'micro_batches_seen':800,'sampling_order_sha256':'fixed'})
    payload.update({k:{'tensor':torch.tensor([1.,2.])} for k in STATE_KEYS})
    payload['canonical_training_state_sha256'] = canonical_training_state_sha256(*(payload[k] for k in STATE_KEYS))
    return payload


def test_execution_migration_only_changes_execution_metadata(tmp_path):
    original = _payload(); parent=tmp_path/'parent.pt';torch.save(original,parent)
    migrated = migrate_payload(original,parent_path=parent,evidence_sha256='evidence')
    assert not original['config']['cuda_graph_recurrence']
    assert migrated['config']['cuda_graph_recurrence']
    assert migrated['encoder_manifest']['backbone_config']['cuda_graph_recurrence']
    assert migrated['protocol_sha256'] != original['protocol_sha256']
    assert migrated['canonical_training_state_sha256'] == original['canonical_training_state_sha256']
    assert migrated['data_cursor']==original['data_cursor'] and migrated['training_elapsed_s']==123.
    for change in ('budget','weights','protocol','source'):
        invalid=deepcopy(original)
        if change=='budget':invalid['config']['batch_size']=8
        elif change=='weights':invalid['encoder_state_dict']['tensor'][0]=9.
        elif change=='protocol':invalid['protocol_sha256']='changed'
        else:invalid['source_code_sha256']='changed'
        with pytest.raises(ValueError):migrate_payload(invalid,parent_path=parent,evidence_sha256='evidence')
