from contextlib import nullcontext
from unittest.mock import patch
import json

import pytest
import torch

from chronaris.evaluation.application_tasks import stage45c as trial
from chronaris.evaluation.application_tasks import common_downstream_smoke as common
from chronaris.evaluation.application_tasks import stage45_recipe
from tests.evaluation.application_tasks.test_stage45b import short_inputs
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA required')
def test_guided_only_trial_reuses_exact_pretraining_and_rejects_contract_change(tmp_path):
    original = stage45_recipe.training_recipe
    def recipe(*args, **kwargs):
        kwargs['full'] = False
        return original(*args, **kwargs)
    with patch.object(common, 'contract_development_inputs', short_inputs), patch.object(common, 'development_gpu_lock', lambda:nullcontext(True)), patch.object(stage45_recipe,'training_recipe',recipe):
        result = common.run_common_contract_smoke(domain='clare', output_root=tmp_path/'parent/units/screen/0',
            full=True, methods=('chronaris',), recipe='stage4_reference', selection_inner_index=0, cuda_graph_recurrence=True)
    result['unit'] = dict(domain='clare', method='chronaris', variant='reference', inner_index=0, seed=17)
    directory=tmp_path/'parent/screen'; directory.mkdir()
    (directory/'0.json').write_text(json.dumps(result))
    sources={str(p):sha256_file(p) for p in (tmp_path/'parent').rglob('*') if p.is_file()}
    config=dict(stage45c_parent=str(tmp_path/'parent'),root=str(tmp_path/'trial'),data_root='unused',registry_path='unused')
    with patch.object(trial,'contract_development_inputs',short_inputs), patch.object(trial,'development_gpu_lock',lambda:nullcontext(True)):
        new=trial.run_unit(config,0,short=True)
        assert new['new_pretraining_updates']==0
        assert new['results'][0]['inherited_unchanged']
        payload=torch.load(new['results'][1]['training']['last_checkpoint_path'],map_location='cpu',weights_only=True)
        assert payload['config']['self_supervised_weight']==0
        assert payload['stage_update_counts']['joint_adaptation']==2
        assert all(r['public_weight']==0 for r in payload['update_rows'])
        assert all(sha256_file(p)==h for p,h in sources.items())
        old,_,_,root=trial.parent_unit(tmp_path/'parent',0)
        path=root/'data_contract.json';contract=json.loads(path.read_text());contract['seed']=29;path.write_text(json.dumps(contract))
        with pytest.raises(ValueError,match='data/targets/policy changed'):
            trial.run_unit(config|{'root':str(tmp_path/'invalid')},0,short=True)
