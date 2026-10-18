import json
import time
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_inputs import FairResidualData
from chronaris.evaluation.application_tasks.readout_selected_training import fit_readout_selected


def test_real_cpu_loop_selects_readout_checkpoint_without_opening_score(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(4);n=12
    subjects=['fit']*6+['select']*3+['score']*3
    rows=[dict(sample_id=str(i),subject=s,target=float(i%3),persistence=0.,accepted=True) for i,s in enumerate(subjects)]
    values={}
    for name,d in [('physiology',16),('vehicle',12)]:
        values[name]=rng.normal(size=(n,96,d)).astype(np.float32)
        values[name+'_mask']=np.ones((n,96,d),bool);values[name+'_age']=np.ones((n,96,d),np.float32)*.01
    values['vehicle'][:,-1,8]=[r['target'] for r in rows]
    values['anchor']=np.zeros((n,96));values['metadata']=np.array(json.dumps(dict(windows=rows)))
    snapshot=tmp_path/'inputs.npz';np.savez_compressed(snapshot,**values)
    q=tmp_path/'q.json';q.write_text(json.dumps(dict(status='passed',gates={'test':True},snapshot_sha256=file_digest(snapshot))))
    fold=dict(name='test',fit=['fit'],select=['select'],score=['score'])
    data=FairResidualData(snapshot,q,fold,'cpu')
    _,receipt=fit_readout_selected('aircraft',data,tmp_path/'weights',tmp_path,'test',{'seed':17},time.monotonic()+120)
    h=pd.read_csv(tmp_path/'test_history.csv')
    best=h[h.epoch==receipt['best_epoch']].iloc[0]
    assert receipt['complete'] and not data.score_open
    assert abs(best.selection_rmse-receipt['selection_rmse'])<1e-8
    assert receipt['selection_policy']=='common_readout'
    assert best.selection_rmse <= h.selection_rmse.min()+data.target_scale*1e-6
