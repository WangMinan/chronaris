import json
from pathlib import Path
import joblib
import torch
from chronaris.evaluation.application_tasks.chapter5_native import unit_record, _inputs, _encode, compare_predictions
from chronaris.evaluation.application_tasks.v4_grouped_consumers import evaluate_native_consumers

root=Path('/home/wangminan/chronaris-migration-20261012/replay-diagnostic'); root.mkdir(exist_ok=True)
torch.set_num_threads(1)
results=[]
for domain,index in [('clare',0),('cogpilot',135),('dingxin',255)]:
    unit=unit_record('/mnt/e/chronaris-v4-results/2026-10-02-stage5-performance-recovery',index)
    for row in unit['results']:
        evaluation=row.get('evaluation',row)
        contract_file=Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json'
        record=json.loads(contract_file.read_text());contract=record['contract']
        _,fold,targets,context,raw=_inputs(unit['unit'],contract,'validation',4)
        output=_encode(unit['unit'],row,record['declaration'],contract,raw,fold,root/domain/row['route'],'validation')
        for family,component in evaluation['consumers']['components'].items():
            saved=joblib.load(component['model_path'])
            prediction=evaluate_native_consumers(saved['consumer'],output=output,targets=targets,context=context)
            expected=json.loads(Path(component['result_path']).read_text())['evaluations']['validation']['prediction_rows']
            expected=[r for r in expected if r['sample_id'] in set(raw.sample_ids)]
            actual=prediction['prediction_rows']
            entry={'domain':domain,'route':row['route'],'family':family,'samples':len(raw.sample_ids)}
            try:
                entry['comparison']=compare_predictions(expected,actual);entry['strict_replay_passed']=True
            except ValueError as e:
                entry['strict_replay_passed']=False;entry['error']=str(e)
            entry['expected']=expected;entry['actual']=actual
            results.append(entry)
            print(json.dumps({k:v for k,v in entry.items() if k not in ['expected','actual']}),flush=True)
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
