import json,os,subprocess,time,hashlib
from pathlib import Path
root=Path('/mnt/e/chronaris-v4-results/2026-10-16-chapter5-chronos-recovery')
launcher='/home/wangminan/projects/chronaris-runtime-chronos-reference-20261016/scripts/evaluation/application_tasks/run_chapter5_delivery.py'
workspace='/home/wangminan/projects/chronaris-runtime-local-validation-20261012'
python='/home/wangminan/env/anaconda3/envs/chronaris/bin/python'
result=root/'worker_results/replay_reference_262.json';log=root/'logs/replay_reference_262.log';log.parent.mkdir(exist_ok=True)
args=[python,launcher,'worker','--root',str(root),'--workspace',workspace,'--stage','replay:reference_262','--result',str(result)]
started=time.time()
with log.open('a') as output:code=subprocess.call(args,cwd=workspace,stdout=output,stderr=subprocess.STDOUT)
assert code==0,f'worker failed {code}: {log}'
value=json.loads(result.read_text());assert value['status']=='completed'
assert len(value['rows'])==2 and all(r['samples']==6 and r['strict']['passed'] and not r['class_flips'] and r['strict']['atol']==r['strict']['rtol']==1e-5 for r in value['rows'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={str(result):sha(result)}|value['files']
for p,digest in files.items():assert sha(p)==digest
plan=json.loads((root/'delivery_plan.json').read_text())
receipt=dict(plan_sha256=plan['plan_sha256'],stage='replay:reference_262',exit_code=code,files=files,started_at=started,seconds=time.time()-started,log=str(log),command=args)
p=root/'steps/replay_reference_262.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(value['rows'],indent=2))
