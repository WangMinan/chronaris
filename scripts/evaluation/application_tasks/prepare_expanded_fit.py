"""Transport read-only extraction of the explicitly registered training cohort."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
from chronaris.dataset.cogpilot_expanded_fit import extract_expanded_fit
from chronaris.dataset.cogpilot_fair_residual import extract_fair_residual
from chronaris.dataset.cogpilot_shape_extract import extract_shape_native


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.partial').exists():raise FileExistsError(a.output)
    root=Path(__file__).resolve().parents[3]
    contract=json.loads(Path('/mnt/e/chronaris-v4-results/2026-09-30-stage5-sensor-recovery/training/135/cogpilot/chronaris/self_supervised/evaluation/common_contract.json').read_text())['contract']
    subjects=sorted({s.split('::')[0] for s in contract['roles']['train']})
    assert subjects==['sub-cp'+s for s in ('005','006','009','013','015','018','020','023','024','027','028','029','030','031','033','035','036','039')]
    meta=json.loads((root/'artifacts/application_evaluation/2026-10-18-controlled-readouts/source_semantics.json').read_text())
    records=[r for r in meta['perf_metrics'] if f"sub-cp{int(r['subject']):03d}" in subjects]
    code='\n'.join((root/'src/chronaris/dataset'/name).read_text() for name in
                   ('cogpilot_performance.py','cogpilot_history_features.py','cogpilot_shape_features.py'))
    code+='\n'+'\n'.join(inspect.getsource(f) for f in (extract_fair_residual,extract_shape_native,extract_expanded_fit))
    code+='\nimport io,sys\nb=io.BytesIO()\nnp.savez_compressed(b,**extract_expanded_fit(request))\nsys.stdout.buffer.write(b.getvalue())\n'
    request=dict(code=code,subjects=subjects,records=records,metadata_sha256=meta['perf_metrics_sha256'],
                 source_root='/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-ils')
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote('import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.partial').open('xb') as out,a.output.with_suffix('.log').open('xb') as log:
        proc=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                             'wangminan@10.69.120.91',command],input=json.dumps(request).encode(),stdout=out,stderr=log,timeout=14400)
    if proc.returncode:raise RuntimeError('source extraction failed; see '+str(a.output.with_suffix('.log')))
    a.output.with_suffix('.partial').rename(a.output)
    print(a.output,flush=True)
