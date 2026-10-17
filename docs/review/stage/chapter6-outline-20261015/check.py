"""Read-only source assets; isolated CPU service and bounded review evidence."""
import json, os, re, signal, subprocess, sys, time, uuid
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / 'src'))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import torch
from chronaris.serving.fusion_workbench.acceptance import verify_demo

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evidence = Path(__file__).parent
    root = PROJECT / 'artifacts' / ('chapter6-review-' + uuid.uuid4().hex)
    root.mkdir()
    for name in ('cases', 'models'):
        source = PROJECT / 'artifacts/fusion-workbench' / name
        if source.exists():
            (root / name).symlink_to(source, target_is_directory=True)
    log = (root / 'service.stderr.log').open('w')
    proc = subprocess.Popen([sys.executable, 'scripts/runtime/run_fusion_workbench.py', '--data-root', str(root), '--port', '0'], cwd=PROJECT, env=os.environ, stdout=subprocess.PIPE, stderr=log, text=True)
    try:
        line = proc.stdout.readline()
        match = re.search(r'127\.0\.0\.1:(\d+)', line)
        assert match, line
        port = int(match[1]); base = f'http://127.0.0.1:{port}'
        def request(path, payload=None):
            body = None if payload is None else json.dumps(payload).encode()
            return json.load(urlopen(Request(base+path, data=body, headers={'Content-Type':'application/json'}), timeout=30))
        models = request('/api/models')
        acceptance_path = verify_demo(root, port=port)
        acceptance = json.loads(acceptance_path.read_text())
        cases = request('/api/cases')
        case = max(cases, key=lambda c: len(c['windows']))
        count = min(15, len(case['windows']))
        measurements = []
        for repeat in range(2):
            started = time.perf_counter()
            job = request('/api/runs', {'case_id':case['id'], 'start':0, 'stop':count})
            deadline = time.monotonic()+60
            while time.monotonic()<deadline:
                status = request('/api/runs/'+job['id'])
                if status['status'] in ('completed','failed'): break
                time.sleep(.05)
            assert status['status']=='completed', status
            result = request('/api/runs/'+job['id']+'/result')
            delivery = request('/api/runs/'+job['id']+'/deliver', {})
            assert delivery['windows']==count and result['downstream_readback']
            measurements.append({'repeat':repeat+1, 'windows':count, 'run_id':job['id'], 'pipeline_elapsed_s':result['elapsed_s'], 'model_load_s':result['model_load_s'], 'http_result_and_delivery_s':time.perf_counter()-started, 'windows_per_s':count/result['elapsed_s'], 'model':result['model']})
        status_text=Path(f'/proc/{proc.pid}/status').read_text()
        memory={k:int(re.search(r'^'+k+r':\s+(\d+)',status_text,re.M)[1])/1024 for k in ('VmRSS','VmHWM')}
        report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(), 'device':'cpu','threads':1,'scope':'isolated service, existing actual observations; two small-load repetitions, no capacity claim','registered_models':[{k:m[k] for k in ('id','semantic_event_enabled','compatible_cases')} for m in models], 'acceptance':acceptance,'measurements':measurements,'service_memory_mib':memory,'run_root':str(root)}
        (evidence/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate();proc.wait(timeout=5)
        log.close()

if __name__=='__main__': main()
