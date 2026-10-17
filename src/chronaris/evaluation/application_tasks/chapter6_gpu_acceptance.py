"""Same-weight CPU/GPU service measurements and standard-output comparisons."""
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from urllib.request import urlopen
import zipfile

import torch

from chronaris.evaluation.application_tasks.chapter6_acceptance import AcceptanceService, verify_http_errors
from chronaris.evaluation.application_tasks.chapter6_mechanisms import verify_application_mechanisms
from chronaris.representation import load_fusion_stream_batch
from chronaris.serving.fusion_workbench.cases import Case, read_json, write_json
from chronaris.serving.fusion_workbench.inference import run_case, implementation_digest


def compare(a, b):
    assert a.sample_ids == b.sample_ids and a.checkpoint_sha256 == b.checkpoint_sha256
    assert a.source_sample_hashes == b.source_sample_hashes
    assert torch.equal(a.valid_mask, b.valid_mask) and torch.equal(a.timestamps_s, b.timestamps_s)
    torch.testing.assert_close(a.sequence_embedding, b.sequence_embedding, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a.pooled_embedding, b.pooled_embedding, atol=1e-5, rtol=1e-5)
    return dict(sequence_max_abs=float((a.sequence_embedding-b.sequence_embedding).abs().max()),
        pooled_max_abs=float((a.pooled_embedding-b.pooled_embedding).abs().max()))


def live_service(project, root, port):
    service = AcceptanceService(project, root, device='cuda')
    matches = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
            if b'scripts/runtime/run_fusion_workbench.py' in args and b'--data-root' in args:
                directory=Path(args[args.index(b'--data-root')+1].decode())
                if not directory.is_absolute(): directory=Path(project)/directory
                if directory.resolve()==Path(root).resolve(): matches.append(int(p.parent.name))
        except (FileNotFoundError, PermissionError):
            pass
    assert len(matches) == 1, matches
    service.process = SimpleNamespace(pid=matches[0])
    service.base = f'http://127.0.0.1:{port}'; service.port=port
    return service


def measure_service(service, evidence, *, device):
    case = Case(service.root/'cases/simulation-windows/case.json')
    rows = []
    for count in (1, 4, 16):
        for repeat in (1, 2, 3):
            start = time.perf_counter()
            job = service.request('/api/runs', dict(case_id=case.manifest['id'], start=0, stop=count))
            service.wait(job)
            completed = time.perf_counter()-start
            result = service.request('/api/runs/'+job['id']+'/result')
            queried = time.perf_counter()-start
            delivery = service.request('/api/runs/'+job['id']+'/deliver', {})
            delivered = time.perf_counter()-start
            assert result['device'] == device and result['reused_windows'] == 0
            assert result['checkpoint_sha256'] == case.manifest['checkpoint_sha256']
            assert delivery['windows'] == count
            trace = result['execution_batches']
            assert len(trace) == count
            assert all(r['parameter_device']==r['forward_output_device']==device
                       and r['input_tensor_devices']==[device] for r in trace)
            timings = read_json(service.root/'runs'/job['id']/'timings.json')
            rows.append(dict(device=device, windows=count, repeat=repeat, run_id=job['id'],
                process_first_task=count==1 and repeat==1, **timings,
                **{k:sum(r[k] for r in trace) for k in ('normalized_s','input_transfer_s','forward_s','output_transfer_pool_s')},
                http_completed_s=completed, http_result_s=queried, http_delivery_s=delivered,
                windows_per_s=count/timings['pipeline_total_s'], **service.memory()))
            write_json(evidence/f'{device.replace(":", "-")}-measurements.json', rows)
            print(device, count, repeat, timings['pipeline_total_s'], flush=True)
    return rows


def download_case(service, case_id, output):
    job = service.request('/api/runs', dict(case_id=case_id, start=0, stop=3))
    service.wait(job)
    result = service.request('/api/runs/'+job['id']+'/result')
    delivery = service.request('/api/runs/'+job['id']+'/deliver', {})
    with urlopen(service.base+'/api/runs/'+job['id']+'/download/features.zip') as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            assert set(archive.namelist())=={'fusion_stream.npz','representation_manifest.json','features.csv','result.json'}
            archive.extractall(output)
    batch=load_fusion_stream_batch(output)
    with (output/'features.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==len(batch.sample_ids)*96
    for i,row in enumerate(rows):
        assert row['sample_id']==batch.sample_ids[i//96]
        assert (row['valid']=='True')==bool(batch.valid_mask[i//96,i%96])
        assert float(row['query_time_s'])==float(batch.timestamps_s[i//96,i%96])
        torch.testing.assert_close(torch.tensor([float(row[f'feature_{d:02}']) for d in range(64)]),
            batch.sequence_embedding[i//96,i%96],atol=0,rtol=0)
    for name in ('pooled_features.csv','delivery.json'):
        shutil.copyfile(service.root/'runs'/job['id']/name,output/name)
    return batch,dict(run_id=job['id'],device=result['device'],shape=list(batch.sequence_embedding.shape),
        delivery=delivery,model=result['model'],execution=result['execution_batches'],csv_exact=True)


def run_gpu_acceptance(project, root, evidence, port=8768):
    project,root,evidence=Path(project),Path(root),Path(evidence)
    evidence.mkdir(parents=True,exist_ok=True)
    gpu=live_service(project,root,port)
    cpu_root=root/'cpu-comparison'
    if cpu_root.exists(): raise ValueError('preserve previous CPU measurements')
    shutil.copytree(root/'cases',cpu_root/'cases')
    cpu=AcceptanceService(project,cpu_root).start()
    comparisons={}
    try:
        rows=measure_service(gpu,evidence,device='cuda:0')+measure_service(cpu,evidence,device='cpu')
        for case_id in ('simulation-validation','dingxin-validation'):
            cpu_batch,cpu_report=download_case(cpu,case_id,evidence/(case_id+'-cpu'))
            gpu_batch,gpu_report=download_case(gpu,case_id,evidence/(case_id+'-gpu'))
            comparison=compare(cpu_batch,gpu_batch)
            case=Case(root/'cases'/case_id/'case.json')
            run_case(case,[0,1,2],root/'verification'/(case_id+'-gpu-batch'),batch_size=2,device='cuda')
            batched=load_fusion_stream_batch(root/'verification'/(case_id+'-gpu-batch'))
            comparison['gpu_batch_vs_window']=compare(gpu_batch,batched)
            comparisons[case_id]=dict(cpu=cpu_report,gpu=gpu_report,comparison=comparison)
            write_json(evidence/(case_id+'-mechanisms.json'),verify_application_mechanisms(case,device='cuda'))
        write_json(evidence/'numeric-comparison.json',dict(status='passed',atol=1e-5,rtol=1e-5,cases=comparisons))
        write_json(evidence/'http-errors.json',verify_http_errors(gpu,Case(root/'cases/dingxin-validation/case.json')))
        with (evidence/'performance.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_json(evidence/'performance.json',rows)
        write_json(evidence/'environment.json',dict(device=gpu.request('/api/runtime'),
            nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
            cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            implementation_digest=implementation_digest(),cpu_startup_s=cpu.startup_s,
            scope='one fresh service per device; first task marked; model reload on every task; no chunk reuse; warm OS cache; batch size 1; CPU threads 1',
            cpu_root=str(cpu_root),gpu_root=str(root)))
    finally: cpu.stop()
    return comparisons


def verify_gpu_recovery(project, root, evidence, port=8768):
    import os
    import signal
    from chronaris.serving.fusion_workbench.server import DemoStore
    project,root,evidence=Path(project),Path(root),Path(evidence)
    gpu=live_service(project,root,port)
    try:
        other=DemoStore(root/'cases',root/'verification/busy-test',device='cuda')
    except RuntimeError as exc:
        assert 'GPU正由其他实验使用' in str(exc)
        busy_rejection=str(exc)
    else:
        other.close();raise AssertionError('another GPU service bypassed the resource lock')
    case=Case(root/'cases/simulation-validation/case.json')
    cpu_case=read_json(evidence/'numeric-comparison.json')['cases']['simulation-validation']['cpu']['run_id']
    resumed=run_case(case,[0,1,2],root/'verification/cross-device-resume',device='cuda',
        resume_root=Path(read_json(evidence/'environment.json')['cpu_root'])/'runs'/cpu_case)
    assert resumed['reused_windows']==0
    assert all(r['forward_output_device']=='cuda:0' for r in resumed['execution_batches'])
    failed=gpu.request('/api/runs',dict(case_id='simulation-windows',start=0,stop=16))
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        status=gpu.request('/api/runs/'+failed['id'])
        if 1 <= status['done'] < 16:break
        time.sleep(.01)
    assert 1 <= status['done'] < 16,status
    os.kill(gpu.process.pid,signal.SIGKILL)
    time.sleep(.3)
    restarted=AcceptanceService(project,root,device='cuda').start()
    try:
        original=restarted.request('/api/runs/'+failed['id'])
        assert original['status']=='failed'
        job=restarted.wait(restarted.request('/api/runs/'+failed['id']+'/retry',{}))
        result=restarted.request('/api/runs/'+job['id']+'/result')
        assert result['device']=='cuda:0' and result['reused_windows']>=status['done']
        assert restarted.request('/api/runs/'+failed['id'])['status']=='failed'
        reference=read_json(evidence/'performance.json')
        prior=next(r for r in reference if r['device']=='cuda:0' and r['windows']==16)
        difference=compare(load_fusion_stream_batch(root/'runs'/prior['run_id']),load_fusion_stream_batch(root/'runs'/job['id']))
        write_json(evidence/'gpu-recovery.json',dict(status='passed',interrupted=original,recovered=job,
            reused_windows=result['reused_windows'],original_preserved=True,comparison=difference,
            cross_device_reused_windows=0,cross_device_actual_execution=resumed['execution_batches'],busy_rejection=busy_rejection))
    finally:restarted.stop()


def capture_gpu_cases(project, root, evidence, base):
    import os
    browser=Path(project)/'artifacts/fusion-workbench/browser'
    env=dict(os.environ,WORKBENCH_URL=base,WORKBENCH_DATA_ROOT=str(root),
        WORKBENCH_CASE_ID='simulation-validation',
        LD_LIBRARY_PATH=str(browser/'libs/extracted/usr/lib/x86_64-linux-gnu'),
        FONTCONFIG_FILE=str(browser/'fonts.conf'),PLAYWRIGHT_BROWSERS_PATH=str(browser/'browsers'))
    for kind, script in (('simulation','fusion_workbench_browser.mjs'),('measured','fusion_workbench_case_capture.mjs')):
        ui=Path(evidence)/(kind+'-ui')
        subprocess.run(['node','tests/runtime/'+script],cwd=project,
            env=dict(env,DEMO_SCREENSHOTS=str(ui)),check=True)
        snapshot=read_json(ui/'preview-data.json');run=snapshot['preview_job']['id']
        assert snapshot['device']=='cuda:0' and snapshot['preview_job']['device']=='cuda:0'
        example=ui/'feature-example';example.mkdir(exist_ok=True)
        for name in ('fusion_stream.npz','representation_manifest.json','features.csv','pooled_features.csv','delivery.json'):
            shutil.copyfile(Path(root)/'runs'/run/name,example/name)
        assert read_json(example/'delivery.json')['run_id']==run
        write_json(ui/'case-job.json',snapshot['preview_job'])
        destination=Path(root).parent/'browser-inputs'/kind;destination.mkdir(parents=True,exist_ok=True)
        for file in list(ui.glob('*observation*.json'))+list(ui.glob('observed-input.json')):
            shutil.move(file,destination/file.name)
