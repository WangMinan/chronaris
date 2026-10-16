"""Reproduce device-sensitive frozen features on their original RTX 4090."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys

HOST='wangminan@10.69.120.91'
SSH='/mnt/c/Windows/System32/OpenSSH/ssh.exe'
WORKSPACE='/home/wangminan/projects/chronaris-runtime-chapter5-reload-20261011'
SOURCE='d89dfcc5bf0c444aea09b6defc058e535b63b9ff723102a3535f570de8625b4b'
PYTHON='/home/wangminan/env/anaconda3/envs/chronaris/bin/python'
ASSETS='/mnt/e/chronaris-v4-results/2026-09-11-stage3-models/assets_sensor.json'
CHRONOS_ASSETS='/mnt/e/chronaris-v4-results/2026-09-11-stage3-models/assets.json'


def supports(spec):
    return (spec['method'],spec['domain']) in (('sensorllm_deepseek','clare'),('chronos2','dingxin'))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ssh(command, *, source=None, target=None):
    result=subprocess.run([SSH,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
        '-o','ServerAliveCountMax=3',HOST,command],stdin=source,stdout=target or subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('Frozen reference executor failed: '+result.stderr.decode(errors='replace')[-3000:])
    return result.stdout


def validate_output(result, request, features):
    import torch
    if (not supports(request) or request['feature_dim']<1
        or result['request']!=request or result['device']!='NVIDIA GeForce RTX 4090'
        or result['source_code_sha256']!=SOURCE or result['optimizer_updates']!=0
        or features.shape!=(len(request['sample_ids']),request['feature_dim']) or features.dtype!=torch.float32
        or not torch.isfinite(features).all()):
        raise ValueError('reference encoder identity, device or complete feature coverage changed')


def bind_evidence(result, root):
    from chronaris.evaluation.application_tasks.chapter5_assets import read,write,verify
    receipts=list(Path(root).glob('*/representation/reference_executor/receipt.json'))
    if len(receipts)!=len(result.get('results',result.get('rows',[]))):
        raise ValueError('reference executor provenance is incomplete')
    for path in receipts:
        record=read(path);verify(record['files'])
        result['files'].update(record['files']|{str(path.resolve()):digest(path)})
    return write(Path(root)/'result.json',result)


def encode_remote(spec, row, declaration, contract, raw, fold, output_root, role):
    import torch
    from chronaris.evaluation.application_tasks.chapter5_assets import read,write,verify
    from chronaris.modeling.fusion_encoders.recent_models import historical_channels
    from chronaris.representation import TrainOnlyRobustNormalizer
    from chronaris.representation.window_features import WindowFeatureBatch
    if not supports(spec) or role not in ('validation','held_out'):
        raise ValueError('reference executor is limited to frozen CLARE SensorLLM and Dingxin Chronos-2 inference')
    assets_path=ASSETS if spec['method']=='sensorllm_deepseek' else CHRONOS_ASSETS
    checkpoint=declaration['checkpoint_path'];saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    normalizer=TrainOnlyRobustNormalizer.from_manifest(saved['normalizer'])
    if (saved['assets_sha256']!=digest(assets_path) or saved['seed']!=spec['seed']
        or saved['method_name']!=spec['method']
        or saved['config']['data_manifest_sha256']!=contract['data_manifest_sha256']
        or not set(normalizer.fit_sample_ids)<=set(fold.train_sample_ids)
        or not set(raw.sample_ids)<=set(getattr(fold,role+'_sample_ids'))
        or set(raw.sample_ids)&set(fold.train_sample_ids)):
        raise ValueError('reference input escaped the frozen model, normalization or evaluation role')
    root=Path(output_root)/'reference_executor';root.mkdir(parents=True,exist_ok=True)
    values,mask=historical_channels(normalizer.transform(raw))
    request=dict(checkpoint=checkpoint,checkpoint_sha256=digest(checkpoint),assets_sha256=digest(assets_path),
        method=spec['method'],domain=spec['domain'],feature_dim=declaration['feature_dim'],
        sample_ids=list(raw.sample_ids),source_sample_hashes=list(raw.source_sample_hashes),
        fold_id=fold.fold_id,role=role,seed=spec['seed'],channel_names=list(saved['channel_names']),
        data_manifest_sha256=contract['data_manifest_sha256'],worker_sha256=digest(__file__),
        values_sha256=hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest(),
        mask_sha256=hashlib.sha256(mask.contiguous().numpy().tobytes()).hexdigest())
    manifest=root/'receipt.json';output=root/'features.pt'
    if manifest.exists():
        record=read(manifest);verify(record['files']);features=torch.load(output,weights_only=True)
        validate_output(record,request,features)
    else:
        payload=root/'input.pt';torch.save(dict(request=request,values=values,mask=mask),payload)
        job=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        base=f'/home/wangminan/chronaris-sensor-reference-20261016/{request["worker_sha256"]}'
        remote=f'{base}/{job}';q=shlex.quote
        with Path(__file__).open('rb') as stream:
            ssh(f'mkdir -p {q(remote)} && cat > {q(base+"/worker.py")}',source=stream)
        with payload.open('rb') as stream:ssh(f'cat > {q(remote+"/input.pt")}',source=stream)
        command=f'cd {q(WORKSPACE)} && PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {q(PYTHON)} {q(base+"/worker.py")} {q(remote)} {q(digest(payload))}'
        (root/'remote.log').write_bytes(ssh(command))
        temporary=root/'features.download'
        with temporary.open('wb') as stream:ssh(f'cat {q(remote+"/features.pt")}',target=stream)
        record=json.loads(ssh(f'cat {q(remote+"/receipt.json")}'))
        if digest(temporary)!=record['features_sha256']:raise ValueError('reference feature transfer hash mismatch')
        features=torch.load(temporary,weights_only=True);validate_output(record,request,features)
        temporary.replace(output)
        record.update(host=HOST,remote_root=remote,files={str(p.resolve()):digest(p) for p in (payload,output,root/'remote.log')})
        write(manifest,record)
    return WindowFeatureBatch(raw.sample_ids,raw.context_durations_s,features,mask.any((1,2)),
        spec['method'],fold.fold_id,digest(checkpoint),raw.source_sample_hashes)


def reference_worker(root, input_sha256):
    sys.path.insert(0,str(Path(WORKSPACE)/'src'))
    import torch
    from chronaris.evaluation.application_tasks.chapter5_assets import read,write
    from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
    from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
    from chronaris.modeling.fusion_encoders.recent_models import verify_assets,Chronos2WindowEncoder
    from chronaris.modeling.fusion_encoders.sensorllm_adapter import SensorLLMWindowEncoder
    from chronaris.evaluation.application_tasks.recent_model_training import load_encoder_checkpoint
    from chronaris.evaluation.application_tasks.recent_model_smoke import _export
    from chronaris.modeling.training.rng import isolated_training_rng
    root=Path(root);payload=root/'input.pt'
    if digest(payload)!=input_sha256:raise ValueError('reference input transfer changed')
    data=torch.load(payload,map_location='cpu',weights_only=True);request=data['request']
    if not supports(request):raise ValueError('unqualified reference method or domain')
    assets_path=ASSETS if request['method']=='sensorllm_deepseek' else CHRONOS_ASSETS
    if (request['worker_sha256']!=digest(__file__) or v4_workflow_source_sha256()!=SOURCE
        or digest(request['checkpoint'])!=request['checkpoint_sha256'] or digest(assets_path)!=request['assets_sha256']):
        raise ValueError('reference code, checkpoint or asset provenance changed')
    values,mask=data['values'],data['mask']
    if (values.dtype!=torch.float32 or mask.dtype!=torch.bool or values.shape!=mask.shape
        or values.ndim!=3 or values.shape[0]!=len(request['sample_ids']) or values.shape[-1]!=len(request['channel_names'])
        or request['role'] not in ('validation','held_out') or not torch.isfinite(values).all()
        or hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()!=request['values_sha256']
        or hashlib.sha256(mask.contiguous().numpy().tobytes()).hexdigest()!=request['mask_sha256']):
        raise ValueError('reference input shape, finite values or hashes differ')
    saved=torch.load(request['checkpoint'],map_location='cpu',weights_only=True)
    if (saved['seed']!=request['seed'] or saved['channel_names']!=request['channel_names']
        or saved['method_name']!=request['method'] or saved['assets_sha256']!=request['assets_sha256']
        or saved['config']['data_manifest_sha256']!=request['data_manifest_sha256']):
        raise ValueError('reference checkpoint metadata differs')
    if (root/'receipt.json').exists():
        record=read(root/'receipt.json');features=torch.load(root/'features.pt',weights_only=True)
        if digest(root/'features.pt')!=record['features_sha256']:raise ValueError('cached reference features changed')
        validate_output(record,request,features);print('reused qualified reference export');return
    torch.set_num_threads(1)
    with development_gpu_lock() as acquired:
        if not acquired:raise RuntimeError('reference GPU is busy; preserve outputs and retry explicitly')
        if torch.cuda.get_device_name()!='NVIDIA GeForce RTX 4090':raise ValueError('reference execution requires original GPU model')
        torch.cuda.set_per_process_memory_fraction(.95)
        assets=verify_assets(assets_path)
        factory=SensorLLMWindowEncoder if request['method']=='sensorllm_deepseek' else Chronos2WindowEncoder
        with isolated_training_rng(request['seed']):encoder=factory(assets,len(request['channel_names']))
        if request['method']=='sensorllm_deepseek':encoder.channel_names=tuple(request['channel_names'])
        if encoder.feature_dim!=request['feature_dim']:raise ValueError('reference feature dimension differs from frozen declaration')
        load_encoder_checkpoint(encoder,saved['encoder'])
        # Release loader temporaries before inference on the Windows-hosted reference GPU.
        torch.cuda.empty_cache()
        features,cost=_export(encoder,{request['role']:(values,mask)},{request['role']:None});features=features[request['role']]
        if digest(request['checkpoint'])!=request['checkpoint_sha256']:raise ValueError('reference checkpoint changed during inference')
        torch.save(features,root/'features.pt')
        record=dict(request=request,device=torch.cuda.get_device_name(),source_code_sha256=SOURCE,
            optimizer_updates=0,features_sha256=digest(root/'features.pt'),cost=cost,torch=torch.__version__)
        validate_output(record,request,features);write(root/'receipt.json',record)
        print('completed frozen reference export',len(features),'windows',flush=True)


if __name__=='__main__':reference_worker(sys.argv[1],sys.argv[2])
