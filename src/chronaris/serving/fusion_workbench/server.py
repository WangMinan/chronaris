"""Persistent processing jobs, immutable input selections and resumable chunks."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime, timezone
import csv
import fcntl
from pathlib import Path
import re
import threading
import uuid

from chronaris.representation import load_fusion_stream_batch
from chronaris.serving.fusion_workbench.catalog import Catalog
from chronaris.serving.fusion_workbench.cases import read_json, write_json
from chronaris.serving.fusion_workbench.fields import FieldReviews
from chronaris.serving.fusion_workbench.inference import run_case, window_key, implementation_digest
from chronaris.serving.fusion_workbench.presentation import display_text
from chronaris.serving.fusion_workbench.http import make_server


class DemoStore:
    def __init__(self, cases_root, runs_root, *, audit_root=None, context_path=None, device="cpu"):
        from chronaris.serving.fusion_workbench.device import execution_identity
        self.execution = execution_identity(device)
        self.device = self.execution["device"]
        self._gpu_context = None
        self.root = Path(runs_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / '.server.lock').open('a')
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise RuntimeError('此产物目录已有服务运行') from None
        try:
            if self.device != 'cpu':
                from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
                self._gpu_context = development_gpu_lock()
                if not self._gpu_context.__enter__():
                    self._gpu_context.__exit__(None, None, None)
                    self._gpu_context = None
                    raise RuntimeError('GPU正由其他实验使用，请稍后再启动GPU服务')
            self.catalog = Catalog(cases_root, audit_root)
            self.cases = self.catalog.refresh()
            self.reviews = FieldReviews(self.root.parent / 'field-reviews.json', context_path)
            for path in self.root.glob('*/job.json'):
                job = read_json(path)
                if job['status'] not in ('completed','failed'):
                    job.update(status='failed',error='服务已中断，可按原参数继续处理。')
                    write_json(path,job)
        except Exception:
            if self._gpu_context:
                self._gpu_context.__exit__(None, None, None)
            self._lock_file.close()
            raise
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.active = None

    def close(self):
        self.executor.shutdown(wait=True)
        if self._gpu_context:
            self._gpu_context.__exit__(None, None, None)
            self._gpu_context = None
        self._lock_file.close()

    def job_root(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise KeyError('任务不存在')
        root = self.root / job_id
        if not (root / 'job.json').is_file():
            raise KeyError('任务不存在')
        return root

    def history(self):
        jobs = sorted((read_json(p) for p in self.root.glob('*/job.json')),key=lambda x:x['created_at'],reverse=True)
        return [dict(j,title=display_text(j['title']),retryable=j['status']=='failed' and 'request' in j) for j in jobs]

    def records(self):
        with self.lock:
            self.cases = self.catalog.refresh()
            return [c.summary() for c in self.cases.values()]

    def import_record(self, payload, *, archive=False):
        with self.lock:
            if self.active:
                raise RuntimeError('请等待当前处理任务完成后接入数据')
            result = self.catalog.prepare_source(payload) if archive else self.catalog.import_record(payload)
            self.cases = self.catalog.cases
            return result

    def submit(self, payload):
        required = {'case_id','start','stop'}
        if not isinstance(payload,dict) or not required <= set(payload) or set(payload)-required-{'model_id','incremental'}:
            raise ValueError('请选择数据、模型和起止窗口')
        if not isinstance(payload['case_id'],str):
            raise ValueError('数据记录标识不合法')
        with self.lock:
            case = self.cases.get(payload['case_id'])
            start,stop = payload['start'],payload['stop']
            if case is None or type(start) is not int or type(stop) is not int or not 0<=start<stop<=len(case.manifest['windows']):
                raise ValueError('起止窗口范围不合法')
            digest = payload.get('model_id',case.manifest['checkpoint_sha256'])
            if not isinstance(digest,str) or type(payload.get('incremental',False)) is not bool:
                raise ValueError('模型或增量参数不合法')
            case = self.catalog.bind(case,digest)
            indices = list(range(start,stop))
            if payload.get('incremental'):
                revision = implementation_digest()
                completed = {key for j in self.history() if j['status']=='completed' and j.get('case_id')==payload['case_id']
                             and j.get('implementation_digest')==revision
                             for key in j.get('window_keys',[])}
                indices = [i for i in indices if window_key(case,i,device=self.device) not in completed]
                if not indices:
                    raise ValueError('所选范围已处理完成，没有新增窗口')
            request = dict(payload,model_id=digest,incremental=payload.get('incremental',False))
            return self._create(case,indices,request)

    def retry(self, job_id):
        with self.lock:
            previous_root = self.job_root(job_id)
            previous = read_json(previous_root / 'job.json')
            if previous['status']!='failed' or 'request' not in previous:
                raise ValueError('此任务没有可恢复的失败请求')
            snapshot = read_json(previous_root / 'input.json')
            original = self.cases.get(previous['case_id'])
            if original is None:
                raise ValueError('原始数据记录已移除')
            case = copy(original)
            case.manifest = snapshot
            case = self.catalog.bind(case,previous['request']['model_id'])
            return self._create(case,previous['selected_indices'],previous['request'],retry_of=job_id)

    def _create(self, case, indices, request, retry_of=None):
        if self.active:
            raise RuntimeError('已有推理任务运行，请等待完成')
        job_id = uuid.uuid4().hex
        root = self.root / job_id
        root.mkdir()
        write_json(root / 'input.json',case.manifest)
        windows = case.manifest['windows']
        job = {'id':job_id, 'case_id':request['case_id'],'title':case.summary()['title'], 'request':request,
            'implementation_digest':implementation_digest(), 'selected_indices':indices, 'window_keys':[window_key(case,i,device=self.device) for i in indices], 'retry_of':retry_of,
            'range_s':[windows[indices[0]]['start_s'],windows[indices[-1]]['start_s']+windows[indices[-1]]['duration_s']],
            'device':self.device, 'execution':self.execution, 'status':'queued','done':0,'total':len(indices),'created_at':datetime.now(timezone.utc).isoformat()}
        write_json(root / 'job.json',job)
        self.active = job_id
        self.executor.submit(self._run,case,indices,root,job)
        return job.copy()

    def _run(self, case, indices, root, job):
        def progress(status,done):
            job.update(status=status,done=done)
            write_json(root / 'job.json',job)
        try:
            options = {'resume_root': self.job_root(job['retry_of'])} if job.get('retry_of') else {}
            result = run_case(case,indices,root,progress,device=self.device,**options)
            job.update(status='completed',done=len(indices),elapsed_s=result['elapsed_s'],reused_windows=result.get('reused_windows',0))
        except Exception as exc:
            job.update(status='failed',error=display_text(exc))
        finally:
            try:
                write_json(root / 'job.json',job)
            finally:
                with self.lock:
                    self.active = None

    def field_rows(self, case_id):
        case = copy(self.cases[case_id])
        payload = self.catalog.payload(case.manifest['checkpoint_sha256'])
        labels = {name: label for name,label in payload.get('vehicle_field_labels',()) if label != name}
        labels.update(case.manifest.get('field_labels',{}))
        case.manifest = dict(case.manifest,field_labels=labels)
        return self.reviews.rows(case)

    def deliver(self, job_id):
        with self.lock:
            root = self.job_root(job_id)
            if read_json(root/'job.json')['status']!='completed':
                raise ValueError('请先完成处理任务')
            data = load_fusion_stream_batch(root)
            temporary = root / 'pooled_features.tmp'
            with temporary.open('w',encoding='utf-8',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', *[f'feature_{i:02}' for i in range(data.pooled_embedding.shape[-1])]])
                writer.writerows([sample,*row.tolist()] for sample,row in zip(data.sample_ids,data.pooled_embedding))
            temporary.replace(root / 'pooled_features.csv')
            receipt = {'run_id':job_id,'status':'completed','consumer':'标准表示读取与汇总特征接口',
                       'windows':len(data.sample_ids),'valid_points':int(data.valid_mask.sum()),
                       'output_dim':data.pooled_embedding.shape[-1],'created_at':datetime.now(timezone.utc).isoformat()}
            write_json(root / 'delivery.json',receipt)
            return receipt
