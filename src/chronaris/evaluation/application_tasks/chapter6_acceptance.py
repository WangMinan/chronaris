"""Reproducible CPU service acceptance with actual observations and frozen weights."""
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import torch

from chronaris.evaluation.application_tasks.chapter6_mechanisms import verify_application_mechanisms
from chronaris.serving.fusion_workbench.acceptance import verify_demo
from chronaris.serving.fusion_workbench.cases import ARRAYS, Case, read_json, write_json
from chronaris.serving.fusion_workbench.catalog import Catalog
from chronaris.serving.fusion_workbench.prepare import prepare_dingxin_case
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


class AcceptanceService:
    def __init__(self, project, root, device="cpu"):
        self.project, self.root = Path(project), Path(root)
        self.process = None
        self.device = device

    def start(self):
        started = time.perf_counter()
        self.log = (self.root / 'service.log').open('a')
        self.process = subprocess.Popen([sys.executable, 'scripts/runtime/run_fusion_workbench.py',
            '--data-root', str(self.root), '--port', '0', '--device', self.device], cwd=self.project, env=os.environ,
            stdout=subprocess.PIPE, stderr=self.log, text=True)
        line = self.process.stdout.readline()
        match = re.search(r'127\.0\.0\.1:(\d+)', line)
        if not match:
            self.stop()
            raise RuntimeError('service startup failed: ' + line)
        self.port = int(match[1])
        self.base = f'http://127.0.0.1:{self.port}'
        self.startup_s = time.perf_counter() - started
        return self

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(self.base + path, data=body, headers={'Content-Type':'application/json'}), timeout=120) as response:
            return json.load(response)

    def wait(self, job):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = self.request('/api/runs/' + job['id'])
            if status['status'] in ('completed', 'failed'):
                assert status['status'] == 'completed', status
                return status
            time.sleep(.05)
        raise TimeoutError('processing task exceeded 180 seconds')

    def memory(self):
        text = Path(f'/proc/{self.process.pid}/status').read_text()
        return {key + '_mib': int(re.search(r'^' + key + r':\s+(\d+)', text, re.M)[1]) / 1024
                for key in ('VmRSS', 'VmHWM')}

    def stop(self, *, interrupted=False):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL if interrupted else signal.SIGINT)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate(); self.process.wait(timeout=5)
        if self.process:
            self.process.stdout.close()
            self.log.close()
        self.process = None


def prepare_application_acceptance(*, project, checkpoint, root):
    project, root = Path(project), Path(root)
    root.mkdir(parents=True, exist_ok=False)
    original = Case(project / 'artifacts/fusion-workbench/cases/dingxin-validation/case.json')
    case = prepare_dingxin_case(checkpoint=checkpoint,
        snapshot_root=original.manifest['source']['snapshot'],
        audit_root=project / 'docs/artifacts/runs/2026-07-10_fixed-data-audit',
        destination=root / 'cases/dingxin-validation')
    catalog = Catalog(root / 'cases', project / 'docs/artifacts/runs/2026-07-10_fixed-data-audit')
    catalog.refresh()
    # Retain the prior model as an explicit alternate selection, with separate identities.
    catalog.register_model(original.checkpoint, original.schema)
    source = catalog.sources()[0]
    record = catalog.prepare_source(dict(source_id=source['id'], start_s=source['start_s'],
        stop_s=source['stop_s'], model_id=case.manifest['checkpoint_sha256']))
    write_json(root / 'acceptance-input.json', dict(record_id=record['id'],
        checkpoint_sha256=case.manifest['checkpoint_sha256'], source=source,
        original_checkpoint_sha256=original.manifest['checkpoint_sha256']))
    return case, record


def run_application_acceptance(*, project, root, evidence):
    project, root, evidence = Path(project), Path(root), Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    case = Case(root / 'cases/dingxin-validation/case.json')
    inputs = read_json(root / 'acceptance-input.json')
    digest = inputs['checkpoint_sha256']
    validity = []
    for i, window in enumerate(case.manifest['windows']):
        sample = case.sample(i)
        streams = {}
        for stream in ('physiology', 'vehicle'):
            values = getattr(sample, stream + '_values')
            mask = getattr(sample, stream + '_feature_mask')
            streams[stream] = dict(shape=list(values.shape), valid_values=int(mask.sum()),
                observed_points=int(mask.any(axis=1).sum()), coverage=float(mask.mean()))
        validity.append(dict(window=i + 1, start_s=window['start_s'], duration_s=window['duration_s'],
            archive_sha256=window['sha256'], streams=streams))
    write_json(evidence / 'input-validity.json', validity)
    write_json(evidence / 'mechanisms.json', verify_application_mechanisms(case))
    measurements = []
    service = AcceptanceService(project, root)
    try:
        # Each scale starts a fresh process. Repeats reload the model and do not reuse result chunks.
        for count in (1, 3, 15):
            service.start()
            for repeat in range(1, 4):
                started = time.perf_counter()
                job = service.request('/api/runs', dict(case_id=inputs['record_id'], start=0, stop=count, model_id=digest))
                service.wait(job)
                complete_s = time.perf_counter() - started
                result = service.request('/api/runs/' + job['id'] + '/result')
                result_s = time.perf_counter() - started
                delivery = service.request('/api/runs/' + job['id'] + '/deliver', {})
                delivered_s = time.perf_counter() - started
                assert result['model']['semantic_event_enabled'] and delivery['windows'] == count
                assert result['checkpoint_sha256'] == digest and result['reused_windows'] == 0
                timings = read_json(root / 'runs' / job['id'] / 'timings.json')
                measurements.append(dict(windows=count, repeat=repeat, run_id=job['id'],
                    process_first_task=repeat == 1, service_startup_s=service.startup_s,
                    **timings, http_completed_s=complete_s, http_result_s=result_s,
                    http_delivery_s=delivered_s, windows_per_s=count/timings['pipeline_total_s'],
                    **service.memory()))
            service.stop()
        with (evidence / 'performance.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
            writer.writeheader(); writer.writerows(measurements)
        write_json(evidence / 'performance.json', measurements)
        service.start()
        acceptance = read_json(verify_demo(root, port=service.port))
        write_json(evidence / 'acceptance.json', acceptance)
        write_json(evidence / 'http-errors.json', verify_http_errors(service, case))
        # Kill only this owned process after a real completed chunk and before task completion.
        failed = service.request('/api/runs', dict(case_id=inputs['record_id'], start=0, stop=15, model_id=digest))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = service.request('/api/runs/' + failed['id'])
            if 1 <= state['done'] < 15:
                break
            time.sleep(.01)
        assert 1 <= state['done'] < 15, state
        service.stop(interrupted=True)
        service.start()
        interrupted = service.request('/api/runs/' + failed['id'])
        assert interrupted['status'] == 'failed'
        retried = service.wait(service.request('/api/runs/' + failed['id'] + '/retry', {}))
        assert retried['reused_windows'] >= state['done'] and retried['retry_of'] == failed['id']
        assert service.request('/api/runs/' + failed['id'])['status'] == 'failed'
        write_json(evidence / 'recovery.json', dict(status='passed', interrupted=interrupted, recovered=retried,
            completed_before_kill=state['done'], original_preserved=True))
        environment = dict(checkpoint_sha256=digest, branch=subprocess.check_output(
            ['git','branch','--show-current'], cwd=project, text=True).strip(),
            base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=project, text=True).strip(),
            implementation_digest=read_json(root/'runs'/retried['id']/'result.json')['implementation_digest'],
            system=platform.platform(), python=sys.version, executable=sys.executable, torch=torch.__version__,
            threads=torch.get_num_threads(), device='cpu', cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
            memory_total=Path('/proc/meminfo').read_text().splitlines()[0],
            timing_scope='fresh service per scale; model reloaded per task; warm OS file cache; no result reuse; process HWM includes prior tasks in same scale',
            conditions='supplementary training completed before measurements; no GPU use',
            input=inputs, service_root=str(root.resolve()), observations_sha256=sha256_file(case.root/'case.json'))
        write_json(evidence / 'environment.json', environment)
    finally:
        service.stop()
    return measurements


def verify_http_errors(service, case):
    sample = case.sample(0)
    window = case.manifest['windows'][0]
    original = dict(format='chronaris.observations.v1', model_id=case.manifest['checkpoint_sha256'],
        schema=case.schema.to_dict(), windows=[dict(sample_id=sample.sample_id,
        start_s=window['start_s'], duration_s=sample.context_duration_s,
        **{key:getattr(sample, key).tolist() for key in ARRAYS})])
    imported = service.request('/api/cases/import', original)
    original['record_id'] = imported['id']
    duplicate = service.request('/api/cases/import', original)
    assert len(duplicate['windows']) == 1
    bad_inputs = []
    value = deepcopy(original); value['windows'][0]['physiology_values'][0][0] += 1
    bad_inputs.append(('conflicting observation identity', value))
    value = deepcopy(original); value['schema']['physiology_feature_names'].reverse()
    bad_inputs.append(('field order changed', value))
    value = deepcopy(original); value['windows'][0]['physiology_timestamps_s'][0] = -1
    bad_inputs.append(('time outside window', value))
    value = deepcopy(original); value['windows'][0]['physiology_feature_mask'][0][0] = 1
    bad_inputs.append(('non boolean mask', value))
    value = deepcopy(original); value['windows'][0]['physiology_values'][0][0] = float('nan')
    bad_inputs.append(('non finite input', value))
    results = []
    for name, value in bad_inputs:
        try:
            service.request('/api/cases/import', value)
        except HTTPError as error:
            assert error.code == 400
            results.append(dict(check=name, http_status=error.code, body=json.load(error)))
        else:
            raise AssertionError(name + ' was accepted')
    assert len(next(c for c in service.request('/api/cases') if c['id'] == imported['id'])['windows']) == 1
    return dict(status='passed', duplicate_window_count=1, original_record_preserved=True, checks=results)


def capture_application_browser(*, project, root, evidence):
    project, root, evidence = Path(project), Path(root), Path(evidence)
    service = AcceptanceService(project, root).start()
    try:
        browser_root = project/'artifacts/fusion-workbench/browser'
        env = dict(os.environ, WORKBENCH_URL=service.base, WORKBENCH_DATA_ROOT=str(root.resolve()),
            DEMO_SCREENSHOTS=str(evidence.resolve()),
            LD_LIBRARY_PATH=str(browser_root/'libs/extracted/usr/lib/x86_64-linux-gnu'),
            FONTCONFIG_FILE=str(browser_root/'fonts.conf'),
            PLAYWRIGHT_BROWSERS_PATH=str(browser_root/'browsers'))
        subprocess.run(['node', 'tests/runtime/fusion_workbench_browser.mjs'], cwd=project, env=env, check=True)
        checks = read_json(evidence/'browser-check.json')
        preview = read_json(evidence/'preview-data.json')
        inputs = read_json(root/'acceptance-input.json')
        assert preview['checkpoint_sha256'] == inputs['checkpoint_sha256']
        assert preview['model']['semantic_event_enabled']
        sample = evidence/'feature-example'
        sample.mkdir(exist_ok=True)
        run = root/'runs'/checks['caseRunId']
        for name in ('fusion_stream.npz','representation_manifest.json','features.csv','pooled_features.csv','delivery.json'):
            shutil.copyfile(run/name, sample/name)
        write_json(evidence/'case-result.json', read_json(run/'result.json'))
        write_json(evidence/'case-job.json', read_json(run/'job.json'))
    finally:
        service.stop()
