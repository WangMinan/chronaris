"""Registered models, validated observation imports and archived-record access."""
from copy import copy
from pathlib import Path
import hashlib
import json
import re
import shutil
import uuid

import numpy as np
import torch

from chronaris.modeling.training import load_common_pretraining_checkpoint
from chronaris.representation import ObservationSchema, DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY
from chronaris.representation.collation import ObservedDualStreamSample
from chronaris.serving.fusion_workbench.cases import Case, ARRAYS, MAX_WINDOWS, package_path, read_json, write_json
from chronaris.serving.fusion_workbench.presentation import display_text
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file, write_deterministic_npz


def inspect_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("training_status") != "completed" or payload.get("method_name") != "chronaris"
            or payload.get("implementation_revision") != "causal_fusion_v4"
            or payload.get("label_used_for_encoder_training")):
        raise ValueError("请选择已完成训练且兼容当前接口的自监督模型")
    return payload


def compatible(schema, payload):
    return all(tuple(payload[s + "_feature_names"]) == getattr(schema, s + "_feature_names")
               for s in ("physiology", "vehicle"))


class Catalog:
    def __init__(self, cases_root, audit_root=None):
        self.root = Path(cases_root)
        self.audit_root = Path(audit_root) if audit_root else None
        self.models = {}
        self._payloads = {}
        self._indexes = {}

    def refresh(self):
        cases = {}
        for path in sorted(self.root.glob("*/case.json")):
            if path.parent.name.startswith('.'):
                continue
            case = Case(path)
            key = case.manifest['id']
            if not isinstance(key,str) or not re.fullmatch(r'[a-z0-9-]{1,64}',key) or key in cases:
                raise ValueError("观测记录标识重复")
            cases[key] = case
            digest = case.manifest['checkpoint_sha256']
            self.models.setdefault(digest, (case.checkpoint, case.schema))
        for path in sorted((self.root.parent / 'models').glob('*/model.json')):
            if path.parent.name.startswith('.'):
                continue
            m = read_json(path)
            checkpoint = package_path(path.parent, m['checkpoint'])
            if sha256_file(checkpoint) != m['checkpoint_sha256']:
                raise ValueError('模型文件完整性校验失败')
            schema = ObservationSchema(**{k: tuple(v) if isinstance(v, list) else v for k, v in m['schema'].items()})
            self.models[m['checkpoint_sha256']] = (checkpoint, schema)
        self.cases = cases
        return cases

    def payload(self, digest):
        path, _ = self.models[digest]
        if sha256_file(path) != digest:
            raise ValueError('模型文件完整性校验失败')
        if digest not in self._payloads:
            self._payloads[digest] = inspect_model(path)
        return self._payloads[digest]

    def model_list(self):
        models = []
        for digest, (_, schema) in self.models.items():
            p = self.payload(digest)
            if not compatible(schema, p):
                raise ValueError('模型登记字段与权重不一致')
            config = p.get('encoder_manifest', {}).get('backbone_config', {})
            semantic = bool(config.get('semantic_event_enabled'))
            models.append({'id': digest, 'label': ('连续对齐与事件语义融合' if semantic else '连续对齐与因果融合') + ' · ' + digest[:8],
                'semantic_event_enabled': semantic, 'training_status': '训练完成', 'training_kind': '自监督学习',
                'seed': p.get('seed'), 'training_windows': len(p.get('fold', {}).get('train_sample_ids', [])),
                'validation_windows': len(p.get('fold', {}).get('validation_sample_ids', [])),
                'compatible_cases': [c.manifest['id'] for c in self.cases.values() if compatible(c.schema, p)],
                'schema': schema.to_dict(), 'checkpoint_sha256': digest})
        return models

    def register_model(self, checkpoint, schema):
        checkpoint = Path(checkpoint)
        payload = inspect_model(checkpoint)
        if not compatible(schema,payload):
            raise ValueError('模型与登记字段不兼容')
        load_common_pretraining_checkpoint(checkpoint,device='cpu')
        digest = sha256_file(checkpoint)
        destination = self.root.parent / 'models' / digest
        if destination.exists():
            raise ValueError('此模型已登记')
        stage = destination.parent / ('.register-' + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        try:
            shutil.copyfile(checkpoint,stage/'encoder.pt')
            if sha256_file(stage/'encoder.pt') != digest:
                raise ValueError('复制期间模型文件发生变化')
            write_json(stage/'model.json',{'checkpoint':'encoder.pt','checkpoint_sha256':digest,
                'schema':schema.to_dict(),'source_checkpoint':str(checkpoint.resolve())})
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        self.refresh()
        return digest

    def bind(self, case, digest):
        if digest not in self.models:
            raise ValueError('所选模型不存在')
        path, schema = self.models[digest]
        if (schema.physiology_feature_names != case.schema.physiology_feature_names
                or schema.vehicle_feature_names != case.schema.vehicle_feature_names):
            raise ValueError('模型与观测字段不兼容')
        bound = copy(case)
        bound.manifest = dict(case.manifest, checkpoint_sha256=digest, model_label='融合模型 · ' + digest[:8])
        bound.checkpoint = path
        return bound

    def import_record(self, payload):
        if not isinstance(payload, dict) or payload.get('format') != 'chronaris.observations.v1':
            raise ValueError('请选择标准观测 JSON 文件')
        digest = payload.get('model_id')
        if digest not in self.models:
            raise ValueError('请选择已登记模型')
        checkpoint, schema = self.models[digest]
        if payload.get('schema') != schema.to_dict():
            raise ValueError('导入字段顺序、角色与模型输入不一致')
        raw = payload.get('windows')
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_WINDOWS:
            raise ValueError('每份观测记录支持 1—512 个窗口')
        labels = payload.get('field_labels', {})
        if not isinstance(labels,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in labels.items()):
            raise ValueError('字段说明须为名称与中文标签的映射')
        old = self.cases.get(payload.get('record_id'))
        if payload.get('record_id') and old is None:
            raise ValueError('追加目标不存在')
        if old and (old.manifest.get('origin') != 'uploaded' or old.schema != schema):
            raise ValueError('请选择字段一致的已导入记录进行追加')
        key = old.manifest['id'] if old else 'record-' + uuid.uuid4().hex
        stage = self.root / ('.import-' + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        try:
            windows = list(old.manifest['windows']) if old else []
            known = {w['sample_id']: w for w in windows}
            files = []
            for w in raw:
                if not isinstance(w,dict) or any(k not in w for k in (*ARRAYS,'start_s','duration_s','sample_id')):
                    raise ValueError('观测窗口缺少必要字段')
                arrays = {}
                for name in ARRAYS:
                    a = np.asarray(w[name])
                    if name.endswith('feature_mask') and a.dtype.kind != 'b':
                        raise ValueError('有效观测掩码必须使用布尔值')
                    arrays[name] = a.astype(bool if name.endswith('feature_mask') else np.float64 if name.endswith('timestamps_s') else np.float32)
                if any(type(w[k]) not in (int,float) for k in ('start_s','duration_s')):
                    raise ValueError('窗口时间须为数值')
                start, duration = float(w['start_s']), float(w['duration_s'])
                if not np.isfinite(start) or start < 0 or not np.isfinite(duration):
                    raise ValueError('窗口时间须为有限非负数')
                sample_id = w['sample_id']
                if (not isinstance(sample_id, str) or not 1 <= len(sample_id) <= 240
                        or sample_id.lstrip().startswith(('=','+','-','@')) or any(ord(c)<32 for c in sample_id)):
                    raise ValueError('窗口标识不合法')
                file = uuid.uuid4().hex + '.npz'
                file_hash = write_deterministic_npz(stage / file, arrays)
                if not (arrays['physiology_feature_mask'].any() or arrays['vehicle_feature_mask'].any()):
                    raise ValueError('窗口内需要至少一个有效观测')
                ObservedDualStreamSample(sample_id, key, schema, source_sample_hash=file_hash,
                    context_duration_s=duration, **arrays)
                prior = known.get(sample_id)
                if prior:
                    if (prior['sha256'], prior['start_s'], prior['duration_s']) != (file_hash, start, duration):
                        raise ValueError('同一窗口标识的内容发生变化，请使用新的记录标识')
                    continue
                row = {'file': file, 'sha256': file_hash, 'sample_id': sample_id, 'group_id': key,
                       'source_sample_hash': file_hash, 'start_s': start, 'duration_s': duration}
                windows.append(row); known[sample_id] = row; files.append(file)
            if len(windows) > MAX_WINDOWS:
                raise ValueError('单份记录最多支持 512 个窗口，请按时段分组导入')
            windows.sort(key=lambda w: (w['start_s'], w['sample_id']))
            if old:
                m = dict(old.manifest, windows=windows)
                for file in files:
                    shutil.move(stage / file, old.root / file)
                write_json(old.root / 'case.json', m)
            else:
                shutil.copyfile(checkpoint, stage / 'encoder.pt')
                title = display_text(payload.get('title') or '导入观测记录')[:100]
                m = {'format': 'chronaris.observations.v1', 'origin': 'uploaded', 'id': key, 'title': title,
                     'description': '', 'source_label': display_text(payload.get('source_label','导入观测'))[:100], 'model_label': '融合模型 · ' + digest[:8],
                     'checkpoint': 'encoder.pt', 'checkpoint_sha256': digest, 'schema': schema.to_dict(),
                     'field_labels': {str(k): display_text(v)[:120] for k,v in labels.items()}, 'windows': windows}
                write_json(stage / 'case.json', m)
                Case(stage / 'case.json')
                stage.rename(self.root / key)
            self.refresh()
            return self.cases[key].summary()
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def sources(self):
        if not self.audit_root or not (self.audit_root / 'context_sample_manifest.jsonl').exists():
            return []
        sources = []
        seen = set()
        contexts = [json.loads(line) for line in (self.audit_root / 'context_sample_manifest.jsonl').read_text().splitlines()]
        for case in self.cases.values():
            source = case.manifest.get('source', {})
            snapshot = source.get('snapshot')
            if not snapshot or not Path(snapshot).is_dir() or snapshot in seen:
                continue
            seen.add(snapshot)
            p = self.payload(case.manifest['checkpoint_sha256'])
            fold = p.get('fold', {})
            allowed = set(fold.get('train_sample_ids', [])) | set(fold.get('validation_sample_ids', []))
            views = {}
            for c in contexts:
                if c['context_id'] in allowed:
                    views.setdefault(c['view_id'], []).append(c)
            for n, (view, rows) in enumerate(sorted(views.items()), 1):
                source_id = hashlib.sha256((snapshot + view).encode()).hexdigest()[:24]
                rows.sort(key=lambda w:w['start_offset_ms'])
                sources.append({'id': source_id, 'title': f'实测数据集 · 观测记录 {n:02}',
                    'case_id': case.manifest['id'], 'snapshot': snapshot, 'rows': rows,
                    'start_s': rows[0]['start_offset_ms']/1000, 'stop_s': rows[-1]['end_offset_ms']/1000,
                    'windows': len(rows)})
        return sources

    def prepare_source(self, payload):
        source = next((s for s in self.sources() if s['id'] == payload.get('source_id')), None)
        if source is None:
            raise ValueError('归档记录不存在')
        start, stop = float(payload['start_s']), float(payload['stop_s'])
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError('请选择有效的时间范围')
        rows = [r for r in source['rows'] if r['start_offset_ms']/1000 >= start and r['end_offset_ms']/1000 <= stop]
        if not 1 <= len(rows) <= MAX_WINDOWS:
            raise ValueError('所选范围需要包含 1—512 个完整窗口')
        from chronaris.evaluation.application_tasks.dingxin_context_data import build_dingxin_lazy_context_index
        snapshot = source['snapshot']
        if snapshot not in self._indexes:
            self._indexes[snapshot] = build_dingxin_lazy_context_index(snapshot_root=snapshot,
                field_role_manifest_path=self.audit_root / 'field_role_manifest.csv',
                context_manifest_path=self.audit_root / 'context_sample_manifest.jsonl',
                maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY)
        index = self._indexes[snapshot]
        case = self.cases[source['case_id']]
        windows = []
        for row in rows:
            sample = index.load_sample(row['context_id'])
            windows.append({'sample_id':sample.sample_id, 'start_s':row['start_offset_ms']/1000,
                'duration_s':sample.context_duration_s, **{k:getattr(sample,k).tolist() for k in ARRAYS}})
        return self.import_record({'format':'chronaris.observations.v1', 'title':f"{source['title']} · {start:g}—{stop:g} 秒",
            'source_label':'实测数据集 · 归档记录', 'schema':index.plan.schema.to_dict(), 'model_id':payload.get('model_id',case.manifest['checkpoint_sha256']),
            'windows':windows, 'field_labels':case.manifest.get('field_labels',{})})
