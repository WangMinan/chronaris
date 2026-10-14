"""Read existing language-model annotations and retain human review decisions."""
import hashlib
from datetime import datetime, timezone

from chronaris.llm.schemas import coerce_field_semantics, coerce_weak_label_review
from chronaris.serving.fusion_workbench.cases import read_json, write_json
from chronaris.serving.fusion_workbench.presentation import display_text

ROLES = {'physiological_load':'生理负荷相关观测', 'risk_signal':'状态变化相关观测'}
TASKS = {'event_replay_tag':'飞行事件复盘规则', 'risk_proxy':'状态变化弱监督规则', 'workload_proxy':'生理负荷弱监督规则'}


class FieldReviews:
    def __init__(self, path, context_path=None):
        self.path = path
        self.context = read_json(context_path) if context_path and context_path.exists() else {}
        self.fields = coerce_field_semantics({'rows':self.context.get('field_semantic_dictionary',[])})
        self.rules = coerce_weak_label_review(self.context)
        self.source_hash = hashlib.sha256(str(self.context).encode()).hexdigest()

    def rows(self, case):
        saved = read_json(self.path) if self.path.exists() else {}
        applicable = case.schema.source_kind == 'dingxin_fixed_snapshot'
        dictionary = {r['feature_name']:r for r in self.fields} if applicable else {}
        labels = case.manifest.get('field_labels', {})
        rows = []
        for stream, name_cn in (('physiology','生理'),('vehicle','航电')):
            for i, name in enumerate(getattr(case.schema,stream+'_feature_names')):
                suggestion = dictionary.get(name, dictionary.get(name.removeprefix(stream+'.')))
                key = hashlib.sha256((case.schema.schema_sha256+name+self.source_hash).encode()).hexdigest()
                rows.append({'id':key, 'label':display_text(labels.get(name, f'{name_cn}字段 {i+1:03}')),
                    'stream':name_cn, 'role':ROLES.get(suggestion['llm_semantic_role'],'观测字段') if suggestion else '观测字段',
                    'unit':(suggestion.get('llm_unit_guess') or '待核定') if suggestion else '待核定',
                    'confidence':suggestion['confidence'] if suggestion else None,
                    'source':'语言模型解释记录' if suggestion else '输入字段定义',
                    'status':saved.get(key,{}).get('decision','pending')})
        for r in (self.rules if applicable else []):
            if r['task_name'] not in TASKS:
                continue
            key = hashlib.sha256((case.schema.schema_sha256+r['task_name']+self.source_hash).encode()).hexdigest()
            rows.append({'id':key,'label':TASKS[r['task_name']], 'stream':'规则',
                'role':'保留现有规则' if r['review_decision']=='keep_current_rule' else '核定规则',
                'unit':'—','confidence':r['confidence'],'source':'语言模型规则复核记录',
                'status':saved.get(key,{}).get('decision','pending')})
        return rows

    def review(self, case, payload):
        ids = payload.get('ids')
        if not isinstance(ids,list) or not ids or len(ids)>100:
            raise ValueError('每次请选择 1—100 项复核条目')
        valid = {r['id'] for r in self.rows(case)}
        if any(not isinstance(i,str) or i not in valid for i in ids) or payload.get('decision') not in ('approved','needs_revision'):
            raise ValueError('复核条目或结论不合法')
        saved = read_json(self.path) if self.path.exists() else {}
        for key in ids:
            saved[key] = {'decision':payload['decision'],'reviewed_at':datetime.now(timezone.utc).isoformat()}
        write_json(self.path,saved)
        return {'reviewed':len(ids)}
