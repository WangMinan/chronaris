"""Audit existing measured field metadata without inferring missing physical units."""
import csv
from pathlib import Path
import re
import torch

from chronaris.representation import DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY, build_dingxin_observation_schema_plan
from chronaris.serving.fusion_workbench.cases import read_json, write_json
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def audit_observed_fields(output):
    audit=Path('docs/artifacts/runs/2026-07-10_fixed-data-audit')
    snapshot=Path('/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot')
    plan=build_dingxin_observation_schema_plan(snapshot_root=snapshot,
        field_role_manifest_path=audit/'field_role_manifest.csv',
        maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY)
    source=Path('artifacts/chapter6-final/service/cases/dingxin-validation/encoder.pt')
    payload=torch.load(source,map_location='cpu',weights_only=True)
    assert tuple(payload['vehicle_feature_names'])==plan.schema.vehicle_feature_names
    retained=payload['fold']['train_sample_ids'][0].split('__pilot')[0]
    role_rows=list(csv.DictReader((audit/'field_role_manifest.csv').open()))
    selected=[]
    for r in role_rows:
        label=r['display_label']
        if r['sortie_id']!=retained or not re.search('载机|本机',label):continue
        if not re.search('速度|加速度|高度|俯仰|横滚|航向|角速',label):continue
        raw=r['measurement']+'.'+r['source_field']
        position=plan.vehicle_raw_to_index[retained].get(raw)
        selected.append(dict(raw_field=raw,canonical_field=plan.schema.vehicle_feature_names[position] if position is not None else None,
            label=label,unit_hint=r['unit_hint'],semantic_key=r['semantic_key'],
            observed_points=int(r['observed_point_count']),metadata_status=r['metadata_status'],
            training_valid_count=payload['normalizer']['vehicle']['valid_count'][position] if position is not None else None,
            training_active=payload['normalizer']['vehicle']['active_mask'][position] if position is not None else None))
    own_rate=[r for r in selected if re.search('角速度|角速率|滚转速率|俯仰速率|偏航速率',r['label'])]
    context=Path('docs/artifacts/runs/2026-06-14_llm-preprocessing-context/llm_preprocessing_context.json')
    llm=read_json(context)
    annotations=llm.get('field_semantic_dictionary',[])
    hits=[r for r in annotations if any(s['raw_field'] in str(r) or (s['canonical_field'] and s['canonical_field'] in str(r)) for s in selected)]
    rows=[dict(relation='速度—对应加速度',status='pair_present_but_units_and_acceleration_definition_unverified',
              detail='同名北向/西向/天向平台系候选存在；字典单位为泛称速度/加速度，缺少量纲换算、比力与惯性加速度及平台系变化定义，不能直接等同速度标量—纵向加速度。'),
          dict(relation='高度—垂直速度',status='pair_present_but_units_and_vertical_reference_unverified',
              detail='海拔/气压/卫星高度及天向速度存在；高度单位为泛称，尚无配套高度基准和速度标度的确证。'),
          *[dict(relation=name+'角—对应角变化率',status='measured_derivative_field_missing',
                 detail='角度标为毫弧度，可确定角度换算；现有已解析载机字段中未找到对应角变化率，目标角速率方差/协方差不属于载机姿态导数。') for name in ('滚转','俯仰','偏航')]]
    assert not own_rate, 'new own-aircraft derivative metadata needs review'
    report=dict(status='audited_not_sufficient_for_five_relations',checkpoint_sha256=sha256_file(source),
        retained_record=retained,fields=selected,relations=rows,matched_language_annotations=hits,
        metadata_source='vehicle_metadata_source.csv: analysis_detail/access_rule_detail',
        source_hashes={str(p):sha256_file(p) for p in (audit/'field_role_manifest.csv',audit/'vehicle_metadata_source.csv',context,snapshot/'snapshot_manifest.json')},
        decision='retain measured semantic model with zero verified kinematic relations; use native simulation full model for five-relation application evidence; no guessed mapping or synthetic derivative')
    write_json(Path(output)/'measured-field-audit.json',report)
    return rows
