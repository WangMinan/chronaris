"""Write a result-led handoff only after independent native forecast verification."""
import json
from pathlib import Path
import pandas as pd
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json


def delivery_text(run,result,check,summary):
    assert run['status']==result['status']=='completed' and check['status']=='passed'
    assert run['completed_models']==check['models']==4 and run['updates']==check['updates']
    conclusion=('联合输入通过本轮开发性能和配对门。' if result['paired_routes'] else
                '本轮完成了原生预测大模型对照，联合输入尚未同时通过开发性能和配对门。')
    lines=['# 原生时序预测对照结果','',conclusion+' Chronos-2（预训练时序预测模型）直接预测未来五秒飞行误差；与此前提取高维表示再接岭回归的路线分别保留。',
           '',f'四个适配模型完成{run["updates"]}次更新，累计主任务拟合窗口暴露{run["windows_seen"]}次。两折使用19/20名拟合人员，18387个合格窗口，种子17；飞机与联合输入均显式获得相同历史绩效和过去五秒均值。所有模型锁定后才评分，未扩展其他折或种子。',
           '', '下表展示逐人等权均方根误差。每折同时列出原生零样本、原生监督适配和既有最强参照，用于判断增加生理输入是否提供稳定增量；不能将选取不同路线的两折最小值拼成一个方法。',
           '', '| 路线 | 第一折 | 第二折 |','|---|---:|---:|']
    labels={'native_frozen_aircraft':'原生零样本飞机','native_frozen_joint':'原生零样本联合',
            'native_adapted_aircraft':'原生适配飞机','native_adapted_joint':'原生适配联合'}
    for method,label in labels.items():
        values=[float(summary[(summary.fold==fold)&(summary.method==method)].rmse.iloc[0]) for fold in ('fold01','fold02')]
        lines.append(f'| {label} | {values[0]:.6f} | {values[1]:.6f} |')
    bests=[]
    for fold in ('fold01','fold02'):
        panel=summary[(summary.fold==fold)&summary.method.str.startswith('reference_')]
        bests.append(panel.loc[panel.rmse.idxmin()])
    lines.append(f'| 各折既有最强参照 | {bests[0].rmse:.6f} | {bests[1].rmse:.6f} |')
    lines+=['','最强参照的机器索引分别为 `'+bests[0].method+'` 与 `'+bests[1].method+'`，来源及全量成绩见 `reference_reports.json`、`summary.csv`。飞前静息校准使用额外无标签观测条件，仅单列为上下文参照，不进入上述同条件主门。',
            '', '下表披露实际训练轮数及选择轮数；选择第0轮表示保持外部预训练权重，不能把已发生的训练量隐去。',
            '', '| 划分及输入 | 完成轮次 | 选择轮次 | 更新次数 |','|---|---:|---:|---:|']
    for u in run['units']:
        label=('第一折' if u['fold']=='fold01' else '第二折')+('飞机' if u['view']=='aircraft' else '联合')
        lines.append(f'| {label} | {u["epochs"]} | {u["best_epoch"]} | {u["updates"]} |')
    lines+=['',f'独立核验通过：{check["protected_files"]}份保护文件未变，{check["predictions"]}条主预测及{check["subject_metrics"]}项逐人指标复算通过；最大指标差异{check["maximum_metric_delta"]:.3g}，新鲜预测重放最大差异{max(check["replay"].values()):.3g}。核验不执行优化器更新。模型、最后可恢复状态及日志见 `asset_manifest.json`；输入合同、真实权重资格、选择记录和逐批训练记录均独立保留。',
            '', '当前结论只覆盖重复暴露的两折开发比较。人员扩展与旧五折确认划分的交叉见 `../eligible-fit-20261019/EXPOSURE.md`；正式确认需重新冻结协议。预训练历史与监督计算量不同均需披露。新增大模型对照推进了导师的比较要求，但尚不能单凭本单元宣布本文方法已胜过全部传统方法或达到完整论文验收。','']
    return '\n'.join(lines)


def deliver_chronos_forecast(report,assets):
    report,assets=Path(report),Path(assets)
    run=json.loads((report/'run_status.json').read_text());result=json.loads((report/'performance_status.json').read_text())
    check=json.loads((report/'verification.json').read_text());summary=pd.read_csv(report/'summary.csv')
    text=delivery_text(run,result,check,summary)
    save_json(report/'asset_manifest.json',dict(root=str(assets.resolve()),files={str(p.relative_to(assets)):file_digest(p) for p in sorted(assets.rglob('*')) if p.is_file()}))
    (report/'README.md').write_text(text,encoding='utf-8')
    status='通过两折开发性能及配对门' if result['paired_routes'] else '未通过两折稳定生理增量门'
    for path,base in [('docs/STATE.md','review/stage/'),('docs/implementation/TASKS.md','../review/stage/')]:
        p=Path(path);body=p.read_text();pos=body.index('\n\n',body.index('更新时间：'))+2
        entry=f'**夜间原生时序预测对照已完成并独立重放，{status}。** 四个适配模型、{run["updates"]}次更新及{check["predictions"]}条主预测完成核验，原冻结来源保持；本轮没有扩展完整矩阵。详见[结果与资产]({base}chronos-forecast-20261019/README.md)，正式确认仍须按[人员暴露核验]({base}eligible-fit-20261019/EXPOSURE.md)重新冻结。\n\n'
        p.write_text(body[:pos]+entry+body[pos:])
    save_json(report/'delivery.json',dict(status='completed',verification_sha256=file_digest(report/'verification.json'),
                                         readme_sha256=file_digest(report/'README.md'),updates=run['updates']))
