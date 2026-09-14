"""Chinese tables and bounded conclusions for the stage-4 acceptance artifact."""
from pathlib import Path

DOMAINS = {'clare':'CLARE 生理数据', 'cogpilot':'CogPilot 虚拟飞行数据', 'dingxin':'鼎新真实数据'}
METHODS = {'chronaris':'Chronaris 连续融合', 'physiology_only':'生理单流', 'vehicle_only':'航电／第二输入流单流',
    'naive_time_sync':'朴素时间同步', 'mult':'MulT 多模态变换器', 'contiformer':'ContiFormer 连续时间变换器',
    'timecma':'TimeCMA 语言模型增强', 'chronos2':'Chronos-2 时序预训练', 'sensorllm_deepseek':'SensorLLM 传感器语言对齐（DeepSeek 变体）'}
ROUTES = {'self_supervised':'自监督', 'task_guided':'任务引导', 'summary_adapted':'汇总标签适配',
    'frozen':'外部预训练冻结', 'history_adapted':'历史片段适配'}
TASKS = {'workload_classification':'负荷分类', 'workload_regression':'负荷评分回归', 'difficulty':'飞行难度分类',
    'event_response':'离线事件条件响应', 'maneuver_classification':'机动强度分类',
    'maneuver_regression':'机动强度回归', 'physiology_regression':'生理字段回归'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def render(report, output_root):
    out = Path(output_root)
    text = ['# 阶段 4：完整开发比较、学习检查与成本验收',
        '本轮完成 25 个模型单元、43 条表示路线的开发计算及结果验收。所有方法输出均有非恒定维度，'
        '已有学习与恢复记录通过检查；结论支持进入配置冻结工作，不以 Chronaris 必须胜出作为验收条件。',
        f"实验于 {report['completed_at']} 完成；本次核验时间为 {report['audited_at']}。"
        f"冻结训练源码散列为 `{report['source_code_sha256']}`。首个开发折、种子 17；正式确认保持关闭。",
        '结果判断与局限见 [结果分析](analysis.md)，真实序列下游的全部补充成绩见 [时序补充表](sequence_supplement.md)。',
        '## 结果及证据范围',
        f"[机器验收记录](acceptance.json)绑定 {len(report['sources'])} 个来源文件、"
        f"{report['consumer_routes']} 组独立下游模型及其结果。核验完成回执、原运行来源、检查点、"
        '表示声明、提取代码和近期方法特征文件；核验每种方法使用相同目标和验证窗口、拟合样本仅属于训练角色。'
        '分组指标按“组内字段等权、组间等权”重新汇总。恢复一致性来自本次核验的真实训练／重放收据；本轮不重训或重新生成模型表示。',
        '这些检查支持来源与评价流程完整。训练／验证角色声明及生产过程中的隔离检查不能替代新的模型内部因果审计；'
        '本轮没有重新运行全部时间机制、压力或外层确认。公开验证数据曾用于下游参数选择，因此以下是开发成绩，不是独立留出成绩。',
        table(['数据','训练窗口／视图','验证窗口／视图'], [[DOMAINS[d],v['train'],v['validation']] for d,v in report['sample_counts'].items()]),
        '## 共同线性下游成绩',
        '下表完整保留各方法和实际监督路线。分类指标为宏平均 F1（越高越好），回归指标为均方根误差 RMSE（越低越好）。'
        '鼎新回归按训练尺度归一，公开数据使用原生目标单位，不能跨任务平均为一个总分。'
        '外部预训练、历史片段适配、汇总标签适配不是同监督条件；不据此生成混合总排名。'
        'CLARE 的第二输入流来自公开数据适配，不是真实航电。']
    for domain in DOMAINS:
        rows = [r for r in report['metrics'] if r['domain']==domain and r['family']=='linear']
        tasks = list(dict.fromkeys(x['task'] for x in rows))
        text += ['### '+DOMAINS[domain], '下表按实际路线分别展示完整开发成绩。具体研究判断见 [结果分析](analysis.md)。']
        for route in dict.fromkeys(x['route'] for x in rows):
            methods = list(dict.fromkeys(x['method'] for x in rows if x['route']==route))
            text += ['以下为'+ROUTES[route]+'路线；外部预训练及实际任务监督声明仍分别保留。',
                table(['方法']+[TASKS[t] for t in tasks], [[METHODS[m]]+
                    [f"{next(x['value'] for x in rows if x['method']==m and x['route']==route and x['task']==t):.4f}" for t in tasks]
                    for m in methods])]
    text += ['## 学习与恢复检查',
        '以下逐路线检查更新量、梯度、非恒定表示维度和已有真实恢复证据。原六方法列的是首次与末次验证选择损失；'
        '近期适配方法列的是最初与最后最多 20 次训练损失均值，两种数值不能横向比较。'
        '近期方法遍历的批次不同，训练损失下降也不等于泛化改善。冻结模型和朴素同步不要求产生梯度。',
        table(['数据','方法','路线','更新／最佳','损失起点→末点','非恒定维数'],
            [[DOMAINS[x['domain']],METHODS[x['method']],ROUTES[x['route']],f"{x['updates']} / {x['best_update'] if x['best_update'] is not None else '不适用'}",
              '不适用' if x['initial_loss'] is None else f"{x['initial_loss']:.4g} → {x['final_loss']:.4g}",x['nonconstant_dimensions']] for x in report['learning']]),
        'SensorLLM 仅进行 CLARE 历史对齐和分类适配，回归作为同一表示的下游评价，不标为接受了回归标签的编码器。'
        'TimeCMA 按实际汇总标签适配；Chronos-2 的历史适配使用训练历史内部片段，不新增目标窗口之外的未来信息。',
        '## 历史成本与新运行估算',
        '成本修复保留原尝试收据统计，并补入迁移记录中有检查点证明但缺少尝试收据的训练前缀。已存在父级收据时不重复补入。未完整计时的启动与未落盘尾段仍标为未知，因此历史投入使用下界。原成本文件不回写。',
        f"所有单元已记录与可证补充耗时合计至少 {report['historical_seconds_lower_bound']/3600:.2f} 小时。"
        '这不是整个项目总成本：外部预训练、独立接入测试、完整回归、性能试验及迁移准备另有记录，未混入单元训练时间。',
        '下表同时列出后续排期用的完整单元估算。一般单元采用实测完整执行时间，保留加载、验证、重放和下游拟合开销；'
        '继承单元取原完整运行时间，而不是零更新复用时间。CogPilot/Chronaris 以恢复后的完整单元耗时，加上第 201—300 次'
        '显卡运算图重放的平均训练时间外推前 200 次，具体估算值见下表。它不是已经测过从零训练。'
        'CLARE 和鼎新继续按各自已测执行方式，不借用 CogPilot 的加速倍数。',
        table(['数据','方法','历史投入下界／小时','后续完整单元估算／小时'],
            [[DOMAINS[x['domain']],METHODS[x['method']],f"{x['historical_seconds_lower_bound']/3600:.3f}",f"{x['fresh_unit_seconds_estimate']/3600:.3f}"] for x in report['cost_units']]),
        '参数、表示维数及监督来源见机器记录的学习条目；近期方法语言缓存的首次生成耗时与实际缓存读取耗时分别保留。'
        '原路线训练、表示导出、下游拟合及峰值显存的分项成本见 [原始成本明细]('
        +str(Path(report['source_root'])/'comparison/cost_report.json')+')，其中旧单元尝试总耗时应以本报告的修正账目解释。',
        '## 下一阶段预算情景',
        '本节交付实际单元计时支持的预算，不自动冻结或启动正式矩阵。所有情景保留现有任务、路线和单元更新预算；'
        '公开三折使用既有开发角色，公开五折使用正式确认角色。鼎新均按单记录三个种子描述，不把种子当作三个独立架次。'
        '换折后的窗口数、预算或模型版本变化需要重新估算；下列 30% 仅为排期预留，不是统计置信区间。',
        table(['情景','总单元','仍需运行','仍需路线','串行估算／小时','加 30% 预留／小时'],
            [['公开开发三折×三种子＋鼎新三种子' if x['name'].startswith('development') else '公开确认五折×三种子＋鼎新三种子',
              x['total_units'],x['remaining_units'],x['remaining_routes'],f"{x['serial_hours_estimate']:.1f}",f"{x['hours_with_30_percent_allowance']:.1f}"] for x in report['budget_scenarios']]),
        '开发扩展情景假设当前 25 个单元的源码、数据与协议仍可复用；正式确认情景不复用开发成绩。'
        '两者是分别估算的阶段工作量：若先扩展开发再执行确认，需要相加。仿真、核心消融、压力矩阵及论文整理尚未包含，'
        '阶段 5 必须在正式结果打开前逐项列入冻结计划，不能把这里的主比较时长称为全论文实验总时长。',
        '逐方法预算列于下表；各方法的任务、实际路线、更新预算、种子与耗时明细完整保留在机器记录中。',
        table(['数据','方法','路线数','开发扩展待跑单元','正式确认待跑单元'],
            [[DOMAINS[x['domain']],METHODS[x['method']],len(x['routes']),2 if x['domain']=='dingxin' else 8,
              3 if x['domain']=='dingxin' else 15] for x in report['cost_units']]),
        '## 阶段交付判断',
        '阶段 4 的完整开发计算、共同下游成绩、学习诊断、恢复证据和成本预算已交付。'
        '研究判断见单独的结果分析，不以单折开发成绩作整体优势结论。'
        '下一阶段应冻结含近期方法的模型版本、任务／监督分层及必要分组、多种子、消融和压力范围，再接续正式队列。'
        '工程失败需要修复，效果不足保留，不根据本次成绩追加无界搜索或改变目标。',
        '## 复现',
        '使用仓库约定的 chronaris 环境执行下列入口，输出目录必须尚不存在；入口只读取旧运行并生成新验收产物。',
        '```bash\n/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/evaluation/application_tasks/close_stage4_comparison.py \\\n  --source-root '+report['source_root']+' \\\n  --output-root /mnt/e/chronaris-v4-results/stage4-closeout-replay\n```']
    (out/'report.md').write_text('\n\n'.join(text)+'\n')
    supplementary = ['# 公开数据的时序下游补充',
        'MiniROCKET 时序卷积特征算法只接收真实序列。下表保留全部公开数据补充成绩，不能用它与仅有窗口末端表示的近期方法形成不对称主排名。']
    for route in ('self_supervised','task_guided'):
        supplementary += ['以下为'+ROUTES[route]+'路线的公开数据补充结果。',
            table(['数据','方法','任务','指标','数值'],[[DOMAINS[x['domain']],METHODS[x['method']],
                TASKS[x['task']],x['metric'],f"{x['value']:.4f}"] for x in report['metrics'] if x['family']!='linear' and x['route']==route])]
    (out/'sequence_supplement.md').write_text('\n\n'.join(supplementary).rstrip()+'\n')
