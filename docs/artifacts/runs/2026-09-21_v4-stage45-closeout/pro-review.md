# 给远端 GPT PRO 的复核任务

请基于本分支的真实代码与随仓库提供的证据，判断 Chronaris 在阶段 4.5 后应该如何继续优化。当前目标是提出有依据、有界且可执行的下一轮方案，本次尚未授权新增训练、改变研究门或打开阶段 5。

## 先读什么

1. [完整结果报告](report.md)：五种候选、八个对照尝试、两路线、两类下游、快照诊断及加速尝试。
2. [机器证据](evidence.json)：全部首轮汇总、六个阶段 4 参照、分组混淆矩阵与回归误差、拟合参数、分支及快照诊断、实际梯度组、执行失败和最终通过记录。
3. [验证程序](verify.py)：运行 `python docs/artifacts/runs/2026-09-21_v4-stage45-closeout/verify.py`，只需 Python 标准库。默认复核不需要 CUDA（统一计算设备架构）、数据库或本机绝对路径。
4. [研究合同](../../../requirements/thesis-stage45-development-20260914.md)、[执行修订](../../../requirements/thesis-stage45-execution-v2-20260918.md)、[阶段 4 结果](../2026-09-14_v4-stage4-closeout/analysis.md)。

`evidence.json` 的 `report` 是原最终汇总，`references` 是六个方法与数据域参照，`diagnostics` 对应十个 Chronaris 单元，`probes` 是旧参考的分支探针，`recipes` 是各单元实际配方。`consumer_details` 保留 96 组下游拟合参数及验证分组指标，删除了单窗预测和训练样本列表；`guided_gradient_audit` 是四个保存检查点的补充只读摘录。`sources` 给出原文件 SHA-256 散列及嵌入 JSON 的内容散列，供追溯使用。

训练冻结代码为 `4a09cc3000d36e411046fdf6b648f2258fefbe58`。本分支之后的整理提交只增加材料和更新状态，不改变这轮训练代码、历史检查点或主成绩。CLARE 结果继承早期已完成普通执行单元，CogPilot 剩余单元使用通过资格的加速版本；这种执行来源差异已明确保留。

## 当前已经知道的结论

18 个首轮单元全部完成，五种 Chronaris 候选均未通过原研究门，入围数为零。22 个后续复核与鼎新检查槽位是按规则跳过，不是实际完成了额外训练。新恢复队列运行约 30.07 小时，完整软件回归 697 通过、22 跳过；本轮是固定首折、种子 17 的开发证据，正式确认仍关闭。

Chronaris 是双流连续对齐与融合模型。CLARE 生理负荷数据上的机制组合、较低预训练学习率显著改善回归，却损伤分类；单流保真在 CLARE 两路线均有收益，但损伤 CogPilot 虚拟飞行数据分类。CogPilot 的第二输入流单独使用仍明显优于最终融合分类。温度尝试收益很小，降低微调学习率没有消除跨任务权衡。

快照显示 CLARE 机制组合第 50 次预训练的线性下游成绩明显优于按原训练内目标选出的第 300 次；这应成为选模协议研究的线索，不能事后替换主结果。机制组合同时改变多个组件，不能直接归因于其中某一项。

## 请重点回答的问题

1. **主要瓶颈在哪里？** 按证据强弱区分强分支信息损失、投影与池化、融合门、任务目标冲突、选模目标不一致和训练时长；不要仅依据某个配置参数认定根因。
2. **哪一项最值得先做？** 提出最多三项按优先级排列的改动，每项写清已有证据、相反证据、具体代码落点、要检验的机制和失败时如何终止。优先考虑一个能隔离问题的最小改动，避免同时增加多个结构与损失。
3. **怎样修订选模才不引入新的选择偏差？** 当前验证窗口已被反复用于开发；如使用下游指标选检查点，应明确新的训练内划分、选择规则和后续确认角色。不能只从现有快照逐任务取最大值后声称获胜。
4. **怎样解释分类与回归的权衡？** 使用 `consumer_details` 检查分组混淆矩阵、类别支持数及逐字段误差；判断退化集中于某些分组还是普遍存在。不要将宏平均 F1 当准确率，也不要把窗口当独立受试者。
5. **哪些机制实际上接受了检验？** 区分观测锚定与未校准运动学关系、启用的时移目标与权重为零的独立事件配对。微调摘要的机制项梯度字段缺失被汇总成零，实际顶层梯度组非零；请勿据此误判断梯度。
6. **下一轮的最小矩阵和成本是什么？** 明确复用哪些现有结果、需新增多少单元、基线和对照如何匹配、成功与停止条件。CogPilot 普通配方完整单元实测约 5.13–5.71 小时，复用预训练的微调配方约 2.78 小时；局部约 13 倍加速不能直接套在全部流程上。
7. **如果不继续统一跨域优化，论文上应如何表述？** 可以提出范围清楚的替代研究问题，但须保留本轮不利结果，并说明独立验证所需证据，不能通过降低旧门槛回填为成功。

## 代码阅读定位

下列入口连接配方、实际损失、选模、表示输出与下游评价；先沿真实调用链核对，再提出修改。

| 功能 | 代码入口 |
| --- | --- |
| 五种配方、预算、机制启用 | [stage45_recipe.py](../../../../src/chronaris/evaluation/application_tasks/stage45_recipe.py) |
| 候选门、对照复用、后续复核 | [stage45.py](../../../../src/chronaris/evaluation/application_tasks/stage45.py) |
| 双路线共同执行与表示导出 | [common_downstream_smoke.py](../../../../src/chronaris/evaluation/application_tasks/common_downstream_smoke.py) |
| 预训练选模与保存 | [candidate_screen.py](../../../../src/chronaris/modeling/training/candidate_screen.py) |
| 公共损失及单流保真 | [pretext.py](../../../../src/chronaris/modeling/training/pretext.py)、[candidate_step.py](../../../../src/chronaris/modeling/training/candidate_step.py) |
| 机制损失、日程与有效计数 | [candidate_mechanisms.py](../../../../src/chronaris/modeling/training/candidate_mechanisms.py) |
| 微调损失、选模、梯度组与首次联合更新 | [application_finetuning.py](../../../../src/chronaris/evaluation/application_tasks/application_finetuning.py) |
| 分支探针与快照摘要 | [stage45_diagnostics.py](../../../../src/chronaris/evaluation/application_tasks/stage45_diagnostics.py) |
| 连续表示封装 | [chronaris_continuous.py](../../../../src/chronaris/modeling/fusion_encoders/chronaris_continuous.py) |
| 时间对齐与融合实现 | [alignment](../../../../src/chronaris/models/alignment)、[fusion](../../../../src/chronaris/models/fusion) |
| 共同评价合同与分组汇总 | [common_downstream_contract.py](../../../../src/chronaris/evaluation/application_tasks/common_downstream_contract.py)、[v4_grouped_consumers.py](../../../../src/chronaris/evaluation/application_tasks/v4_grouped_consumers.py) |
| 执行容差、恢复和图递推 | [execution_equivalence.py](../../../../src/chronaris/evaluation/application_tasks/execution_equivalence.py)、[stage45_resume.py](../../../../src/chronaris/evaluation/application_tasks/stage45_resume.py)、[cuda_recurrence.py](../../../../src/chronaris/models/alignment/cuda_recurrence.py) |

## 希望收到的输出

请先给出主判断及证据，再给一个最小实施方案。每项建议标明“直接观察”“基于代码的推断”或“需要实验验证”，指出具体文件与函数，并列出验证与停止条件。若认为应该停止某条优化路线，也请明确说明依据和应保留的论文结论。

后续工作不依赖新增鼎新一手数据、专家真值或额外算力。公开适配和鼎新组件证据各按自身范围解释。原研究门、旧主成绩和全部不利结果保持原记录；任何新研究问题或选模规则须另立前瞻性协议，阶段 5 不能自动打开。
