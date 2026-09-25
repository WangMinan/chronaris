# 给远端 GPT PRO：阶段 5 停机后的分支整体复核

请检查本分支“航空人机异构时序数据连续对齐与语义融合”的实验与代码，提出完成阶段 5 的最小、有界、可验证方案。用户已于 9 月 23 日授权进入阶段 5，结束追加模型优化；旧 9 月 21 日任务说明中“阶段 5 未授权”的口径是历史记录。本次请先给诊断和实施建议，不假定已有新训练、修复或外层确认发生。

## 阅读顺序

1. [分支实验总览](report.md)：历史基线、阶段 1—5、全部主要不利结果及未完成工作。
2. [阶段 5 首折三种子结果](stage5_results.json)：12 个完整单元（四个继承、八个新增），48 条主指标，主/补充下游分组指标及拟合参数。
3. [失败单元](failed_unit.json)、[失败堆栈](failure_traceback.txt)、[方差诊断](variance_diagnosis.json)：第二折首个单元的完整预训练、部分线性结果和筛选逻辑。
4. [队列终态](pipeline_state_snapshot.json)、[来源核验](source_inventory.json)、[阶段 5 协议](../../../requirements/thesis-stage5-execution-20260923.md)。
5. 对照[阶段 4 分析](../2026-09-14_v4-stage4-closeout/analysis.md)、[阶段 4.5 全部尝试](../2026-09-21_v4-stage45-closeout/report.md)、[阶段 4.5-B/C 复盘](../2026-09-22_v4-stage45c/report.md)。

先运行本目录 `verify.py`，仅需标准 Python。验证范围是随包文件完整性、逐种子均值/标准差、分组指标汇总及通道筛选决定；源训练张量和大检查点未入包。来源散列供服务器复核，不等于远端已经独立重跑训练。

## 优先回答的问题

1. **筛选失败的直接原因与上游原因各是什么？** 当前筛选要求同一通道在每个有效窗口都具有时间变化；36 个部分恒定窗口导致通道失败集合覆盖全部 64 维。请区分已知代码行为与尚需验证的缺失模式、掩码和表示问题，不把局部时间恒定直接等同于整体塌缩。
2. **最小且公平的修复在哪里？** 检查“所有窗口都要有变化”是否为当前 MiniROCKET 库的必需条件，或仅为仓库包装器约束。比较维持主线性下游、补充下游不可计算的显式状态与必要算法修订。任何改变均须对所有方法一致处理，保留原失败，不通过调低阈值、换指标或静默跳过来过关。
3. **如何保留已有计算？** 给出 12 个完整单元、失败单元的 300 次预训练、表示和部分线性下游分别可复用或需重算的条件；明确源码/输入指纹、选模、结果谱系、恢复与回滚。不要笼统要求全部重训，也不要绕过来源检查。
4. **阶段 5 采用的原参考能支撑哪些论文论断？** 原参考没有采用失败的机制组合。区分公开融合表示的任务价值，与必须由相应仿真、物理和时间机制配置支撑的主张；首折三种子也不能代替跨划分或独立确认。
5. **怎样尽快完成阶段 5？** 给出修复、恢复、正式接口补齐、预算/模型冻结、主表与核心消融/压力的依赖顺序及最小验收。正式方法必须保留近期模型；279/477 是范围计划，旧六方法入口不能冒称覆盖全部方法。尽量复用现有执行器，不再开启没有边界的阶段 4.5 搜索。

## 代码入口

| 要检查的问题 | 文件与函数 |
| --- | --- |
| 零通道报错与数据转置 | [application_consumers.py](../../../../src/chronaris/evaluation/application_tasks/application_consumers.py)：`MiniRocketFrozenConsumer.fit_features`、`_as_collection` |
| 下游拟合、有效窗口、主与补充分派 | [v4_grouped_consumers.py](../../../../src/chronaris/evaluation/application_tasks/v4_grouped_consumers.py)：`fit_native_consumers`、`run_native_method_consumers` |
| 先自监督导出、后任务引导的调用顺序 | [common_downstream_smoke.py](../../../../src/chronaris/evaluation/application_tasks/common_downstream_smoke.py)：`run_common_contract_smoke` |
| 数据角色、监督与消费者约束 | [common_downstream_contract.py](../../../../src/chronaris/evaluation/application_tasks/common_downstream_contract.py)：`build_common_contract`、`run_common_downstream` |
| 本批范围、继承与最终汇总 | [stage5.py](../../../../src/chronaris/evaluation/application_tasks/stage5.py)：`units`、`plan`、`run_stage5`、`progress_report` |
| 失败停止、收据、显式重试 | [v4_pipeline.py](../../../../src/chronaris/evaluation/application_tasks/v4_pipeline.py)：`execute_pipeline` |
| 原参考与机制组合的实际开关 | [stage45_recipe.py](../../../../src/chronaris/evaluation/application_tasks/stage45_recipe.py)：`training_recipe` |
| 尚待接续的正式入口 | [v4_configuration_freeze.py](../../../../src/chronaris/evaluation/application_tasks/v4_configuration_freeze.py)、[v4_native_confirmation_cohort.py](../../../../src/chronaris/evaluation/application_tasks/v4_native_confirmation_cohort.py)、[recent_model_smoke.py](../../../../src/chronaris/evaluation/application_tasks/recent_model_smoke.py) |

请按“直接观察／代码推断／待验证假设”组织结论，列出最多三项优先动作及其文件、检查项、计算复用和失败停止条件。保留所有不利结果，不更换冻结主成绩，不依赖新增鼎新数据、专家标签或额外算力。当前不要求再证明模型必胜，而要求对已有方法作完整、公平、可复现的研究评价。
