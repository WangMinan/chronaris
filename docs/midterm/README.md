# Chronaris 中期报告材料入口

更新时间：2026-06-19

本目录用于承接中期报告写作前的事实冻结、证据分层、风险说明和后续论文检索准备。这里不是新的实验产物目录；实验原始报告、CSV、JSON、PNG 仍以 `docs/artifacts/` 为准。

## 使用顺序

1. 先读 [midterm-fact-sheet-2026-06-13.md](midterm-fact-sheet-2026-06-13.md)：冻结当前可以写入中期报告的事实、指标、图表和源路径。
2. 再读 [boundaries-and-risks-2026-06-13.md](boundaries-and-risks-2026-06-13.md)：明确哪些结论能讲、哪些只能作为边界或风险说明。
3. 写正文或答辩 PPT 前读 [claims-matrix-2026-06-13.md](claims-matrix-2026-06-13.md)：逐条核对论断强度、证据层级和禁止表述。
4. 引用实验材料时回到 [../artifacts/ARTIFACTS.md](../artifacts/ARTIFACTS.md) 和 [../artifacts/stage_i/README.md](../artifacts/stage_i/README.md) 找原始报告与资产；当前中期图表优先使用 [../artifacts/stage_i/stage-i-thesis-materials-20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh.md](../artifacts/stage_i/stage-i-thesis-materials-20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh.md)。
5. 讨论中期 LLM 内容时读 [../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md](../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md)、[../artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md](../artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md) 和 [llm-preprocessing-comparison-summary-2026-06-14.md](llm-preprocessing-comparison-summary-2026-06-14.md)，当前可写成 DeepSeek 在线 preprocessing context 已完成小样本真实切片 run 与 A0-A4 对比实验，但不能写成人工真值或核心因果证据。
6. 追溯 P21 设计时读 [llm-preprocessing-comparison-plan-2026-06-14.md](llm-preprocessing-comparison-plan-2026-06-14.md)；引用结论时以 [llm-preprocessing-comparison-summary-2026-06-14.md](llm-preprocessing-comparison-summary-2026-06-14.md) 为准。

## 当前中期核心事实

- 论文题目方向：航空人机异构时序数据连续对齐与语义融合。
- 当前私有主线：`chronaris_opt`，只能写成 `private proxy benchmark / proxy evidence`。
- 当前论文任务主线：`risk_proxy / workload_proxy / event_replay_tag`，只能写成 `thesis weak-label evidence`。
- 当前公开支撑线：UAB/NASA public adapter 与 calibration，只能写成 `public adapter / calibration evidence`。
- 当前 runtime：`native aligned` 是真实输入边界；`canonical exact` 是服务层契约化 payload 能力。
- 当前刚体旋转：`translation + vertical` 已启用；`rotation` 因缺少成对角速度字段保持 disabled diagnostics。
- 当前 LLM 实现：P20 已完成 DeepSeek 在线时序数据预处理小样本真实切片 run，用于字段语义归一、weak-label 复核、schema gap policy 和 runtime 解释；输出是 preprocessing context，不是人工真值。
- 当前 LLM 对比：P21 已完成 A0-A4 对比实验，结果为 `333/333` task entries attach 后 `label_unchanged=true`、semantic query coverage `3 -> 7`、runtime explanation 子集 `4/12`、human review packet `15` 条且 `human_review_completed=false`。
- 当前中期图表：r5 leakage-safe refresh 已输出 `11` 张 PNG 与 `11` 张 CSV，替换 evidence layer all-one bar、runtime schema、runtime semantic case、rotation audit、weak-label sweep、private ablation、public transfer、semantic fusion 和 LLM review flow 的旧表达，并新增 model backbone / task adapter 防泄漏消融图。

## 当前最重要入口

- 当前状态：[../STATE.md](../STATE.md)
- 当前任务队列：[../implementation/TASKS.md](../implementation/TASKS.md)
- 需求规格：[../requirements/SPEC.md](../requirements/SPEC.md)
- 产物索引：[../artifacts/ARTIFACTS.md](../artifacts/ARTIFACTS.md)
- Stage I 报告索引：[../artifacts/stage_i/README.md](../artifacts/stage_i/README.md)
- 中期证据包历史入口：[../artifacts/mid-term/README.md](../artifacts/mid-term/README.md)
- P20 DeepSeek 在线时序数据预处理计划：[../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md](../implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md)
- P20 DeepSeek 在线时序数据预处理 run：[../artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md](../artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md)
- P21 LLM 预处理对比实验计划：[llm-preprocessing-comparison-plan-2026-06-14.md](llm-preprocessing-comparison-plan-2026-06-14.md)
- P21 LLM 预处理对比实验摘要：[llm-preprocessing-comparison-summary-2026-06-14.md](llm-preprocessing-comparison-summary-2026-06-14.md)
- P21 LLM 预处理工程报告：[../artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md](../artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md)
- 当前中期图表 r5：[../artifacts/stage_i/stage-i-thesis-materials-20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh.md](../artifacts/stage_i/stage-i-thesis-materials-20260619T-stage-i-thesis-materials-r5-leakage-safe-refresh.md)
- P21 执行 prompt（历史追溯）：[../implementation/notes/goal-prompt-stage-i-p21-llm-comparison-2026-06-14.md](../implementation/notes/goal-prompt-stage-i-p21-llm-comparison-2026-06-14.md)
