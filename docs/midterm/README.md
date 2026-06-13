# Chronaris 中期报告材料入口

更新时间：2026-06-13

本目录用于承接中期报告写作前的事实冻结、证据分层、风险说明和后续论文检索准备。这里不是新的实验产物目录；实验原始报告、CSV、JSON、PNG 仍以 `docs/artifacts/` 为准。

## 使用顺序

1. 先读 [midterm-fact-sheet-2026-06-13.md](midterm-fact-sheet-2026-06-13.md)：冻结当前可以写入中期报告的事实、指标、图表和源路径。
2. 再读 [boundaries-and-risks-2026-06-13.md](boundaries-and-risks-2026-06-13.md)：明确哪些结论能讲、哪些只能作为边界或风险说明。
3. 写正文或答辩 PPT 前读 [claims-matrix-2026-06-13.md](claims-matrix-2026-06-13.md)：逐条核对论断强度、证据层级和禁止表述。
4. 引用实验材料时回到 [../artifacts/ARTIFACTS.md](../artifacts/ARTIFACTS.md) 和 [../artifacts/stage_i/README.md](../artifacts/stage_i/README.md) 找原始报告与资产。

## 当前中期核心事实

- 论文题目方向：航空人机异构时序数据连续对齐与语义融合。
- 当前私有主线：`chronaris_opt`，只能写成 `private proxy benchmark / proxy evidence`。
- 当前论文任务主线：`risk_proxy / workload_proxy / event_replay_tag`，只能写成 `thesis weak-label evidence`。
- 当前公开支撑线：UAB/NASA public adapter 与 calibration，只能写成 `public adapter / calibration evidence`。
- 当前 runtime：`native aligned` 是真实输入边界；`canonical exact` 是服务层契约化 payload 能力。
- 当前刚体旋转：`translation + vertical` 已启用；`rotation` 因缺少成对角速度字段保持 disabled diagnostics。

## 当前最重要入口

- 当前状态：[../STATE.md](../STATE.md)
- 当前任务队列：[../implementation/TASKS.md](../implementation/TASKS.md)
- 需求规格：[../requirements/SPEC.md](../requirements/SPEC.md)
- 产物索引：[../artifacts/ARTIFACTS.md](../artifacts/ARTIFACTS.md)
- Stage I 报告索引：[../artifacts/stage_i/README.md](../artifacts/stage_i/README.md)
- 中期证据包历史入口：[../artifacts/mid-term/README.md](../artifacts/mid-term/README.md)
