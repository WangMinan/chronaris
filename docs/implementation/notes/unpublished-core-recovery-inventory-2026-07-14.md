# 未推送核心任务恢复工作清单（2026-07-14）

## 结论

本轮从 `main@23b968e0e9e2b33080fb87781622f325715df9e7` 独立开展，不整体合并本机未推送实验。现有未推送分支和本地工件均保留原位；本轮只把其中可复核的失败结论作为协议风险输入，必要时再逐项复用通过测试的通用工具。

## 工作区与分支

- 原工作树：`/home/wangminan/projects/chronaris`，`main` 工作树干净，HEAD 与 `origin/main` 均为 `23b968e`。
- 本轮独立工作树：`/home/wangminan/projects/chronaris-dingxin-core-feasibility-20260714`。
- 本轮分支：`codex/dingxin-core-feasibility-20260714`，从 `23b968e` 创建。
- 已推送但不整体合并的结构候选：`codex/chronaris-v2-mainline-20260712@97fdb43`。
- 未推送核心任务恢复分支：`codex/chronaris-core-task-recovery-20260713@aaf53c0`。

## 未推送核心任务恢复分支

`aaf53c0` 相对 `main` 新增或修改 53 个文件，约 7488 行。主要内容包括任务审计、观测残差、候选开发、配置锁定、一次性确认和外层支持重叠审计。

该分支的开发报告曾记录机动分类 Macro-F1 `0.817460`、生理响应 RMSE `0.826802`、高响应 AUPRC `0.733498`，不满足本轮预声明的上限或安全融合门槛。其一次性外层确认随后发现：

- `leave_one_view_out__fold01`：31/31 个训练上下文与外层留出上下文存在完整时间支持重叠；
- `leave_one_view_out__fold02`：31/31 个训练上下文与外层留出上下文存在完整时间支持重叠；
- `leave_one_view_out__fold03`：0/38 个训练上下文与外层留出上下文重叠。

因此该分支将外层汇总标记为 `protocol_valid=false`、`promotion_passed=false`，没有修改历史确认指标，也没有启动仿真次级确认。这个结论作为本轮 fail-closed 和支持区间审计的直接依据；该分支的模型与结果不作为本轮候选起点。

## 本地重型工件

以下工件保留在原工作树的被忽略目录，不提交 Git：

- `artifacts/application_evaluation/2026-07-13_chronaris-core-task-recovery-development/`，约 123 MB；
- `artifacts/application_evaluation/2026-07-13_chronaris-core-task-recovery-smoke/`，约 4.4 MB；
- Chronaris v2 的锁定训练、表示、consumer、仿真、公开数据适配和证据包根；
- `artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot/`，作为本轮只读鼎新原始点输入；
- `artifacts/application_evaluation/2026-07-11_dingxin-nested-targets/`、`2026-07-11_dingxin-nested-validation/` 与 `2026-07-12_dingxin-locked-representations-coalesced/`，仅作为已冻结输入或历史消费者谱系读取。

## 本轮复用边界

- 不 cherry-pick 或 merge `aaf53c0`。
- 不读取该分支的一次性外层预测来选择本轮候选。
- 允许复核其“支持区间重叠必须 fail closed”的结论。
- 若复用通用思路，必须在本轮重新实现、补充测试并绑定新协议 SHA。
- 本轮候选开发只允许 `inner-train` 和 `inner-validation`；`held_out/outer-test` 请求必须失败并进入访问审计。
