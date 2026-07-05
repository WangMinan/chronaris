# Chronaris Review 入口

更新时间：2026-06-07

本目录用于保存 code review 相关产出，包括 review 计划、发现、修复记录和阶段性检查结果。

## 组织规则

- 父入口：本文档。
- 阶段目录：[stage/](stage/)。
- 子目录命名与 [../implementation/TASKS.md](../implementation/TASKS.md) 的阶段一致。
- 每次 review 应说明：
  - review 范围
  - 关键发现
  - 修复状态
  - 剩余风险
  - 已跑测试

## 当前建议

当前最值得 review 的范围已经切到中期前 `P10-P15` 主动任务队列，尤其是：

- `P10 evidence runner`：manifest、`evidence_layer`、失败保留 partial manifest、`--reuse-existing / --only` 行为。
- `P11 thesis weak-label multitask sweep`：小网格是否有边界，是否保持 `risk_proxy / workload_proxy / event_replay_tag` 的 weak-label 表述。
- `P12 chronaris_opt component ablation`：分类任务、回归任务和检索任务 是否始终写成 Dingxin weak-label benchmark，不越界成人工真值。
- `P13/P14 public adapter calibration / transfer boundary`：UAB/NASA 是否保持 public adapter / calibration evidence 边界。
- `P15 rigid_body rotation audit`：字段启用或缺失诊断是否可复现，报告是否避免把缺失项写成已验证约束。
- 所有新增产物是否能从 `docs/artifacts/ARTIFACTS.md` 或 `docs/implementation/TASKS.md` 追溯。
