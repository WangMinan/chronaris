# Chronaris Review 入口

更新时间：2026-06-07

本目录用于保存 code review 相关产出，包括 review 计划、发现、修复记录和阶段性检查结果。

## 组织规则

- 父入口：本文档。
- 阶段目录：[stage/](stage/)。
- 子目录命名与 [../implementation/PLAN.md](../implementation/PLAN.md) 的阶段一致。
- 每次 review 应说明：
  - review 范围
  - 关键发现
  - 修复状态
  - 剩余风险
  - 已跑测试

## 当前建议

当前最值得 review 的范围是 Stage I thesis mainline Phase C 的未提交工作区，尤其是：

- task heads
- `L_task + L_causal` objective contract
- thesis weak-label task builder
- multitask train pipeline
- private benchmark `proxy_evidence / thesis_task_evidence` 分层
