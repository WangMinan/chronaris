# Chronaris 文档入口

更新时间：2026-09-10

当前在 `codex/thesis-v4-recovery-20260905` 继续研究开发，已包含 `main` 的整理成果。研究主线为“各方法提取融合特征，再分别训练相同类型的下游算法”。优先读取新方案、当前状态和任务；历史协议承担来源追溯职责。写作、术语和代码边界统一遵守仓库根目录的 [AGENTS.md](../AGENTS.md)，此处只维护导航。

## 当前入口

- [STATE.md](STATE.md)：当前实验进度、结论和代码来源。
- [implementation/TASKS.md](implementation/TASKS.md)：当前任务与后续执行顺序。
- [requirements/SPEC.md](requirements/SPEC.md)：毕业论文目标和仓库能力边界。
- [融合表示下游评价与近期模型接入方案](requirements/thesis-downstream-representation-plan-20260909.md)：已确认目标、任务含义、接入范围、开发验收与论文节点。
- [v4 开发合同](requirements/thesis-v4-development-plan.md)：已有六方法实现的预算与数据角色来源；执行优先级服从新方案。
- [artifacts/ARTIFACTS.md](artifacts/ARTIFACTS.md)：报告、指标、图表与来源清单。
- [review/REVIEW.md](review/REVIEW.md)：复核计划、发现、修复和验证。

## 目录职责

| 目录 | 内容与入口 |
| --- | --- |
| `implementation/` | 当前任务与[历史计划笔记](implementation/notes/README.md) |
| `requirements/` | [基础合同](requirements/foundation/)、[模型合同](requirements/model-contracts/)、[原始选题材料](requirements/选题报告与基金申请书/)与[中期考核表](requirements/中期报告/) |
| `artifacts/runs/` | 按日期与任务组织的紧凑证据，由产物索引导航 |
| `artifacts/archive/` | 早期报告与资产，保留历史来源 |
| `artifacts/cleanup/` | [既有清理记录](artifacts/cleanup/20260703-thesis-prep-cleanup.md)，其中已删除资产不视为当前依赖 |
| `midterm/` | [六月至七月中期材料](midterm/README.md)，不替代当前状态 |
| `review/` | 按阶段保存复核与验证结果 |

旧应用任务协议 [downstream-evaluation-spec.md](requirements/downstream-evaluation-spec.md)、[七月简化任务](requirements/simple-downstream-evaluation-v1.md)、[仿真生成器规格](requirements/synthetic-benchmark-spec.md)和 [v3.2.3 冻结协议](requirements/thesis-frozen-paper-evaluation-v3.2.3.md)用于核对对应版本的公式、配置与证据，不直接作为新阶段执行队列。

## 本地与兼容入口

`docs/SECRETS.md` 是被忽略的本地连接信息，不属于远端文档交付。原始数据、稠密表示和检查点位于被忽略的 `artifacts/`。解析或编辑 Word 文档须使用文档技能或插件。

`docs/planning`、`docs/foundation`、`docs/models` 分别链接至 `implementation/notes`、`requirements/foundation`、`requirements/model-contracts`，未复制正文。新引用使用实际目录，已有兼容入口保留。
