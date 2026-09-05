# Chronaris 文档入口

更新时间：2026-09-05

当前研究基线已集成到 `main`。优先读取当前状态、任务和冻结协议；历史报告承担追溯职责。写作、术语和代码边界统一遵守仓库根目录的 [AGENTS.md](../AGENTS.md)，此处只维护导航。

## 当前入口

- [STATE.md](STATE.md)：当前实验进度、结论和代码来源。
- [implementation/TASKS.md](implementation/TASKS.md)：当前任务与后续执行顺序。
- [requirements/SPEC.md](requirements/SPEC.md)：毕业论文目标和仓库能力边界。
- [requirements/thesis-frozen-paper-evaluation-v3.2.3.md](requirements/thesis-frozen-paper-evaluation-v3.2.3.md)：当前冻结实验合同；此前版本用于追溯。
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

旧应用任务协议 [downstream-evaluation-spec.md](requirements/downstream-evaluation-spec.md) 和[仿真生成器规格](requirements/synthetic-benchmark-spec.md)是基础或历史合同；当前执行优先服从九月冻结协议及其明确继承项。

## 本地与兼容入口

`docs/SECRETS.md` 是被忽略的本地连接信息，不属于远端文档交付。原始数据、稠密表示和检查点位于被忽略的 `artifacts/`。解析或编辑 Word 文档须使用文档技能或插件。

`docs/planning`、`docs/foundation`、`docs/models` 分别链接至 `implementation/notes`、`requirements/foundation`、`requirements/model-contracts`，未复制正文。新引用使用实际目录，已有兼容入口保留。
