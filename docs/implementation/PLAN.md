# Chronaris 执行计划

更新时间：2026-06-07

## 1. 计划定位

本文是当前 AI coding 的主动执行入口。旧 `coding-roadmap.md`、Stage I roadmap、coding plan 和 gap 评估已保留在 [notes/](notes/)，但后续默认先读本文。

目标不是继续扩大旧 benchmark，而是把已有研究原型收敛成更贴近选题报告的论文主线。

## 2. 总路线

1. 读取指定架次的人机多源数据及元信息。
2. 建立统一 schema、统一时间参考和统一样本组织。
3. 实现双流连续潜态建模。
4. 实现物理一致性约束时间对齐。
5. 实现因果掩码跨模态融合。
6. 输出标准化融合特征与中间态接口。
7. 面向典型任务开展对比、消融和案例验证。

## 3. 阶段结构

### Stage A：仓库初始化与最小设计

状态：已完成。

作用：固化仓库边界、分层、统一对象和最小 pipeline 设计。

### Stage B：真实元信息与数据访问接入

状态：已完成 preview 路径。

作用：打通真实 MySQL 元信息读取、InfluxDB 时序读取和最小真实架次数据链路。

### Stage C：统一样本组织与数据核验

状态：已完成。

作用：把“能读出来”提升为“知道数据是否可用”，并确认 full coverage 下存在可用重叠窗口。

### Stage D：数据集工程化与批量构建

状态：后置。

作用：把单架次链路扩展成可复用、可批量运行的数据集工程。

### Stage E0：单架次最小训练输入适配

状态：已完成 preview 路径。

作用：在完整数据集工程化前，形成可直接喂给模型的单架次实验输入。

### Stage E：双流连续潜态对齐

状态：已完成收口。

作用：完成最小训练闭环、`relative_mse` 真实回归、样本级诊断、checkpoint 导出。

### Stage F：物理一致性约束对齐

状态：历史 weak/full physics family 已完成收口；当前下一步是刚体运动物理约束补强。

当前目标：把已有弱物理约束整理成可选择、可诊断、可单测的 `rigid-body physics` family。

### Stage G：因果掩码跨模态融合

状态：`G(min)` 已完成收口；当前下一步是语义事件融合补强。

当前目标：从时间步注意力升级到 `SemanticQueryBank / EventTokenExtractor / CausalEventFusion`，输出事件级归因。

### Stage H：标准化融合特征导出

状态：已完成收口。

作用：稳定导出双流 view，并保持 `load_stage_h_feature_run()` 与 all-window contract 作为后续私有资产和历史基线依赖。

### Stage I：典型任务评测、论文证据与运行时

状态：历史公开 benchmark 已收口；论文主线 Phase A/B/C 首轮收敛已完成，当前需要固化 Phase C 并补真实资产联合训练证据。

当前主线：

1. `Phase A/B/C`：主线边界、统一骨干、checkpoint inference export、任务头与 weak-label thesis task builder。
2. `Phase D`：Stage F rigid-body physics 和 Stage G semantic event fusion。
3. `Phase E`：runtime inference。

## 4. 当前优先级

1. P0：冻结当前 Phase C 工作区，补跑必要测试并提交。
2. P1：补一条真实资产上的 Stage I multitask 联合训练证据。
3. P2：补 Stage F 刚体运动物理约束。
4. P3：补 Stage G 语义事件融合。
5. P4：补 runtime inference。

## 5. 历史计划入口

- [notes/coding-roadmap.md](notes/coding-roadmap.md)
- [notes/stage-i-thesis-mainline-roadmap-2026-05-15.md](notes/stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md](notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [notes/thesis-coding-gap.md](notes/thesis-coding-gap.md)
- [notes/iteration-playbook.md](notes/iteration-playbook.md)
