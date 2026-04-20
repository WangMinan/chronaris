# Implementation Summary History (2026-04-15 ~ 2026-04-17)

## 1. 合并说明

本文档用于合并以下两份历史实现记录，减少规划文档重复维护成本：

1. `implementation-summary-2026-04-15.md`
2. `implementation-summary-2026-04-17.md`

两份原始文档在 2026-04-20 后已并入本摘要，不再单独维护。

## 2. 阶段演进摘要

### 2026-04-15：从 preview 联调推进到阶段 E 就绪

主要完成项：

1. 真实 MySQL / Influx 访问链路与 preview 数据链路打通
2. 阶段 C overlap 结论确认：完整数据存在真实重叠
3. overlap-focused E0 输入适配完成（`25` 个样本，生理 `12` 特征，飞机 `21` 特征）
4. `AlignmentBatch` 最小模型输入协议落地

阶段判断：

- 已满足阶段 E 最小启动条件

### 2026-04-17：阶段 E 最小训练闭环落地

主要完成项：

1. 服务器 WSL Ubuntu + GPU 训练环境完成迁移与验证
2. `split / reference grid / torch batch` 基础模块稳定
3. 最小确定性双流 ODE-RNN forward + reconstruction/alignment loss 完成
4. 最小 preview `train / validation / test` pipeline 落地
5. 已完成一轮真实 overlap-focused 训练回归

阶段判断：

- 阶段 E 从“前向原型”推进到“最小训练闭环”
- 新 blocker 转移到 vehicle stream 尺度主导问题

## 3. 对当前（2026-04-20）仍有效的历史事实

1. 当前联调目标架次与 overlap-focused 查询窗口保持不变
2. `E0ExperimentSample -> AlignmentBatch -> Stage E preview` 链路已连续可运行
3. 训练环境优先级仍为“实验室服务器 4090 > 家里 4070 Ti”
4. 阶段 F/G（物理约束/因果融合）继续后置，不前置到阶段 E

## 4. 后续阅读顺序

1. [coding-roadmap.md](coding-roadmap.md)
2. [stage-e-execution-plan.md](stage-e-execution-plan.md)
3. [implementation-summary-2026-04-20.md](implementation-summary-2026-04-20.md)
