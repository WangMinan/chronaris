# 论文主线现场保护与基线预检

日期：2026-09-01。分支：`research/thesis-continuous-semantic-fusion-202609`。

## 结论

论文主线已从安全滞后感知融合历史提交 `eab2c0b22cc9d7bcb34fe2aa4f5adb52a0ce2a30` 建立，历史证据标签为 `evidence/safe-lag-exploration-20260724`。当前用户未提交的指标删除、CLARE 报告修改、中期材料和 `goal.md` 均保持原样，没有执行重置、恢复或覆盖。

完整基线测试为 `429 passed, 8 skipped, 319 warnings`，耗时 `185.58 s`；`git diff --check` 通过。该结果只说明当前工程基线健康，不代表研究门禁已通过。

## 环境与数据

- 主机：`DESKTOP-8RNE6N6`，Python `3.11.15`，PyTorch `2.11.0+cu130`。
- CUDA：NVIDIA GeForce RTX 4090，24,564 MiB，总预检空闲显存约 22,843 MiB。
- CPU：64 logical cores；工作盘可用空间约 203 GiB。
- CogPilot 数据约 44 GiB，CLARE 数据约 6.3 GiB。
- 鼎新冻结输入约 5.3 MiB，正式仿真输入约 1,018 MiB，安全滞后感知鼎新重型产物约 145 MiB。

## 约束

论文长跑必须通过 CUDA 预检，不允许静默切换 CPU。公开数据、仿真和程序化错误配对分别作为外部校准、受控机制验证和负对照材料；它们不等价于新增鼎新真实数据或专家真值。
