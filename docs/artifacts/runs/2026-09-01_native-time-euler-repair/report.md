# 原生时间输入与连续演化子步修复

## 阶段结论

CogPilot 与 CLARE 的样本解析已从研究脚本移入 `src/chronaris/dataset`，训练入口改为 lazy batch provider。两个流分别保留原始时间戳、采样密度与缺失 mask；同一流内的多传感器按时间戳并集组织，不再进行跨流统一插值。ODE-RNN 同时增加可选的 Euler 最大子步，兼容默认值 `None` 保持旧 checkpoint 行为。

本轮只验证数据与数值积分机制可运行，不选择最终 ODE 方案，也不产生论文任务指标。单步 Euler、0.5 秒 Euler 子步与 RK4 的仿真 validation 比较仍需按评价协议 v3 运行。

## 真实数据冒烟

- CogPilot 飞行难度样本保留 18,961 个生理观测时刻和 135 个航电观测时刻；生理多传感器 mask 占比为 0.301，最长时间分别为 29.999 秒与 29.959 秒。
- CogPilot 事件响应样本保留 7,583 个生理观测时刻和 55 个航电观测时刻；最长时间均未超过 12 秒上下文。
- CLARE 样本保留 2,560 个中枢脑电观测时刻和 293 个外周观测时刻；外周多传感器 mask 占比为 0.5，最长时间均未超过 10 秒上下文。
- 三类样本均能直接进入 Chronaris，CogPilot 和 CLARE 均得到有限的 `[1,96,64]` 标准化融合表示。

上述点数是抽样冒烟证据，用于证明原生时间与 mask 合同生效，不作为数据集总体统计或任务结论。

## 工程验证

- 原生时间缓存、时间戳并集、窗口边界和 Euler 子步聚焦测试：`26 passed`。
- 完整测试：`439 passed, 8 skipped`，用时 134.77 秒。
- RTX 4090 上 Euler 子步前向与反向有限，输出形状为 `[3,8]`。
- Python 编译、Ruff 和 `git diff --check` 通过。

详细判据见 [acceptance_checks.csv](acceptance_checks.csv)。
