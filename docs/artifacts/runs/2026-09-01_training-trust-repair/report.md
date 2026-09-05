# 论文主线可信训练修复

## 阶段结论

论文主线已完成第一批训练可信度修复：公共预训练统一复用 validation-backed candidate trainer，checkpoint 升级为 v2，并封闭管理 Python、NumPy、Torch CPU/CUDA 随机状态。表示导出只在适配器内部归一化一次，公共数据脚本的下游缩放器只在外层训练折拟合。

本轮只形成工程与协议证据，不产生论文任务指标。CogPilot 与 CLARE 的原生时间样本解析、受控缓存和公开数据完整重跑仍属于下一子阶段。

## 主要变更

- `common_pretraining` 改为 candidate trainer 的兼容入口，不再独立生成“末轮等于最佳轮”的 checkpoint。
- candidate、common、locked 三条训练路径共享封闭随机状态；v2 checkpoint 保存模型、任务头、优化器、epoch、patience 和完整随机状态，v1 只允许加载推理、禁止续训。
- 滞后损失从 `positive.auxiliary["alignment_output"]` 读取对齐结果；启用目标但缺少输出时立即失败。
- 训练适配器内部完成唯一一次归一化，并按真实有效查询 mask 汇聚。
- CogPilot 与 CLARE 的外层训练集增加确定性的受试者分组 inner-validation；下游消费者使用训练折拟合的 sklearn pipeline。
- CogPilot 飞行难度、CogPilot 事件响应和 CLARE 认知负荷上下文长度分别固定为 30 秒、12 秒和 10 秒。

## 验证结果

- 聚焦训练与分组划分测试：`16 passed`。
- 完整测试：`433 passed, 8 skipped`，用时 138.13 秒。
- CUDA 短重放：AMP 前后向与因果时序消费者训练 `2 passed`，实际设备为 RTX 4090。
- Python 编译检查通过。
- `git diff --check` 通过。

详细判据见 [acceptance_checks.csv](acceptance_checks.csv)。

## 未完成边界

当前尚未声称原生时间公开数据输入完成，也尚未运行修复后的公开数据或鼎新应用指标。下一步先完成 CogPilot/CLARE 原生时间解析与 lazy batch provider，再进入连续演化和可学习事件语义实现。
