# Alignment Batch Contract

## 1. 目的

这份文档定义阶段 E 之前的最后一层协议：

- `E0ExperimentSample`
- 到
- `AlignmentBatch`

这一步的作用是让模型代码不再直接依赖窗口对象或原始 dataclass，而是只面对数值矩阵和掩码。

## 2. 输入

输入是：

- `E0ExperimentSample[]`

每个样本已经包含：

- physiology 数值矩阵
- vehicle 数值矩阵
- 时间偏移
- feature 名称

## 3. 输出

输出是：

- `AlignmentBatch`

批次中包含：

- physiology values
- physiology mask
- physiology offsets
- vehicle values
- vehicle mask
- vehicle offsets
- sample ids
- feature names

## 4. 当前实现边界

当前批处理层使用 `numpy`，不绑定具体深度学习框架。

这样做的意义：

1. 先把输入协议稳定下来
2. 让后续 `torch` / 其他框架接入更直接
3. 先把 padding / mask / offset 对齐逻辑固定

## 5. 进入模型编写阶段的意义

当 `AlignmentBatch` 可稳定生成时，就可以开始真正写模型：

1. 编码器
2. 连续时间状态推进
3. 重构损失
4. 对齐损失

也就是说，模型阶段不再缺输入协议。
