# 六方法公共预训练与线性下游闭环冒烟报告

## 结论

- 状态：完成；验收 20/20 通过。
- 五个可训练编码器在同一 8 条仿真训练轨迹上使用相同增强和三个公共目标完成 1 epoch；朴素时间同步只拟合训练折无监督变换。
- 三个公共目标共产生 30 条 active step 记录；五方法累计训练耗时 7.29 秒。
- 六方法 train/validation/held-out 共导出 18 份表示，恢复复核复用 18 份；删除 Chronaris 留出折后单项重建的哈希一致。
- 固定 Logistic/Ridge 线性 consumer 产生 72 条 smoke 指标，其中 72 条可计算。
- 仿真 workload 真值在五个 checkpoint 完成后才打开；表示预训练没有读取 oracle 或下游标签。
- 本 run 不比较模型优劣，不更新论文主指标。

## 训练资源

| 方法 | 状态 | 主干参数 | 预训练头参数 | 训练秒数 | step |
| --- | --- | ---: | ---: | ---: | ---: |
| physiology_only | resumed | 110208 | 2663 | 0.320 | 2 |
| vehicle_only | resumed | 111168 | 2663 | 0.264 | 2 |
| mult | resumed | 1014784 | 2663 | 0.542 | 2 |
| contiformer | resumed | 112512 | 2663 | 0.275 | 2 |
| chronaris | resumed | 123222 | 2663 | 5.894 | 2 |

## 下一步

1. 实现 MiniRocket 与 TCN/Viterbi 下游 consumer，并把真实弱监督任务接入相同表示合同。
2. 将 smoke 数据扩展到完整 G1 train/validation，运行 seed 17 四候选开发筛选。
3. 开发筛选期间继续禁止读取 G2 锁定测试，并保留所有 mixed/negative 结果。
