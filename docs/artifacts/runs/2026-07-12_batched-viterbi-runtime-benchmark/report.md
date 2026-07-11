# 持续时间约束批量解码运行时基准

状态：completed；该优化只向量化样本维，不改变半马尔可夫递推、持续时间范围、转移概率或回溯规则。

输入为 seed 17 生理单流在 `timestamp_jitter_000ms` 场景的真实 TCN logits，形状 `[192,96,5]`。批量解码耗时 1.764 秒；逐样本前 4 条耗时 1.616 秒，线性外推 192 条约 77.546 秒，对应约 43.95 倍加速。批量输出与逐样本输出的前 4 条逐位完全一致，独立随机批次单元测试也逐位一致。

旧标量压力运行保存在被忽略目录 `artifacts/application_evaluation/2026-07-12_simulation-locked-stress-consumers-scalar-viterbi-runtime/`，未完成任何 seed—场景单元，不进入正式压力结果。正式 run 从空根重启。
