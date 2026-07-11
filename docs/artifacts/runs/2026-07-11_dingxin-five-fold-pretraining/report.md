# 鼎新五折六方法公共预训练与统一表示报告

## 结论

- 状态：完成；聚合验收 13/13 通过，五个子 run 合计 60/60。
- 三个留一视图主协议折与两个留一架次辅助协议折均完成；六方法共形成 30 个 checkpoint 和 90 份 train/validation/outer-test 表示。
- 90 份 archive、manifest、样本顺序、source hash 与 checkpoint lineage 已逐项重验；每折第二遍恢复均复用 18/18。
- 五折五个可训练方法累计训练 1066.45 秒；所有运行实测最高峰值内存 2047.1 MB，低于 2.5 GB 门限。
- 预训练仍未打开机动分类或生理响应目标，outer-test 未计算任务指标；本报告不构成模型排名。

## 下一步

1. 用同一五折表示接入固定线性与 MiniROCKET 工程冒烟。
2. 正式候选筛选前按每折 inner-train 重建嵌套任务目标。
3. 五折 consumer 完成后再启动 Chronaris 候选 screen，不提前读取锁定仿真测试。
