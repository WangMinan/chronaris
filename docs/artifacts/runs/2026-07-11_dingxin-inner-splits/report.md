# 鼎新外层折训练内验证划分报告

## 结论

- 状态：完成；验收 11/11 通过。
- 五个外层折均形成 inner-train、validation、overlap embargo 和 outer-test 四种角色；每折 93 个完整输入只出现一次。
- 外层训练组含两个不同架次时，完整留出一个训练架次做 validation；只含同一架次时，按时间块划分并删除所有与 validation 30 秒窗口重叠的中间上下文。
- inner-train 与 validation 在共享航电流上的原始时间区间重叠数为 0；样本集合与 outer-test 也完全分离。
- 每折分类的三种角色都覆盖低、中、高三类；生理响应三种角色均有连续目标和高/非高两类。
- 当前目标阈值由 outer-train 拟合，只允许本阶段固定配置 smoke 使用；正式候选筛选必须按 inner-train 重拟合嵌套目标。
- 本 run 不训练模型、不读取 outer-test 指标，也不形成候选排名。

## 下一步

1. 以本 split manifest 为唯一输入拟合五折归一化器和公共预训练 checkpoint。
2. 为六方法导出 inner-train/validation/outer-test 表示并核对样本哈希。
3. 固定 consumer smoke 可使用现有 outer-train 目标；进入正式 screen 前先生成 inner-train 嵌套目标 archive。
