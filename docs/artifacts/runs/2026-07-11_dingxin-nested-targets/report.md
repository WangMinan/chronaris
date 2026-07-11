# 鼎新 inner-train 嵌套目标报告

## 结论

- 状态：完成；验收 10/10 通过。
- 五折机动分位阈值、语义尺度、生理字段 IQR 与高响应阈值均只使用各折 inner-train 重拟合。
- 生成 10 个确定性目标 archive；机动分类 440 个角色上下文，生理响应 425 个可用角色上下文。
- validation 与 outer-test 仅应用训练内参数，不参与字段选择、尺度或阈值估计。
- 本 run 不训练模型、不生成指标，也不形成候选排名。

## 下一步

1. 用嵌套目标复跑 validation consumer，保持 outer-test 指标关闭。
2. 完成后进入 seed 17 固定候选 screen。
