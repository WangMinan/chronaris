# 鼎新简化下游评价协议审计

## 结论

新版未来任务合同已经按冻结原始点完成实跑核验：共形成 90 个完整视图上下文和 60 个独立机动上下文。

历史冻结表示不能作为新版主结果直接复用，兼容性结论为 `retraining_required`。原因是旧输入合同排除了 20 个可用于未来预测的历史运动学字段，且旧的内部划分没有覆盖全部新版上下文。旧表示仅保留为历史敏感性证据。

## 任务与划分核验

- 输入严格截止于目标起点，目标窗口固定为随后 5 秒。
- 未来机动评价按独立飞机上下文聚合，共享航电轨迹的两个视图不会重复计权。
- 未来生理状态逐字段预测，尺度和字段选择均只由当前训练架次拟合。
- 留一架次为主评价；本轮未训练模型，也未打开或修改任何确认指标。

- `leave_one_sortie_out__fold01`：训练折选择 12 个生理字段。
- `leave_one_sortie_out__fold02`：训练折选择 11 个生理字段。

## 可复核产物

- context_manifest：`artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/context_manifest.csv`
- maneuver_statistics：`artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/maneuver_statistics.csv`
- physiology_statistics：`artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/physiology_statistics.csv`
- fold_manifest：`artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/fold_manifest.csv`
- maneuver_targets：`artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/maneuver_targets.csv`
- physiology_targets：`artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/physiology_targets.csv`
- thresholds：`artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/fold_fitted_parameters.csv`
