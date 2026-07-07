# Next Prompt: Deep Baseline Representation Export for E3

仅在后续任务明确允许训练，或已经提供可恢复 checkpoint 时使用。

## 目标

为 Dingxin E3 产出公平的 MulT 与 ContiFormer 融合表示流，同时不污染 confirmed metrics。

## 为什么当前审计不能补齐四方法 E3

当前仓库 artifact 中没有 Dingxin MulT / ContiFormer 已训练 checkpoint，也没有向量 representation table。可找到的备份只有任务预测、检索 rank 或 scalar diagnostics。预测值、rank、embedding norm 或 attention diagnostic 都不是融合表示流，不能转换成 `fusion_feature_*` 进入 E3。

## 公平导出协议

1. 复用既有 Dingxin 样本、标签、split manifest 和 leakage-safe protocol。
2. 仅在明确批准的新 run root 下训练或恢复 MulT 与 ContiFormer。
3. checkpoint 必须记录 model name、task name、split strategy、seed、fold、model hyperparameters、modality dimensions 和 source manifest paths。
4. 使用模型 API 的 `pooled_embedding` 导出表示；如需使用 `sequence_embedding`，必须先文档化 pooling 规则。
5. E3 extraction 不使用 task label。label 只能作为 post-hoc metadata 参与结构诊断。
6. 导出 long table，字段至少包括：
   - `method_name`
   - `sortie_id`
   - `view_id`
   - `window_id`
   - `time`
   - `fusion_feature_1..fusion_feature_d`
   - `source_model_name`
   - `source_artifact_path`
   - `source_task_name`
   - `source_checkpoint_path`
   - `source_fold_or_seed`
   - `representation_family`
7. evidence manifest 必须记录 `training_invoked`、`confirmed_metrics_changed=false`、`thesis_protocol_snapshot_modified=false` 和固定 extraction 参数。
8. 只有 MulT 与 ContiFormer 均有有效 representation stream 后，才运行 `chronaris`、`naive_time_sync`、`mult`、`contiformer` 四方法 E3。

## 禁止项

- 不把 logits、predicted labels、regression values、retrieval rank 或 scalar diagnostics 当成 `fusion_feature_*`。
- 不修改 `docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/`。
- 不更新 confirmed metrics tables。
- 不删除历史检索 artifacts。
- 除非四类方法使用可比 representation stream，否则不把 Dingxin 四方法 E3 写成论文主结果。
