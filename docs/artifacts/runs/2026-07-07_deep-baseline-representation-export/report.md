# Deep Baseline Representation Export for E3

- run_id: `2026-07-07_deep-baseline-representation-export`
- status: `completed`
- 评价定位：为 Dingxin E3 结构诊断补齐 MulT / ContiFormer 的 held-out pooled embedding 流。
- training_invoked: `true`
- confirmed_metrics_changed: `false`
- thesis_protocol_snapshot_modified: `false`
- representation_family: `T2_response_lovo_seed17_pooled_embedding`
- task: `T2_next_window_physiology_response` / `regression`
- split: `leave_one_view_out`
- seed: `17`
- epochs: `20`

## OOF 协议

每个 `fusion_feature_*` 行来自该样本所属 held-out view 的 inference。训练只使用 train fold 的 T2 标签；导出的 E3 主输入只使用 `pooled_embedding`，不包含 logits、预测值、rank 或诊断标量。

## 模型状态

- `contiformer`: OOF embedding available = `True`
- `mult`: OOF embedding available = `True`

## 输出

- `deep_baseline_oof_embeddings_long.csv`
- `deep_baseline_oof_embeddings_summary.json`
- `checkpoint_manifest.csv` / `checkpoint_manifest.json`
- `representation_manifest.json`
- `training_curves.csv`
- `fold_status.csv`

## 边界

该产物只作为 E3 融合表示流结构评价的输入，不回写分类任务、回归任务或检索任务 confirmed metrics，也不修改论文协议快照。E3 结果是否进入论文仍需后续人工 review。
