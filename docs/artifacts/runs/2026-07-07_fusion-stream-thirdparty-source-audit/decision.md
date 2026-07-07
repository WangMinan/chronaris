# Fusion Stream Third-Party Source Audit Decision

结论：`C. no_reusable_sources`

## 判据

- 当前鼎新第三方模型对比 artifact 保留了 summary、metrics、figures、logs、task/split manifests 和 sequence schema，但当前 repo 根下没有保留 `.pt` / `.pth` checkpoint、向量 cache、hidden-state table、embedding table 或可进入 E3 的预测表。
- 本机外置 dense prediction backup 只有任务预测或检索 rank。这些是监督任务输出，不是 E3 融合表示流。
- 旧 deep baseline roots 只有 `embedding_norm`、`attention_entropy`、`top_event_concentration`、`event_mask_interference` 等 scalar diagnostics，以及 raw sequence bundle。它们不能展开为 `fusion_feature_*` 向量。
- MulT / ContiFormer wrapper 在代码上有 representation API，但公平使用这些 API 需要匹配的已训练 checkpoint 或另行批准的 representation export run。

## 执行决定

本轮不实现 `deep_baseline_sources.py`。Dingxin E3 中的 `mult` 和 `contiformer` 继续保留为 `method_unavailable`，当前 Dingxin run 只作为 two-method validation。

## 后续入口

只有在明确允许训练或提供可恢复 checkpoint 后，才另起 deep baseline representation export 任务。该任务必须先固定 seed、split、checkpoint policy、representation family 和输出合同，再让结果进入 E3。
