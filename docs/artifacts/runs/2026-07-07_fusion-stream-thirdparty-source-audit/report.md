# 融合表示流第三方来源审计报告

- run_id: `2026-07-07_fusion-stream-thirdparty-source-audit`
- 审计结论：`C. no_reusable_sources`
- 是否找到 MulT / ContiFormer 可复用融合表示流：`false`
- 是否找到 Dingxin MulT / ContiFormer 可加载 checkpoint：`false`
- 是否实现 adapter：`false`
- 是否运行 Dingxin 四方法 E3 validation：`false`
- 是否训练：`false`
- 是否修改 confirmed metrics：`false`
- 是否回写论文协议快照：`false`

## 结论

当前 E3 工程链路已经完成，合成数据四方法 validation 与小规模 Dingxin 两方法 validation 均可复现。Dingxin real run 暂时只能作为 `chronaris` 与 `naive_time_sync` 的 two-method validation；MulT 和 ContiFormer 没有在当前 repo artifact 中保留可复用的融合表示流，也没有找到可加载的 Dingxin 第三方模型 checkpoint。

因此，本轮不应硬做四方法 E3，不应把任务预测、检索 rank、embedding norm 或 attention entropy 诊断标量伪装成 `fusion_feature_*`。MulT / ContiFormer 必须继续保留 `method_unavailable`，直到另起 deep baseline representation export 任务产出同 split、同样本、可追溯 checkpoint 和表示向量。

## 审计范围

重点审计了以下来源：

- `docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/`
- `docs/artifacts/runs/2026-07-02_selected-model-summary/`
- `docs/artifacts/runs/2026-07-02_selected-model-reevaluation/`
- `docs/artifacts/runs/2026-05-01_deep-real-sortie-prepared/`
- `docs/artifacts/runs/2026-05-01_deep-comparison-prepared/`
- `docs/artifacts/runs/2026-05-01_full-loso-deep-comparison/`
- `docs/artifacts/runs/`
- host-local backup roots under `/home/wangminan/projects/chronaris-local-artifacts/`

结构化 inventory 见 `source_inventory.csv` 与 `source_inventory.json`。

## 关键发现

| id | 来源 | 发现 | E3 含义 |
| --- | --- | --- | --- |
| `dingxin_thirdparty_root_current` | `docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/` | current repo root keeps summary, metrics, manifests, logs, figures, split/task manifests, and sequence schema; no .pt/.pth/.npz/.parquet or fold_predictions.csv remains | no reusable MulT/ContiFormer representation or checkpoint in current repo artifact root |
| `deep_model_code_api` | `src/chronaris/modeling/common/deep_models.py; src/chronaris/evaluation/dingxin/pipelines/thirdparty_comparison.py` | MulT and ContiFormer wrappers return pooled_embedding and sequence_embedding, and the third-party comparison code can call _embed_all after training | an export adapter is feasible only if a matching trained checkpoint and fold/source metadata exist |
| `no_feature_frame_mapping` | `src/chronaris/evaluation/dingxin/pipelines/benchmark_data.py; thirdparty_comparison.py FEATURE_MODEL_SOURCES` | build_variant_feature_frames creates naive_sync, E/F projection, G variants, and optimized Chronaris frames; FEATURE_MODEL_SOURCES maps chronaris_full/naive_time_sync/classical_baseline only | current E3 Dingxin loader cannot discover MulT/ContiFormer feature_values without a separate representation export |
| `deep_real_scalar_summary` | `docs/artifacts/runs/2026-05-01_deep-real-sortie-prepared/.../{mult,contiformer}/sample_summary.csv` | sample summaries contain sample_id, view_id, sortie_id, pilot_id, diagnostic verdict, embedding_norm, attention_entropy, top_event_concentration, and event_mask_interference only | scalar diagnostics are not a fusion_feature_* vector stream and should not enter E3 main input |
| `sequence_bundle_raw_inputs` | `docs/artifacts/runs/2026-05-01_deep-real-sortie-prepared/feature_export_case_sequences/sequence_bundle.npz` | sequence bundle contains raw/prepared physiology and vehicle arrays plus masks, time_axis, labels, metadata, and diagnostics | it is an input bundle, not trained MulT/ContiFormer output; using it would require training or a checkpoint |
| `external_prediction_backup` | `/home/wangminan/projects/chronaris-local-artifacts/dense-predictions-pruned-20260702/` | external backup has fold_predictions.csv files, including Dingxin third-party predictions and older deep-real scalar diagnostics | predictions/ranks or scalar diagnostics are explicitly invalid as E3 fusion streams |
| `external_checkpoint_backup` | `/home/wangminan/projects/chronaris-local-artifacts/checkpoints-history-20260702/` | external checkpoint backup has alignment and multitask checkpoints but no Dingxin MulT/ContiFormer third-party checkpoint candidate | no found checkpoint that would support no-training MulT/ContiFormer E3 export |


## Inventory 摘要

- inventory rows: `214`
- status counts: `{'no_reusable_source_found': 3, 'not_direct_fusion_stream': 127, 'not_e3_dingxin_thirdparty_checkpoint': 15, 'prediction_only_not_valid_fusion_stream': 52, 'raw_or_prepared_sequence_not_model_representation': 11, 'scalar_summary_or_report_only': 6}`
- kind counts: `{'checkpoint_candidate': 15, 'method_named_report_or_summary': 127, 'no_candidate_files': 3, 'prediction_table_candidate': 52, 'representation_named_candidate': 6, 'tensor_cache_candidate': 11}`

其中：

- `prediction_only_not_valid_fusion_stream` 表示只有任务预测或检索 rank，不能作为融合表示流。
- `raw_or_prepared_sequence_not_model_representation` 表示只有原始/准备序列或输入张量，不是训练后模型表示。
- `scalar_summary_or_report_only` 表示文件名或图名出现 representation，但内容是稳定性图或 scalar summary，不是可展开的向量列。
- `not_e3_dingxin_thirdparty_checkpoint` 表示 checkpoint 属于 alignment/multitask 等其他路径，不是 Dingxin MulT/ContiFormer 第三方比较模型。

## 代码导出点判断

`src/chronaris/modeling/common/deep_models.py` 的 MulT 与 ContiFormer wrapper 已有 `pooled_embedding` 和 `sequence_embedding` 输出；`src/chronaris/evaluation/dingxin/pipelines/thirdparty_comparison.py` 也有 `_embed_all(...)`，能在训练后提取 `pooled_embedding`。这说明后续可以设计无训练 checkpoint export adapter。

但当前可审计 artifact 缺少 matching checkpoint。没有 checkpoint 时，调用这些 API 只能得到未训练模型输出，不符合公平比较边界。本轮也不能重跑分类任务、回归任务或历史检索任务的训练流程。

## 结果边界

E3 是无监督结构诊断，不替代分类任务和回归任务。合成数据四方法 run 只证明工程链路和 evaluator API 可执行；当前 Dingxin two-method validation 不能写成四方法正式论文结论。短序列下 CLaP unavailable 是已记录边界，不是本轮 blocked 原因；真正 blocked 点是 MulT / ContiFormer 缺少可复用融合表示流或 checkpoint。

下一步见 `next_prompt_for_deep_baseline_export.md`。
