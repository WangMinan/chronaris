# E3 融合表示流结构评价执行报告

- run_id: `2026-07-07_fusion-stream-structure-dingxin-dry-run`
- 评价定位：E3 是无监督结构诊断，用于观察融合表示流是否形成连续、稳定、可复盘的状态轨迹。
- 论文边界：E3 不替代分类任务和回归任务；历史检索任务 artifact 未删除。
- 本轮是否训练：`false`。
- 是否修改 confirmed metrics：`false`。
- 可用方法：chronaris, naive_time_sync。
- 不可用方法：mult=no_reusable_fusion_feature_frame; contiformer=no_reusable_fusion_feature_frame。
- 外部库状态：claspy=unavailable；stumpy=unavailable。

## 输出文件

- `e3_metrics_long.csv`
- `e3_summary.json`
- `plots/state_timeline_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/transition_graph_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/fragment_replay_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/state_timeline_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/transition_graph_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/fragment_replay_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/state_timeline_naive_time_sync_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/transition_graph_naive_time_sync_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/fragment_replay_naive_time_sync_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/state_timeline_naive_time_sync_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/transition_graph_naive_time_sync_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/fragment_replay_naive_time_sync_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `evidence_manifest.json`

## 结果边界

本轮输出只进入 E3 专属 long 表，`evidence_quadrant = fusion_stream_structure`。Composite score 只作为固定权重汇总展示，不作为 winner 结论。若外部库缺失或序列过短，对应指标保留为 unavailable，不静默删除。

## 当前摘要

- metric rows: 46
- completed metric rows: 0
- unavailable metric rows: 46
