# E3 融合表示流结构评价执行报告

- run_id: `2026-07-07_fusion-stream-structure-dingxin-four-method-validation`
- 评价定位：E3 是无监督结构诊断，用于观察融合表示流是否形成连续、稳定、可复盘的状态轨迹。
- 论文边界：E3 不替代分类任务和回归任务；历史检索任务 artifact 未删除。
- 本轮是否训练：`false`。
- 是否修改 confirmed metrics：`false`。
- 可用方法：chronaris, contiformer, mult, naive_time_sync。
- 不可用方法：无。
- 外部库状态：claspy=available(0.2.8, import_available=True)；stumpy=available(1.14.1, import_available=True)。
- evaluator 状态：ClaSP=completed:12；CLaP=clap_unavailable:12；STUMPY=completed:12。
- MulT / ContiFormer embedding 来源：`docs/artifacts/runs/2026-07-07_deep-baseline-representation-export`。
- representation_family：`T2_response_lovo_seed17_pooled_embedding`。

## 输出文件

- `e3_metrics_long.csv`
- `e3_summary.json`
- `plots/state_timeline_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/transition_graph_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/fragment_replay_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/state_timeline_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/transition_graph_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/fragment_replay_chronaris_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/state_timeline_chronaris_20251005__01_ACT-4___J20_22_01_20251005__01_ACT-4___J20_22_01__pilot_10033.png`
- `plots/transition_graph_chronaris_20251005__01_ACT-4___J20_22_01_20251005__01_ACT-4___J20_22_01__pilot_10033.png`
- `plots/fragment_replay_chronaris_20251005__01_ACT-4___J20_22_01_20251005__01_ACT-4___J20_22_01__pilot_10033.png`
- `plots/state_timeline_contiformer_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/transition_graph_contiformer_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/fragment_replay_contiformer_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/state_timeline_contiformer_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/transition_graph_contiformer_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/fragment_replay_contiformer_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/state_timeline_contiformer_20251005__01_ACT-4___J20_22_01_20251005__01_ACT-4___J20_22_01__pilot_10033.png`
- `plots/transition_graph_contiformer_20251005__01_ACT-4___J20_22_01_20251005__01_ACT-4___J20_22_01__pilot_10033.png`
- `plots/fragment_replay_contiformer_20251005__01_ACT-4___J20_22_01_20251005__01_ACT-4___J20_22_01__pilot_10033.png`
- `plots/state_timeline_mult_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/transition_graph_mult_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/fragment_replay_mult_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10033.png`
- `plots/state_timeline_mult_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/transition_graph_mult_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `plots/fragment_replay_mult_20251002__01_ACT-8___J16_12_01_20251002__01_ACT-8___J16_12_01__pilot_10035.png`
- `evidence_manifest.json`

## 结果边界

本轮输出只进入 E3 专属 long 表，`evidence_quadrant = fusion_stream_structure`。Composite score 只作为固定权重汇总展示，不作为 winner 结论。若外部库缺失、短序列状态检测不足或方法无可复用融合流，对应指标保留为 unavailable，不静默删除。

若本 run 消费 deep baseline OOF embedding，该 embedding 仅表示 T2 任务训练后的模型层 pooled representation；E3 不替代分类任务和回归任务，也不给单一 winner。结果是否可用于论文必须由后续人工 review 决定。

## 当前摘要

- metric rows: 160
- completed metric rows: 124
- unavailable metric rows: 36
