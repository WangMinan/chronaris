# Stage I Alignment Support - 20260506T120000Z-stage-i-support

- generated_at_utc: `2026-05-06T02:58:16.462491Z`
- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support`
- machine summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/support_summary.json`
- main ablation matrix: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/ablation_matrix.csv`

## Alignment Chain

| stage | sample_count | mean_projection_cosine | mean_projection_l2_gap | threshold_verdict |
| --- | ---: | ---: | ---: | --- |
| `E baseline` | 3 | 0.760173 | 0.211933 | `PASS` |
| `F full` | 3 | 0.699514 | 0.212016 | `PASS` |
| `F - E delta` | 3 | -0.060659 | 0.000083 | `PASS` |

## Stage H Export Stability

- sortie_count: `2`
- generated_view_count: `3`
- view verdict counts: `{'PASS': 2, 'WARN': 1}`
- partial_data_entry_count: `1`
- partial_data_built_entry_count: `1`

## 对齐结论

1. `E baseline -> F full` 的投影诊断已经形成连续证据链，说明不是简单拼接。
2. `F full` 相比 `E baseline` 的 mean_projection_cosine 变化为 `-0.060659`，mean_projection_l2_gap 变化为 `+0.000083`。
3. `Stage H` 已把对齐结果稳定导出为 3 个双流 view，可直接被下游与 case-study 消费。
4. 这条报告回答的是“对齐是否做出来且可复用”，不是“对齐已在公开数据上证明最优”。

## 本结论能支撑什么

- 可以支撑论文中“连续对齐已形成稳定 export contract，并已被下游 Phase 2 case-study 消费”的表述。
- 可以支撑“物理一致性约束后的双流 view 已稳定导出，不是一次性手工拼接样例”。

## 本结论不能支撑什么

- 不能单独支撑“F(full) 在全部指标上显著优于 E baseline”这类过强表述。
- 不能替代公开数据上的泛化结论，也不能替代鼎新 weak-label 任务上的最优性结论。
