# Stage I Ablation Support - 20260506T120000Z-stage-i-support

- generated_at_utc: `2026-05-06T02:58:16.462491Z`
- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support`
- machine summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/support_summary.json`
- main matrix: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/ablation_matrix.csv`
- overview plot: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/support_overview.png`

## 固定六路径主矩阵

| variant | source | mean projection cosine | export views | mean attention entropy | mean top event | mean top contribution | delta top event | delta top contribution | private T1 macro-F1 | private T2 RMSE | private T3 top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `E baseline` | `projection_diagnostics_summary` | 0.760173 | 0.000000 | - | - | - | - | - | - | - | - |
| `F(full)` | `projection_diagnostics_summary + stage_h_run_manifest` | 0.699514 | 3.000000 | - | - | - | - | - | - | - | - |
| `G(min)` | `causal_fusion_summary + phase2_case_study` | 0.705586 | 3.000000 | 0.930686 | 1.000000 | 2.399792 | +0.000000 | +0.000000 | - | - | - |
| `G(no causal mask)` | `chronaris_opt_no_causal_mask_private_proxy` | - | - | - | - | - | - | - | 0.173333 | 313.232477 | 0.027027 |
| `vehicle_delta_suppressed` | `phase2_case_study_bundle_only` | 0.705586 | 3.000000 | 0.937500 | 0.000000 | 0.000000 | -1.000000 | -2.399792 | - | - | - |
| `no_event_bias` | `phase2_case_study_bundle_only` | 0.705586 | 3.000000 | 0.932829 | 1.000000 | 2.126160 | +0.000000 | -0.273633 | - | - | - |

## 中文结论

1. 去掉双流连续对齐后，只剩 `E baseline` 级预览证据；它能说明预览存在，但不能替代稳定 export 与下游消费闭环。
2. 保留 `F(full)` 与 `G(min)` 后，Stage H/Phase 2 已形成 `3` 个真实双流 view、`2 PASS + 1 WARN` 的可解释证据链。
3. 去掉因果掩码后，`chronaris_opt_no_causal_mask` 在私有 T1/T2/T3 三任务同时退化，说明因果掩码对当前主线不是装饰项。
4. 去掉关键事件偏置时，`no_event_bias` 的 mean top contribution 下降；压制 vehicle delta 时，`vehicle_delta_suppressed` 的干预幅度最大，说明事件与机动变化都是当前融合读数的重要支撑。

## 本结论能支撑什么

- 可以直接回答“去掉双流 / 去掉因果掩码 / 去掉关键事件偏置后会怎样”。
- 可以把 `E/F/G/H + Phase 2 + private no-mask` 收束成论文第三阶段可复述的一张主矩阵。

## 本结论不能支撑什么

- 不能把这张主矩阵写成公开 benchmark 的统一对照结论；公开数据仍应由 `chronaris public opt` 与历史 classical / MulT / ContiFormer 报告负责。
- `no_state_normalization` 只保留在 appendix 级 support matrix 中，不进入主矩阵。
