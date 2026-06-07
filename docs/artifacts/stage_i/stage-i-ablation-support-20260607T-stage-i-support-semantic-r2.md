# Stage I Ablation Support - 20260607T-stage-i-support-semantic-r2

- generated_at_utc: `2026-06-07T10:56:06.555406Z`
- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2`
- main_ablation_matrix: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/ablation_matrix.csv`
- overview_plot: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_overview.png`
- 固定六路径主矩阵：`e_baseline / f_full / g_min / g_no_causal_mask / vehicle_delta_suppressed / no_event_bias`

| variant | source | sample_count | proj_cosine | proj_l2_gap | views | attention_entropy | top_event | top_contribution | delta_top_contribution | private_t1_macro_f1 | private_t2_rmse | private_t3_top1 | supports | limits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `E baseline` | `projection_diagnostics_summary` | 3.0 | 0.760173 | 0.211933 | 0.0 | - | - | - | - | - | - | - | 对齐预览存在 | 不含稳定导出与因果解释 |
| `F(full)` | `projection_diagnostics_summary + stage_h_run_manifest` | 3.0 | 0.699514 | 0.212016 | 3.0 | - | - | - | - | - | - | - | 稳定导出与双 pilot 可读性 | 不直接给出因果 ablation 胜负 |
| `G(min)` | `causal_fusion_summary + phase2_case_study` | 111.0 | 0.705586 | 0.101908 | 3.0 | 0.930686 | 1.000000 | 2.399792 | +0.000000 | - | - | - | 非对称注意力与双 pilot 差异可读 | 不等价于任务级 superiority |
| `G(no causal mask)` | `chronaris_opt_no_causal_mask_private_proxy` | - | - | - | - | - | - | - | - | 0.173333 | 313.232477 | 0.027027 | 去掉因果掩码后三任务同步退化 | 当前来自私有 proxy，不直接等价于公开 benchmark |
| `vehicle_delta_suppressed` | `phase2_case_study_bundle_only` | 3.0 | 0.705586 | 0.101908 | 3.0 | 0.937500 | 0.000000 | 0.000000 | -2.399792 | - | - | - | 事件/机动敏感性可读 | 仅是 frozen Stage H view 上的 bundle-only 干预 |
| `no_event_bias` | `phase2_case_study_bundle_only` | 3.0 | 0.705586 | 0.101908 | 3.0 | 0.932829 | 1.000000 | 2.126160 | -0.273633 | - | - | - | 事件/机动敏感性可读 | 仅是 frozen Stage H view 上的 bundle-only 干预 |

## 六路径矩阵怎么读

1. `e_baseline -> f_full -> g_min` 给出从对齐到导出再到最小因果融合的主链路。
2. `g_no_causal_mask` 是任务级反证，回答“如果掩码不存在会怎样”。
3. `vehicle_delta_suppressed / no_event_bias` 是 case-study 扰动证据，回答“事件与机动上下文到底有没有被用到”。
4. 这 6 条路径合在一起，可以支撑论文里的 `alignment / export / causal / ablation` 证据主矩阵。
