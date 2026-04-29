# Stage I Phase 2 Case Study - 20260429T000000Z-stage-i-phase2-case-study

- generated at UTC: `2026-04-29T03:26:29.113999Z`
- Stage H run manifest: `artifacts/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json`
- artifact root: `artifacts/stage_i/20260429T000000Z-stage-i-phase2-case-study`
- report path: `docs/reports/stage-i-case-study-phase2-2026-04-29.md`
- top-k windows per view: `5`

## Fixed Ablation Family

1. `projection_refusion_baseline`
2. `no_event_bias`
3. `no_state_normalization`
4. `vehicle_delta_suppressed`

## View Summary

- `hidden-vs-projection cosine` 使用 fused L2 norm profile 的 cosine；原因是当前 Stage H 导出的 hidden fused 为 `96` 维，而 projection rerun baseline 为 `48` 维，不能直接做逐向量余弦。

| view | verdict | windows | case samples | mean cosine | cosine cv | l2 gap | l2 gap cv | mean attention entropy | mean top contribution | hidden-vs-projection cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `PASS` | 37 | 8 | 0.819445 | 0.087503 | 0.100333 | 0.206975 | 0.929764 | 2.124254 | 0.758325 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `PASS` | 37 | 8 | 0.733690 | 0.105452 | 0.108193 | 0.166338 | 0.934539 | 2.646300 | 0.771731 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `WARN` | 37 | 8 | 0.563622 | 0.298227 | 0.097198 | 0.558502 | 0.927755 | 2.428823 | 0.767319 |

## Same-Sortie Pilot Comparison

| sortie | reference view | comparison view | delta mean cosine | delta cosine cv | delta l2 gap | delta l2 gap cv | delta attention entropy | delta top contribution | delta hidden/projection cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | -0.170068 | +0.192775 | -0.010995 | +0.392164 | -0.006784 | -0.217476 | -0.004412 |

## Ablation Summary

### 20251005_四01_ACT-4_云_J20_22#01__pilot_10033

| ablation | mean attention entropy | delta entropy | mean max attention | delta max attention | mean top event | delta top event | mean top contribution | delta top contribution | mean fused L2 | delta fused L2 | cosine to projection baseline | delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `projection_refusion_baseline` | 0.929764 | +0.000000 | 0.232805 | +0.000000 | 1.000000 | +0.000000 | 2.124254 | +0.000000 | 1.955358 | +0.000000 | 1.000000 | +0.000000 |
| `no_event_bias` | 0.932657 | +0.002893 | 0.230288 | -0.002517 | 1.000000 | +0.000000 | 1.936862 | -0.187392 | 1.954349 | -0.001009 | 0.999925 | -0.000075 |
| `no_state_normalization` | 0.927463 | -0.002301 | 0.235615 | +0.002810 | 1.000000 | +0.000000 | 2.095528 | -0.028726 | 1.953442 | -0.001916 | 0.999967 | -0.000033 |
| `vehicle_delta_suppressed` | 0.937500 | +0.007736 | 0.211296 | -0.021509 | 0.000000 | -1.000000 | 0.000000 | -2.124254 | 2.395143 | +0.439785 | 0.858288 | -0.141712 |

### 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035

| ablation | mean attention entropy | delta entropy | mean max attention | delta max attention | mean top event | delta top event | mean top contribution | delta top contribution | mean fused L2 | delta fused L2 | cosine to projection baseline | delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `projection_refusion_baseline` | 0.934539 | +0.000000 | 0.228140 | +0.000000 | 1.000000 | +0.000000 | 2.646300 | +0.000000 | 1.992961 | +0.000000 | 1.000000 | +0.000000 |
| `no_event_bias` | 0.937162 | +0.002623 | 0.217219 | -0.010922 | 1.000000 | +0.000000 | 2.337433 | -0.308867 | 1.989666 | -0.003295 | 0.999924 | -0.000076 |
| `no_state_normalization` | 0.930550 | -0.003989 | 0.239459 | +0.011319 | 1.000000 | +0.000000 | 2.809703 | +0.163403 | 1.995829 | +0.002869 | 0.999932 | -0.000068 |
| `vehicle_delta_suppressed` | 0.937500 | +0.002961 | 0.211296 | -0.016845 | 0.000000 | -1.000000 | 0.000000 | -2.646300 | 2.059054 | +0.066093 | 0.896225 | -0.103775 |

### 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033

| ablation | mean attention entropy | delta entropy | mean max attention | delta max attention | mean top event | delta top event | mean top contribution | delta top contribution | mean fused L2 | delta fused L2 | cosine to projection baseline | delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `projection_refusion_baseline` | 0.927755 | +0.000000 | 0.237629 | +0.000000 | 1.000000 | +0.000000 | 2.428823 | +0.000000 | 1.581462 | +0.000000 | 1.000000 | +0.000000 |
| `no_event_bias` | 0.928669 | +0.000914 | 0.237132 | -0.000497 | 1.000000 | +0.000000 | 2.104184 | -0.324640 | 1.578037 | -0.003425 | 0.999900 | -0.000100 |
| `no_state_normalization` | 0.930713 | +0.002958 | 0.232430 | -0.005200 | 1.000000 | +0.000000 | 2.513583 | +0.084760 | 1.585092 | +0.003630 | 0.999908 | -0.000092 |
| `vehicle_delta_suppressed` | 0.937500 | +0.009745 | 0.211296 | -0.026334 | 0.000000 | -1.000000 | 0.000000 | -2.428823 | 1.994985 | +0.413523 | 0.793179 | -0.206821 |

## Top Windows

### 20251005_四01_ACT-4_云_J20_22#01__pilot_10033

| sample | window index | start ms | end ms | top event offset s | top event score | top contribution offset s | top contribution score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01:0029` | 29 | 145000 | 150000 | 0.666667 | 1.000000 | 0.333333 | 2.200609 |
| `20251005_四01_ACT-4_云_J20_22#01:0030` | 30 | 150000 | 155000 | 0.666667 | 1.000000 | 0.333333 | 2.200609 |
| `20251005_四01_ACT-4_云_J20_22#01:0031` | 31 | 155000 | 160000 | 0.666667 | 1.000000 | 0.333333 | 2.200609 |
| `20251005_四01_ACT-4_云_J20_22#01:0032` | 32 | 160000 | 165000 | 0.666667 | 1.000000 | 0.333333 | 2.200609 |
| `20251005_四01_ACT-4_云_J20_22#01:0033` | 33 | 165000 | 170000 | 0.666667 | 1.000000 | 0.333333 | 2.200609 |

### 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035

| sample | window index | start ms | end ms | top event offset s | top event score | top contribution offset s | top contribution score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01:0029` | 29 | 145000 | 150000 | 0.333333 | 1.000000 | 0.333333 | 2.663459 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0030` | 30 | 150000 | 155000 | 0.333333 | 1.000000 | 0.333333 | 2.663459 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0031` | 31 | 155000 | 160000 | 0.333333 | 1.000000 | 0.333333 | 2.663459 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0032` | 32 | 160000 | 165000 | 0.333333 | 1.000000 | 0.333333 | 2.663459 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0033` | 33 | 165000 | 170000 | 0.333333 | 1.000000 | 0.333333 | 2.663459 |

### 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033

| sample | window index | start ms | end ms | top event offset s | top event score | top contribution offset s | top contribution score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01:0036` | 36 | 180000 | 180991 | 0.066067 | 1.000000 | 0.066067 | 2.496039 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0029` | 29 | 145000 | 150000 | 0.333333 | 1.000000 | 0.333333 | 2.419221 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0030` | 30 | 150000 | 155000 | 0.333333 | 1.000000 | 0.333333 | 2.419221 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0031` | 31 | 155000 | 160000 | 0.333333 | 1.000000 | 0.333333 | 2.419221 |
| `20251002_单01_ACT-8_翼云_J16_12#01:0032` | 32 | 160000 | 165000 | 0.333333 | 1.000000 | 0.333333 | 2.419221 |

## WARN View Interpretation

### 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033

- 该视图 `verdict=WARN`。
- 触发的阈值项：mean_projection_cosine=0.563622 >= 0.650000, projection_cosine_cv=0.298227 <= 0.150000, projection_l2_gap_cv=0.558502 <= 0.250000。
- 同 sortie 对比显示：projection cosine 差值 `-0.170068`，projection cosine CV 差值 `+0.192775`，projection L2 gap CV 差值 `+0.392164`。
- 在 bundle-only 消融中，对 top contribution 影响最大的路径为 `vehicle_delta_suppressed`，delta=`-2.428823`。
- 这说明该视图不是导出失败，而是对投影一致性与事件驱动变化更敏感，需要在 Phase 3 与第二公开数据集结果一起综合解释。

## Machine Artifacts

- summary json: `artifacts/stage_i/20260429T000000Z-stage-i-phase2-case-study/case_study_summary.json`
- view summary csv: `artifacts/stage_i/20260429T000000Z-stage-i-phase2-case-study/view_summary.csv`
- ablation summary csv: `artifacts/stage_i/20260429T000000Z-stage-i-phase2-case-study/ablation_summary.csv`
- window rankings csv: `artifacts/stage_i/20260429T000000Z-stage-i-phase2-case-study/window_rankings.csv`

