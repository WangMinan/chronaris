# Stage I Anchor Report - 20260506T165435Z-stage-i-anchor

- generated_at_utc: `2026-05-06T16:55:22.667722Z`
- stage_h_run_manifest_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json`
- anchor_windows_csv_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_anchor/20260506T165435Z-stage-i-anchor/anchor_windows.csv`

## Overview

- selected_view_count: `3`
- selected_anchor_count: `9`
- view_verdict_counts: `{'PASS': 2, 'WARN': 1}`

## Selection Policy

- top_k_windows_per_view: `3`
- view_verdict_filter: `all`
- anchor_score_formula: `top_contribution + 0.5*top_event + 0.25*abs(pair_delta_top_contribution) + 0.1*abs(pair_delta_projection_cosine) + warn_bonus`

## View Summary

| view | verdict | strongest ablation | strongest delta top contribution | strongest delta top event | paired abs delta cosine | paired abs delta contribution |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `PASS` | `vehicle_delta_suppressed` | -2.124254 | -1.000000 | 0.000000 | 0.000000 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `PASS` | `vehicle_delta_suppressed` | -2.646300 | -1.000000 | 0.170068 | 0.217477 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `WARN` | `vehicle_delta_suppressed` | -2.428823 | -1.000000 | 0.170068 | 0.217477 |

## Anchor Windows

| rank | view | sample | window index | verdict | anchor score | top event | top contribution | strongest ablation |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01:0036` | 36 | `WARN` | 3.317415 | 1.000000 | 2.496039 | `vehicle_delta_suppressed` |
| 2 | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01:0029` | 29 | `WARN` | 3.240597 | 1.000000 | 2.419221 | `vehicle_delta_suppressed` |
| 3 | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01:0030` | 30 | `WARN` | 3.240597 | 1.000000 | 2.419221 | `vehicle_delta_suppressed` |
| 4 | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01:0029` | 29 | `PASS` | 3.234835 | 1.000000 | 2.663459 | `vehicle_delta_suppressed` |
| 5 | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01:0030` | 30 | `PASS` | 3.234835 | 1.000000 | 2.663459 | `vehicle_delta_suppressed` |
| 6 | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01:0031` | 31 | `PASS` | 3.234835 | 1.000000 | 2.663459 | `vehicle_delta_suppressed` |
| 7 | `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01:0029` | 29 | `PASS` | 2.700609 | 1.000000 | 2.200609 | `vehicle_delta_suppressed` |
| 8 | `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01:0030` | 30 | `PASS` | 2.700609 | 1.000000 | 2.200609 | `vehicle_delta_suppressed` |
| 9 | `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01:0031` | 31 | `PASS` | 2.700609 | 1.000000 | 2.200609 | `vehicle_delta_suppressed` |

## Paired-Pilot Comparison

| sortie | reference view | comparison view | delta mean cosine | delta top contribution |
| --- | --- | --- | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | -0.170068 | -0.217477 |

## Private No-Mask Comparison

| task | target metrics | no-mask metrics | target beats no-mask |
| --- | --- | --- | --- |
| `T1_maneuver_intensity_class` | `macro_f1=1.000000, balanced_accuracy=1.000000` | `macro_f1=0.173333, balanced_accuracy=0.333333` | `True` |
| `T2_next_window_physiology_response` | `rmse=201.489565, mae=113.851926` | `rmse=313.232477, mae=173.648719` | `True` |
| `T3_paired_pilot_window_retrieval` | `top1_accuracy=1.000000, mrr=1.000000` | `top1_accuracy=0.027027, mrr=0.113556` | `True` |
