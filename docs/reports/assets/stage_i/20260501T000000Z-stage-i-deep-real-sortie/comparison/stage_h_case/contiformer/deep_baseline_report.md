# Stage I Deep Baseline - stage_h_case - contiformer

- profile: `real_sortie_v1`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/comparison/stage_h_case/contiformer`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/stage_h_case_sequences`

## Real Sortie Summary

- view count: `3`
- sample count: `24`
- smoke training target: `projection_diagnostics_verdict_code`

| view | sortie | pilot | verdict | samples | stability | attention entropy | top concentration | event-mask interference |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01` | 10033 | `PASS` | 8 | 0.971358 | 2.749520 | 0.093194 | 0.061620 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10035 | `PASS` | 8 | 0.985847 | 2.767265 | 0.076540 | 0.031078 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | `WARN` | 8 | 0.967130 | 2.771659 | 0.065589 | 0.123053 |

## Same-Sortie Dual-Pilot Delta

| sortie | reference pilot | comparison pilot | delta stability | delta entropy | delta top concentration | delta event-mask interference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | 10035 | +0.018718 | -0.004393 | +0.010951 | -0.091975 |
