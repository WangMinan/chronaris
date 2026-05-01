# Stage I Deep Baseline - stage_h_case - mult

- profile: `real_sortie_v1`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/comparison/stage_h_case/mult`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/stage_h_case_sequences`

## Real Sortie Summary

- view count: `3`
- sample count: `24`
- smoke training target: `projection_diagnostics_verdict_code`

| view | sortie | pilot | verdict | samples | stability | attention entropy | top concentration | event-mask interference |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01` | 10033 | `PASS` | 8 | 0.974267 | 2.754593 | 0.095305 | 0.129090 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10035 | `PASS` | 8 | 0.952978 | 2.767657 | 0.077386 | 0.095135 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | `WARN` | 8 | 0.944533 | 2.759104 | 0.076756 | 0.132576 |

## Same-Sortie Dual-Pilot Delta

| sortie | reference pilot | comparison pilot | delta stability | delta entropy | delta top concentration | delta event-mask interference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | 10035 | +0.008445 | +0.008553 | +0.000630 | -0.037441 |
