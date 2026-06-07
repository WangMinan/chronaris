# Stage I Deep Baseline - stage_h_case - contiformer

- profile: `real_sortie_v1`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/stage_h_case/contiformer`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/stage_h_case_sequences`

## Real Sortie Summary

- view count: `3`
- sample count: `24`
- smoke training target: `projection_diagnostics_verdict_code`

| view | sortie | pilot | verdict | samples | stability | attention entropy | top concentration | event-mask interference |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01` | 10033 | `PASS` | 8 | 0.926563 | 2.770962 | 0.067093 | 0.115253 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10035 | `PASS` | 8 | 0.966087 | 2.743656 | 0.092428 | 0.103820 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | `WARN` | 8 | 0.950226 | 2.762621 | 0.085484 | 0.123210 |

## Same-Sortie Dual-Pilot Delta

| sortie | reference pilot | comparison pilot | delta stability | delta entropy | delta top concentration | delta event-mask interference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | 10035 | +0.015861 | -0.018965 | +0.006944 | -0.019390 |
