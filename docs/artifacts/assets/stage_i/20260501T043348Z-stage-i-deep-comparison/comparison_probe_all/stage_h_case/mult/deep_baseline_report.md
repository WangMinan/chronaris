# Stage I Deep Baseline - stage_h_case - mult

- profile: `real_sortie_v1`
- artifact root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/comparison_probe_all/stage_h_case/mult`
- prepared root: `/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/stage_h_case_sequences`

## Real Sortie Summary

- view count: `3`
- sample count: `24`
- smoke training target: `projection_diagnostics_verdict_code`

| view | sortie | pilot | verdict | samples | stability | attention entropy | top concentration | event-mask interference |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `20251005_四01_ACT-4_云_J20_22#01` | 10033 | `PASS` | 8 | 0.999823 | 2.736039 | 0.080087 | 0.123575 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10035 | `PASS` | 8 | 0.952734 | 2.725721 | 0.117114 | 0.058188 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | `WARN` | 8 | 0.972622 | 2.750155 | 0.104355 | 0.150917 |

## Same-Sortie Dual-Pilot Delta

| sortie | reference pilot | comparison pilot | delta stability | delta entropy | delta top concentration | delta event-mask interference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20251002_单01_ACT-8_翼云_J16_12#01` | 10033 | 10035 | -0.019888 | -0.024434 | +0.012759 | -0.092728 |
