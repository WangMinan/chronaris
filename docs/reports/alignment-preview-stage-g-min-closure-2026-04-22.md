# Stage G Minimal Causal Fusion Comparison

- baseline: `F baseline (full)`
- candidate: `F+G(min)`
- fusion state source: `hidden`
- sample count: `25`
- split: train `15`, validation `5`, test `5`
- vehicle metadata status: `loaded`
- vehicle metadata fields: `96`

| metric | F baseline | F+G(min) |
| --- | ---: | ---: |
| final train total | 1.839854 | 1.839854 |
| final validation total | 1.672444 | 1.672444 |
| test total | 1.769787 | 1.769787 |
| test physics total | 1.539611 | 1.539611 |
| threshold verdict | PASS | PASS |

## Stage G Attention Summary

| metric | value |
| --- | ---: |
| exported fusion samples | 3 |
| reference point count | 16 |
| state dim | 32 |
| fused dim | 96 |
| mean attention entropy | 0.931446 |
| mean max attention | 0.225623 |
| mean causal option count | 8.500000 |
| mean top event score | 1.000000 |
| mean top contribution score | 2.609973 |

## Decision

- Stage G minimal candidate verdict: `PASS` (F baseline `PASS`)
- Stage G is considered active when causal fusion exports non-empty attention and contribution artifacts

## Reports

- F baseline report: `/home/wangminan/projects/chronaris/docs/reports/alignment-preview-stage-g-min-closure-2026-04-22-stage-f-baseline.md`
- F+G(min) report: `/home/wangminan/projects/chronaris/docs/reports/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min.md`

## Diagnostic Artifacts

- fusion summary json: `/home/wangminan/projects/chronaris/docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/causal_fusion_summary.json`
- fusion samples csv: `/home/wangminan/projects/chronaris/docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/causal_fusion_samples.csv`
