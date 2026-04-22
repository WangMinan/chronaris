# Stage F Full Physics Comparison

- baseline normalization: `zscore_train`
- physics family: `full`
- physics mode: `feature_first_with_latent_fallback`
- sample count: `25`
- split: train `15`, validation `5`, test `5`
- vehicle metadata status: `loaded`
- vehicle metadata fields: `96`

| metric | E baseline | E+F(full) |
| --- | ---: | ---: |
| final train total | 1.559523 | 1.839854 |
| final validation total | 1.460789 | 1.672444 |
| test total | 1.561467 | 1.769787 |
| test physics total | 0.000000 | 1.539611 |
| threshold verdict | PASS | PASS |

## Test Physics Components

| component | E baseline | E+F(full) |
| --- | ---: | ---: |
| physiology_envelope | 0.000000 | 0.000000 |
| physiology_latent | 0.000000 | 0.170024 |
| physiology_pairwise | 0.000000 | 0.042706 |
| physiology_smoothness | 0.000000 | 0.457301 |
| physiology_spo2_delta | 0.000000 | 0.000000 |
| vehicle_envelope | 0.000000 | 0.000000 |
| vehicle_latent | 0.000000 | 0.070477 |
| vehicle_semantic | 0.000000 | 0.752882 |
| vehicle_smoothness | 0.000000 | 0.046221 |

## Decision

- Stage F closure candidate verdict: `PASS` (baseline `PASS`)
- physics constraints are considered active when at least one E+F(full) component is non-zero

## Reports

- E baseline report: `/home/wangminan/projects/chronaris/docs/reports/alignment-preview-stage-f-closure-2026-04-22-e-baseline.md`
- E+F(full) report: `/home/wangminan/projects/chronaris/docs/reports/alignment-preview-stage-f-closure-2026-04-22-stage-f-full.md`
