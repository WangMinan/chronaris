# Alignment Preview - 20251005_四01_ACT-4_云_J20_22#01

## Sample Summary

- sample count: `25`
- max physiology feature count: `12`
- max vehicle feature count: `21`

## Split Summary

- train: `15`
- validation: `5`
- test: `5`
- skipped between train/validation: `0`
- skipped between validation/test: `0`

## Final Train Metrics

- physiology reconstruction: `1.060884`
- vehicle reconstruction: `1.203194`
- reconstruction total: `2.264078`
- alignment: `0.225833`
- vehicle physics: `0.004309`
- physiology physics: `11088.632031`
- physics total: `11088.636328`
- total: `1111.353581`

## Final Validation Metrics

- physiology reconstruction: `0.734154`
- vehicle reconstruction: `0.989305`
- reconstruction total: `1.723459`
- alignment: `0.094877`
- vehicle physics: `0.006274`
- physiology physics: `11088.239258`
- physics total: `11088.245117`
- total: `1110.642944`

## Reference Intermediate Export

- partition: `test`
- exported sample count: `3`
- reference point count: `16`
- exported sample ids: `20251005_四01_ACT-4_云_J20_22#01:0020, 20251005_四01_ACT-4_云_J20_22#01:0021, 20251005_四01_ACT-4_云_J20_22#01:0022`
- physiology mean reference projection L2: `1.146591`
- vehicle mean reference projection L2: `1.130940`
- mean cross-stream projection cosine: `0.459597`

## Test Metrics

- physiology reconstruction: `0.758069`
- vehicle reconstruction: `0.989699`
- reconstruction total: `1.747768`
- alignment: `0.094877`
- vehicle physics: `0.009819`
- physiology physics: `11088.649414`
- physics total: `11088.659180`
- total: `1110.708618`

## Physics Constraint Diagnostics

- enabled: `True`
- family: `minimal`
- mode: `feature_first_with_latent_fallback`
- vehicle metadata status: `loaded`
- vehicle metadata fields: `96`

| metric | train | validation | test |
| --- | ---: | ---: | ---: |
| vehicle physics | 0.004309 | 0.006274 | 0.009819 |
| physiology physics | 11088.632031 | 11088.239258 | 11088.649414 |
| physics total | 11088.636328 | 11088.245117 | 11088.659180 |

### Component Breakdown

| component | train | validation | test |
| --- | ---: | ---: | ---: |
| physiology_envelope | 0.000000 | 0.000000 | 0.000000 |
| physiology_latent | 0.000000 | 0.000000 | 0.000000 |
| physiology_pairwise | 0.000000 | 0.000000 | 0.000000 |
| physiology_smoothness | 11088.632031 | 11088.239258 | 11088.649414 |
| physiology_spo2_delta | 0.000000 | 0.000000 | 0.000000 |
| vehicle_envelope | 0.000000 | 0.000000 | 0.000000 |
| vehicle_latent | 0.000000 | 0.000000 | 0.000000 |
| vehicle_rigid_body_rotation | 0.000000 | 0.000000 | 0.000000 |
| vehicle_rigid_body_translation | 0.000000 | 0.000000 | 0.000000 |
| vehicle_rigid_body_vertical | 0.000000 | 0.000000 | 0.000000 |
| vehicle_semantic | 0.004309 | 0.006274 | 0.009819 |
| vehicle_smoothness | 0.000000 | 0.000000 | 0.000000 |



## Sample-Level Projection Diagnostics

- sample count: `3`
- reference point count: `16`
- mean projection cosine: `0.459597`
- min projection cosine: `-0.296839`
- max projection cosine: `0.662065`
- mean projection L2 gap: `0.042234`
- mean projection L2 ratio (vehicle/physiology): `0.985914`
- std projection cosine (cross-sample): `0.000000`
- cv projection cosine (cross-sample): `0.000000`
- std projection L2 gap (cross-sample): `0.000000`
- cv projection L2 gap (cross-sample): `0.000000`
- std projection L2 ratio (cross-sample): `0.000000`
- cv projection L2 ratio (cross-sample): `0.000000`

### Threshold Evaluation

- verdict: `WARN`

| check | actual | operator | expected | result |
| --- | ---: | :---: | ---: | :---: |
| sample_count | 3.000000 | >= | 1.000000 | PASS |
| mean_projection_cosine | 0.459597 | >= | 0.650000 | WARN |
| mean_projection_l2_gap | 0.042234 | <= | 0.250000 | PASS |
| mean_projection_l2_ratio_deviation | 0.014086 | <= | 0.300000 | PASS |
| projection_cosine_cv | 0.000000 | <= | 0.150000 | PASS |
| projection_l2_gap_cv | 0.000000 | <= | 0.250000 | PASS |

| sample id | mean cosine | min cosine | max cosine | mean L2 gap | mean L2 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20251005_四01_ACT-4_云_J20_22#01:0020 | 0.459597 | -0.296839 | 0.662065 | 0.042234 | 0.985914 |
| 20251005_四01_ACT-4_云_J20_22#01:0021 | 0.459597 | -0.296839 | 0.662065 | 0.042234 | 0.985914 |
| 20251005_四01_ACT-4_云_J20_22#01:0022 | 0.459597 | -0.296839 | 0.662065 | 0.042234 | 0.985914 |

## Visual Artifacts

### Train/Validation Total Loss

![Train/Validation Total Loss](assets/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r1/train_validation_total_loss.png)

### Train/Validation Alignment Loss

![Train/Validation Alignment Loss](assets/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r1/train_validation_alignment_loss.png)

### Per-Stream Reconstruction Loss

![Per-Stream Reconstruction Loss](assets/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r1/reconstruction_stream_loss.png)

### Train/Validation Physics Loss

![Train/Validation Physics Loss](assets/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r1/train_validation_physics_loss.png)

### Final Physics Constraint Components

![Final Physics Constraint Components](assets/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r1/constraint_component_breakdown.png)

### Reference Projection Cosine

![Reference Projection Cosine](assets/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r1/reference_projection_cosine.png)
