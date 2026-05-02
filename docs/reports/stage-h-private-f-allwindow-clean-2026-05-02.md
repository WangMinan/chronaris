# Stage H Export v1 - 20260502T092753Z-stage-h-f-allwindow-clean

- export version: `stage-h-v1`
- export profile: `validation`
- generated at UTC: `2026-05-02T09:41:46.186501+00:00`
- artifact root: `docs/reports/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean`
- run manifest: `docs/reports/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
- report path: `docs/reports/stage-h-private-f-allwindow-clean-2026-05-02.md`
- sortie count: `2`
- generated view count: `3`
- generated view ids: `20251005_四01_ACT-4_云_J20_22#01__pilot_10033, 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035, 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033`

## Frozen Config

- input normalization: `zscore_train`
- physics constraints enabled: `True`
- physics constraint family: `full`
- causal fusion enabled: `False`
- causal fusion state source: `hidden`
- intermediate partition: `all`
- window duration ms: `5000`
- window stride ms: `5000`
- physiology point limit per measurement: `None`
- vehicle point limit per measurement: `None`
- point limit note: `no per-measurement point cap`

## Sortie Summary

| sortie | pilots | physiology availability | vehicle family | views |
| --- | --- | --- | --- | --- |
| `20251005_四01_ACT-4_云_J20_22#01` | `10033` | `eeg, spo2, tshirt_ecg_accel_gyro, tshirt_heartrate, tshirt_hrv, tshirt_resp, tshirt_respiratory_rate, tshirt_temp, wristband_gsr, wristband_ppg_accel, wristband_spo2` | `BUS6000019110015, BUS6000019110016, BUS6000019110017, BUS6000019110018, BUS6000019110019, BUS6000019110020` | `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` |
| `20251002_单01_ACT-8_翼云_J16_12#01` | `10035, 10033` | `eeg, spo2, tshirt_ecg_accel_gyro, tshirt_heartrate, tshirt_hrv, tshirt_resp, tshirt_respiratory_rate, tshirt_temp, wristband_gsr, wristband_ppg_accel, wristband_spo2` | `BUS6000019110021, BUS6000019110022, BUS6000019110023, BUS6000019110024, BUS6000019110025, BUS6000019110026` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035, 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` |

## View Packages

| view | windows | model samples | diagnostics | stage G | feature bundle |
| --- | ---: | ---: | --- | --- | --- |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | 37 | 37 | `WARN` (mean_projection_cosine) | `disabled` | `docs/reports/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/sorties/20251005_四01_ACT-4_云_J20_22#01/views/20251005_四01_ACT-4_云_J20_22#01__pilot_10033/feature_bundle.npz` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | 37 | 37 | `PASS` | `disabled` | `docs/reports/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035/feature_bundle.npz` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | 37 | 37 | `WARN` (projection_cosine_cv, projection_l2_gap_cv) | `disabled` | `docs/reports/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033/feature_bundle.npz` |

## Diagnostics Warnings

- `20251005_四01_ACT-4_云_J20_22#01__pilot_10033`: mean_projection_cosine=0.625065 >= 0.650000
- `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033`: projection_cosine_cv=0.156733 <= 0.150000; projection_l2_gap_cv=0.323991 <= 0.250000
