# Stage H Export v1 - 20260426T072340Z-stage-h-v1

- export version: `stage-h-v1`
- generated at UTC: `2026-04-26T07:23:40.349020+00:00`
- artifact root: `artifacts/stage_h/20260426T072340Z-stage-h-v1`
- run manifest: `artifacts/stage_h/20260426T072340Z-stage-h-v1/run_manifest.json`
- report path: `docs/reports/stage-h-export-v1-2026-04-26.md`
- sortie count: `2`
- generated view count: `3`
- generated view ids: `20251005_四01_ACT-4_云_J20_22#01__pilot_10033, 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035, 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033`

## Frozen Config

- input normalization: `zscore_train`
- physics constraint family: `full`
- causal fusion enabled: `True`
- causal fusion state source: `hidden`
- intermediate partition: `test`
- window duration ms: `5000`
- window stride ms: `5000`

## Sortie Summary

| sortie | pilots | physiology availability | vehicle family | views |
| --- | --- | --- | --- | --- |
| `20251005_四01_ACT-4_云_J20_22#01` | `10033` | `eeg, spo2, tshirt_ecg_accel_gyro, tshirt_heartrate, tshirt_hrv, tshirt_resp, tshirt_respiratory_rate, tshirt_temp, wristband_gsr, wristband_ppg_accel, wristband_spo2` | `BUS6000019110015, BUS6000019110016, BUS6000019110017, BUS6000019110018, BUS6000019110019, BUS6000019110020` | `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` |
| `20251002_单01_ACT-8_翼云_J16_12#01` | `10035, 10033` | `eeg, spo2, tshirt_ecg_accel_gyro, tshirt_heartrate, tshirt_hrv, tshirt_resp, tshirt_respiratory_rate, tshirt_temp, wristband_gsr, wristband_ppg_accel, wristband_spo2` | `BUS6000019110021, BUS6000019110022, BUS6000019110023, BUS6000019110024, BUS6000019110025, BUS6000019110026` | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035, 20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` |

## View Packages

| view | windows | model samples | diagnostics | stage G | feature bundle |
| --- | ---: | ---: | --- | --- | --- |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | 37 | 37 | `WARN` | `enabled` | `artifacts/stage_h/20260426T072340Z-stage-h-v1/sorties/20251005_四01_ACT-4_云_J20_22#01/views/20251005_四01_ACT-4_云_J20_22#01__pilot_10033/feature_bundle.npz` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | 37 | 37 | `PASS` | `enabled` | `artifacts/stage_h/20260426T072340Z-stage-h-v1/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035/feature_bundle.npz` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | 37 | 37 | `WARN` | `enabled` | `artifacts/stage_h/20260426T072340Z-stage-h-v1/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033/feature_bundle.npz` |

## Partial Data

- manifest path: `artifacts/stage_h/20260426T072340Z-stage-h-v1/partial_data/partial_data_manifest.jsonl`
- window manifest path: `artifacts/stage_h/20260426T072340Z-stage-h-v1/partial_data/vehicle_only_window_manifest.jsonl`
- feature bundle path: `None`
- entry count: `1`
- built entry count: `0`
- skipped entry count: `1`
