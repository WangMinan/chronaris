# Stage H Export v1 - 20260427T000000Z-stage-h-closure

- export version: `stage-h-v1`
- export profile: `validation`
- generated at UTC: `2026-04-27T13:41:48.885541+00:00`
- artifact root: `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure`
- run manifest: `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json`
- report path: `docs/artifacts/stage_h/stage-h-closure-2026-04-27.md`
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
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | 37 | 37 | `PASS` | `enabled` | `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/sorties/20251005_四01_ACT-4_云_J20_22#01/views/20251005_四01_ACT-4_云_J20_22#01__pilot_10033/feature_bundle.npz` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | 37 | 37 | `PASS` | `enabled` | `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035/feature_bundle.npz` |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | 37 | 37 | `WARN` (mean_projection_cosine, projection_cosine_cv, projection_l2_gap_cv) | `enabled` | `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033/feature_bundle.npz` |

## Diagnostics Warnings

- `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033`: mean_projection_cosine=0.563622 >= 0.650000; projection_cosine_cv=0.298227 <= 0.150000; projection_l2_gap_cv=0.558502 <= 0.250000
- 该 `WARN` 表示投影诊断阈值提醒，不表示 view 包导出失败；对应 feature bundle、window manifest、projection diagnostics 和 Stage G 摘要均已生成。

## Partial Data

- manifest path: `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/partial_data/partial_data_manifest.jsonl`
- raw window manifest and feature bundle were pruned from docs/LFS on 2026-06-19; see `docs/artifacts/cleanup/20260619-lfs-docs-prune.md`.
- entry count: `1`
- built entry count: `1`
- skipped entry count: `0`

## Closure Evidence

- `load_stage_h_feature_run()` 已读取 run manifest 中的 `3` 个 view。
- 三个 view 的 `fused_representation.shape` 均为 `(8, 16, 96)`。
- 已清理的 `vehicle_only_window_manifest.jsonl` 行数为 `1478`。
- 已清理的 `vehicle_only_feature_bundle.npz` 的 `values.shape` 为 `(1478, 105, 823)`。
- partial-data 构建使用 Flux 侧 `5s window + 每字段最多 32 点` 限流，避免一次性拉取全天原始点。
