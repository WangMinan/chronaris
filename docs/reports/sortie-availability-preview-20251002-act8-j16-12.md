# Sortie Availability Preview - 20251002 ACT-8 J16 12#01

## Purpose

This note records a lightweight Stage H-oriented availability check for a second real sortie:

- sortie: `20251002_单01_ACT-8_翼云_J16_12#01`
- goal: confirm MySQL metadata, dual-pilot physiology streams, and vehicle-side Influx coverage
- scope: availability and engineering fit only; this is not a new Stage E/F/G baseline run

## MySQL Confirmation

- `flight_task_id`: `6000002`
- `flight_batch_id`: `6000002`
- `batch_number`: `20251002_单01`
- `mission_code`: `ACT-8`
- `aircraft_model`: `J16`
- `aircraft_number`: `12`
- `up_pilot_id`: `10035`
- `down_pilot_id`: `10033`
- `source_sortie_id`: `(null)`
- `flight_date`: `2025-10-02`
- `car_start_time`: `2025-10-02 15:45:00`
- `car_end_time`: `2025-10-02 22:00:00`

Associated physiology task:

- `collect_task_id`: `2100450`
- `coding`: `TASK_20251002_01`
- `subject`: `ACT-8`
- `collect_start_time`: `2025-10-02 15:45:00`
- `collect_end_time`: `2025-10-02 22:00:00`

Implication:

- the sortie can resolve physiology context even though `source_sortie_id` is empty
- the current fallback path `collect_task_id + up/down_pilot_id` is required for this sortie

## Influx Confirmation

All timestamps below are UTC.

### Physiology

- bucket: `physiological_input`
- tag filter root: `collect_task_id=2100450`
- pilot ids present: `10033`, `10035`
- distinct measurements found: `11`
- measurement list:
  - `eeg`
  - `spo2`
  - `tshirt_ecg_accel_gyro`
  - `tshirt_heartrate`
  - `tshirt_hrv`
  - `tshirt_resp`
  - `tshirt_respiratory_rate`
  - `tshirt_temp`
  - `wristband_gsr`
  - `wristband_ppg_accel`
  - `wristband_spo2`

Core measurement time ranges:

| pilot_id | measurement | first | last |
| --- | --- | --- | --- |
| `10033` | `eeg` | `2025-10-02T07:45:00Z` | `2025-10-02T23:59:59.75Z` |
| `10033` | `spo2` | `2025-10-02T07:45:00Z` | `2025-10-02T12:21:38Z` |
| `10035` | `eeg` | `2025-10-02T07:45:00Z` | `2025-10-02T23:59:59.75Z` |
| `10035` | `spo2` | `2025-10-02T07:45:00Z` | `2025-10-02T12:21:38Z` |

### Vehicle

- bucket: `bus`
- sortie tag: `sortie_number=20251002_单01_ACT-8_翼云_J16_12#01`
- confirmed MySQL `storage_data_analysis` family:
  - `BUS6000019110021`
  - `BUS6000019110022`
  - `BUS6000019110023`
  - `BUS6000019110024`
  - `BUS6000019110025`
  - `BUS6000019110026`

Observed time ranges:

| measurement | first | last |
| --- | --- | --- |
| `BUS6000019110021` | `2025-10-02T08:35:00Z` | `2025-10-02T10:22:00.468Z` |
| `BUS6000019110022` | `2025-10-02T08:35:00Z` | `2025-10-02T10:38:01.574Z` |
| `BUS6000019110023` | `2025-10-02T08:35:00Z` | `2025-10-02T10:06:21.685Z` |
| `BUS6000019110024` | `2025-10-02T08:35:00Z` | `2025-10-02T10:38:07.823Z` |
| `BUS6000019110025` | `2025-10-02T08:35:00Z` | `2025-10-02T08:40:59.98Z` |
| `BUS6000019110026` | `2025-10-02T08:35:00Z` | `2025-10-02T08:38:00.764Z` |

## Conclusion

This sortie is confirmed as a valid second real-data candidate for Stage H multi-sortie availability work.

Confirmed facts:

1. MySQL metadata is complete enough to resolve `flight_task`, `collect_task`, and dual pilot ids.
2. Influx physiology data exists for both pilots under one shared `collect_task_id`.
3. Influx vehicle data exists as a six-measurement BUS family, not a single measurement.
4. There is real cross-stream overlap. Vehicle data begins at `2025-10-02T08:35:00Z`, while both pilots' `eeg` and `spo2` streams start at `2025-10-02T07:45:00Z`.

## Engineering Meaning

The current single-sortie Stage E/F/G preview baseline remains unchanged and should stay on:

- sortie: `20251005_四01_ACT-4_云_J20_22#01`
- vehicle measurement: `BUS6000019110020`

But this new sortie changes two implementation assumptions for future Stage H work:

1. physiology lookup must support `collect_task_id + pilot_id[]`, not only one hard-coded pilot id
2. bus-side availability and manifesting must support a per-sortie BUS measurement family, not only one fixed measurement id

This turn updated the access layer accordingly:

- physiology live reader now pushes all resolved `pilot_id` values into Flux queries
- MySQL access now exposes a reader that can list all BUS `storage_data_analysis` rows for one sortie

## Notes

- `eeg` currently extends to `2025-10-02T23:59:59.75Z`, which is later than the `collect_task` end time. For future experiment input or feature export, clipping should continue to follow explicit sortie/collect-task windows rather than assume every raw physiology point is flight-effective.
