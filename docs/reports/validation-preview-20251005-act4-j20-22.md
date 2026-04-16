# Preview Validation - 20251005 ACT-4 J20 22#01

- sortie: `20251005_四01_ACT-4_云_J20_22#01`
- physiology points: `100` across `2` measurements
- vehicle points: `50` across `1` measurements

## Stream Coverage

### `physiology`
- points: `100`
- measurements: `2`
- first timestamp: `2025-10-05T01:10:00+00:00`
- last timestamp: `2025-10-05T01:10:49+00:00`
- span ms: `49000`
- measurement `eeg`: `50` points, `2025-10-05T01:10:00+00:00` -> `2025-10-05T01:10:12.250000+00:00`
- measurement `spo2`: `50` points, `2025-10-05T01:10:00+00:00` -> `2025-10-05T01:10:49+00:00`

### `vehicle`
- points: `50`
- measurements: `1`
- first timestamp: `2025-10-05T01:35:00+00:00`
- last timestamp: `2025-10-05T01:35:12.254000+00:00`
- span ms: `12254`
- measurement `BUS6000019110020`: `50` points, `2025-10-05T01:35:00+00:00` -> `2025-10-05T01:35:12.254000+00:00`

## Cross-Stream Timing

- relation: `physiology_before_vehicle`
- gap duration ms: `1451000`
- leading stream: `physiology`

## Window Trials

- duration `5000` ms / stride `5000` ms -> `0` windows
- duration `30000` ms / stride `30000` ms -> `0` windows
- duration `300000` ms / stride `300000` ms -> `0` windows
- duration `1800000` ms / stride `1800000` ms -> `1` windows
  first window offsets: `0` -> `1512255`

## Notes

- Preview physiology measurements are limited to `eeg` and `spo2`.
- Preview point limits are: physiology `50` rows per measurement, bus `50` rows.
- This report is for Stage C coverage analysis, not final dataset quality certification.
- Under the current preview, small windows do not produce joint samples because the two streams are separated by about 24 minutes.
