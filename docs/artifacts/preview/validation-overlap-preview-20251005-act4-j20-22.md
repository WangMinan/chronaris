# Overlap Validation Preview - 20251005 ACT-4 J20 22#01

## Purpose

This note records the follow-up Stage C finding after the first preview validation.

The first preview report showed a gap between physiology and vehicle streams, but that preview was built from heavily limited points near the head of each stream. This document records the result of:

1. a lightweight full-coverage probe, and
2. an overlap-focused preview loader run.

## Full Coverage Probe

### Physiology

- bucket: `physiological_input`
- tag filter: `collect_task_id=2100448`, `pilot_id=10033`
- distinct measurements found: `11`
- first observed physiology time: `2025-10-05T01:10:00Z`
- last observed physiology time: `2025-10-05T06:45:40Z`

### Vehicle

- bucket: `bus`
- measurement: `BUS6000019110020`
- tag filter: `sortie_number=20251005_四01_ACT-4_云_J20_22#01`
- first observed vehicle time: `2025-10-05T01:35:00Z`
- last observed vehicle time: `2025-10-05T01:38:00.764Z`

## Conclusion

The full data does contain real overlap.

That means:

- the earlier "no overlap" conclusion was caused by preview sampling strategy,
- not by the underlying sortie data itself.

The true overlap interval is at least:

- `2025-10-05T01:35:00Z`
- to `2025-10-05T01:38:00.764Z`

## Overlap-Focused Preview Run

Focused preview configuration:

- physiology measurements: `eeg`, `spo2`
- physiology point limit per measurement: `500`
- bus point limit: `500`
- query clip window:
  - start: `2025-10-05T01:35:00Z`
  - stop: `2025-10-05T01:38:01Z`

Result:

- physiology points: `681`
- vehicle points: `500`
- aligned reference time: `2025-10-05T01:35:00Z`

Window trial:

- duration: `5000 ms`
- stride: `5000 ms`
- generated joint windows: `25`
- first window offsets: `0 -> 5000`
- first window point counts:
  - physiology: `25`
  - vehicle: `20`

## Engineering Meaning

This result is enough to support the next move:

1. Stage C can stop treating "stream overlap" as the primary blocker.
2. The current blocker shifts to defining a stable minimal experiment input.
3. The repo is ready to move toward `E0` using an overlap-focused preview path.
