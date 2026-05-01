# Stage I Real-Sortie Deep Comparison

更新时间：2026-05-01

- 机器资产根目录：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie`
- sequence 资产：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/stage_h_case_sequences`
- comparison 资产：`/home/wangminan/projects/chronaris/docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/comparison`

## 1. 范围

本轮只执行增强实验第一批：

1. 输入固定为 Stage H 收口资产 `run_manifest.json`
2. 只跑 `stage_h_case / real_sortie_v1`
3. 对比模型固定为 `MulT` 与 `ContiFormer`
4. `UAB / NASA` 深度 benchmark 本轮尚未实跑

## 2. 输入事实

- `3` 个真实双流 view
- 每个 view 的 `window_count` 都是 `37`
- 导出的 deep sequence 样本数：`24`
- 每个样本长度：`16`
- 双模态输入：
  - `physiology_reference_projection`
  - `vehicle_reference_projection`
- sidecar：
  - `vehicle_event_scores`
  - `attention_weights`
  - `window_manifest`
  - `projection_diagnostics_verdict`

## 3. 结果概览

### 3.1 MulT

| view | verdict | stability | attention entropy | top concentration | event-mask interference |
| --- | --- | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `PASS` | 0.974267 | 2.754593 | 0.095305 | 0.129090 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `PASS` | 0.952978 | 2.767657 | 0.077386 | 0.095135 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `WARN` | 0.944533 | 2.759104 | 0.076756 | 0.132576 |

同 sortie 双 pilot 差异：

- `20251002_单01_ACT-8_翼云_J16_12#01`
- `10035 - 10033`
  - `delta stability = +0.008445`
  - `delta entropy = +0.008553`
  - `delta top concentration = +0.000630`
  - `delta event-mask interference = -0.037441`

### 3.2 ContiFormer

| view | verdict | stability | attention entropy | top concentration | event-mask interference |
| --- | --- | ---: | ---: | ---: | ---: |
| `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` | `PASS` | 0.971358 | 2.749520 | 0.093194 | 0.061620 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035` | `PASS` | 0.985847 | 2.767265 | 0.076540 | 0.031078 |
| `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` | `WARN` | 0.967130 | 2.771659 | 0.065589 | 0.123053 |

同 sortie 双 pilot 差异：

- `20251002_单01_ACT-8_翼云_J16_12#01`
- `10035 - 10033`
  - `delta stability = +0.018718`
  - `delta entropy = -0.004393`
  - `delta top concentration = +0.010951`
  - `delta event-mask interference = -0.091975`

## 4. 当前判断

1. sequence contract 已经能在不回开 MySQL / InfluxDB 临时支线的前提下，直接消费冻结 Stage H 资产。
2. `MulT` 与 `ContiFormer` 两条 wrapper 都已完成真实 sortie 前向、短训练 smoke、view-level 汇总与 dual-pilot 对比。
3. 第一批增强实验已经满足“真实 sortie 优先”的系统级验证入口，但还不能替代后续 `UAB / NASA` 的量化 benchmark。

## 5. 本轮未做

1. 没有把 `3 views` 误写成监督 benchmark。
2. 没有把 `NASA` 改写成 workload。
3. 没有执行 `UAB / NASA` 的全量 LOSO 深度基线。
