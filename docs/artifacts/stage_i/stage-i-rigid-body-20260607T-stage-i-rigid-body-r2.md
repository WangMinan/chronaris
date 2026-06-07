# Stage I Rigid-Body Ablation - 20260607T-stage-i-rigid-body-r2

- summary_json: `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json`
- source_sortie: `20251005_四01_ACT-4_云_J20_22#01`
- pilot_id: `10033`
- setting: `epoch_count=1, batch_size=8, input_normalization_mode=zscore_train, device=cpu, strict_mysql_field_labels=true`

## Test Comparison

| family | total | alignment | physics_total | metadata | translation | vertical | rotation | report |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `minimal` | 1110.708618 | 0.094877 | 11088.659180 | `loaded` (96) | 0.000000 | 0.000000 | 0.000000 | `/home/wangminan/projects/chronaris/docs/artifacts/stage_i/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r2.md` |
| `full` | 2.133460 | 0.108486 | 2.637427 | `loaded` (96) | 0.000000 | 0.000000 | 0.000000 | `/home/wangminan/projects/chronaris/docs/artifacts/stage_i/stage-i-rigid-body-full-20260607T-stage-i-rigid-body-r2.md` |
| `rigid_body` | 2.484067 | 0.111622 | 6.105468 | `loaded` (96) | 1.133332 | 3.946612 | 0.000000 | `/home/wangminan/projects/chronaris/docs/artifacts/stage_i/stage-i-rigid-body-rigid-body-20260607T-stage-i-rigid-body-r2.md` |

## Rigid-Body Mapping Diagnostics

- enabled_residuals: `['translation', 'vertical']`
- missing_requirements: `{'rotation': ['pitch/pitch_rate', 'roll/roll_rate', 'yaw/yaw_rate']}`

| group | matched feature | matched label |
| --- | --- | --- |
| `speed` | `BUS6000019110020.code1027` | `[TSPI数据][载机平台系速度][速度_北向][_速度]` |
| `speed` | `BUS6000019110020.code1028` | `[TSPI数据][载机平台系速度][速度_西向][_速度]` |
| `acceleration` | `BUS6000019110020.code1024` | `[TSPI数据][载机平台系加速度][加速度_北向][_加速度]` |
| `acceleration` | `BUS6000019110020.code1025` | `[TSPI数据][载机平台系加速度][加速度_西向][_加速度]` |
| `acceleration` | `BUS6000019110020.code1036` | `[TSPI数据][载机法向过载][_过载]` |
| `altitude` | `BUS6000019110020.code1033` | `[TSPI数据][载机海拔高度][_高度]` |
| `altitude` | `BUS6000019110020.code1043` | `[TSPI数据][载机气压高度][_高度]` |
| `altitude` | `BUS6000019110020.code1044` | `[TSPI数据][载机卫星高度][_高度]ICD_1` |
| `vertical_speed` | `BUS6000019110020.code1026` | `[TSPI数据][载机平台系加速度][加速度_天向][_加速度]` |
| `vertical_speed` | `BUS6000019110020.code1029` | `[TSPI数据][载机平台系速度][速度_天向][_速度]` |
| `pitch` | `BUS6000019110020.code1030` | `[TSPI数据][载机俯仰角][_角度_毫弧度]` |
| `pitch_rate` | `-` | `-` |
| `roll` | `BUS6000019110020.code1032` | `[TSPI数据][载机横滚角][_角度_毫弧度]` |
| `roll_rate` | `-` | `-` |
| `yaw` | `-` | `-` |
| `yaw_rate` | `-` | `-` |

## Reading

1. 三组结果都已经使用本地 `127.0.0.1:3306` 的真实 MySQL `vehicle_field_metadata`，元数据状态统一为 `loaded`。
2. `minimal` 仍然在这条真实 smoke 上出现极高的 physiology smoothness，`physics_total` 明显失稳，不适合作为当前最小可用配置。
3. `rigid_body` 现在已经同时触发 `translation` 与 `vertical`，其中 `vehicle_rigid_body_vertical` 从旧版的 0 变为非零，说明 `速度_天向 + 高度` 的真实标签映射已经生效。
4. `rotation` 仍为 0，不是 MySQL 元数据问题，而是当前真实字段里没有与 `pitch/roll/yaw` 成对的角速度标签。

## Next

- 下一步应继续扩充 `yaw -> 真航向` 等角度映射，并确认是否存在可用角速度字段；若确实没有，就把 rotation 结论写成“当前 sortie 无可用 rate field”。
- 如果要把 `rigid_body` 作为论文主消融之一，建议后续至少再补一条不同 sortie 的 smoke，确认 vertical residual 不是单例现象。
