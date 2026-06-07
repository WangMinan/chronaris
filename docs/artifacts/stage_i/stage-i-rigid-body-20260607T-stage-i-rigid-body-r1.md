# Stage I Rigid-Body Ablation - 20260607T-stage-i-rigid-body-r1

- summary_json: `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r1/rigid_body_ablation_summary.json`
- source_sortie: `20251005_四01_ACT-4_云_J20_22#01`
- pilot_id: `10033`
- setting: `epoch_count=1, batch_size=8, input_normalization_mode=zscore_train, device=cpu`

## Test Comparison

| family | total | alignment | physics_total | vehicle metadata | key vehicle component | report |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `minimal` | 1110.708618 | 0.094877 | 11088.659180 | `loaded` (96) | `vehicle_semantic=0.009819` | `/home/wangminan/projects/chronaris/docs/artifacts/stage_i/stage-i-rigid-body-minimal-20260607T-stage-i-rigid-body-r1.md` |
| `full` | 2.133460 | 0.108486 | 2.637427 | `loaded` (96) | `vehicle_semantic=1.444417` | `/home/wangminan/projects/chronaris/docs/artifacts/stage_i/stage-i-rigid-body-full-20260607T-stage-i-rigid-body-r1.md` |
| `rigid_body` | 2.106053 | 0.104823 | 2.438377 | `loaded` (96) | `vehicle_rigid_body_translation=1.413177` | `/home/wangminan/projects/chronaris/docs/artifacts/stage_i/stage-i-rigid-body-rigid-body-20260607T-stage-i-rigid-body-r1.md` |

## Vehicle Component Breakdown

| family | vehicle_semantic | vehicle_smoothness | vehicle_rigid_body_translation | vehicle_rigid_body_vertical | vehicle_rigid_body_rotation | vehicle_latent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `minimal` | 0.009819 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `full` | 1.444417 | 0.060723 | 0.000000 | 0.000000 | 0.000000 | 0.106621 |
| `rigid_body` | 0.000000 | 0.000000 | 1.413177 | 0.000000 | 0.000000 | 0.000000 |

## Reading

1. 本次已通过 `127.0.0.1:3306` 读取到真实 MySQL `vehicle_field_metadata`，三组结果的 `vehicle_field_metadata.status` 都已变为 `loaded`。
2. `minimal` 仍然在这条真实 smoke 上出现极高的 physiology smoothness，`physics_total` 明显失稳，不适合作为当前最小可用配置。
3. `full` 与 `rigid_body` 都能稳定收敛到同量级的 `total / alignment / physics_total`。
4. `rigid_body` 的 `vehicle_rigid_body_translation` 已经非零，说明刚体语义残差不再只走 latent fallback，而是已经使用了真实 vehicle_field_metadata。
5. 当前 `vehicle_rigid_body_vertical / rotation` 仍为 0，说明这条 sortie 在现有字段映射下主要只触发了 translation 残差；这属于真实字段语义覆盖范围问题，不再是 MySQL 连通性问题。

## Next

- 优先检查 `BUS6000019110020` 字段标签里哪些列可稳定映射到 `altitude / vertical_speed / pitch_rate / roll_rate / yaw_rate`，决定是否需要扩充 token 映射。
- 在论文表述里，现在可以把这轮 `rigid_body` 写成“translation residual 已经在真实链路生效”，但不能把 vertical / rotation 一并写成已验证。
