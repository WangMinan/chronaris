# 阶段 H 收口记录（2026-04-27）

## 1. 收口结论

阶段 H 已完成收口，可以进入阶段 I 的下游任务验证准备。

本轮收口完成了三个 gate：

1. 双流 Stage H 导出从 `preview` 复核到 `validation` profile，取消默认每 measurement `500` 点上限，但仍使用已验证的双流时间范围。
2. `20251110_单01_ACT-2_涛_J20_26#01` 从 partial-data seed 补齐为真实 vehicle-only entry，并生成单流窗口 manifest 与 feature bundle。
3. 下游读取接口 `load_stage_h_feature_run()` 已能从 `run_manifest.json` 直接读取 3 个 view，不依赖训练中间对象。

`20251110...` 仍被定义为 vehicle-only partial-data，不提升为双流 Stage H view。

## 2. 真实数据核验

只读核验结果：

- MySQL `flight_task / flight_batch`：`flight_date=2025-11-10`
- Influx bucket：`bus`
- BUS measurement family：
  - `BUS6000019110027`
  - `BUS6000019110028`
  - `BUS6000019110029`
  - `BUS6000019110030`
  - `BUS6000019110031`
- 查询范围：`2025-11-10T00:00:00Z` 到 `2025-11-11T00:00:00Z`
- tag filter：`sortie_number=20251110_单01_ACT-2_涛_J20_26#01`
- 上述 5 个 measurement 均可在当天范围内读到样例点。

## 3. 代码与配置变更

主要变更：

- `configs/partial-data/stage-h-seed-v1.jsonl` 已补齐真实 `bucket / time_range / measurement_family / tag_filters`
- `src/chronaris/pipelines/partial_data_contracts.py`：partial-data entry、manifest、bundle key 与 JSONL 契约
- `src/chronaris/pipelines/partial_data_sources.py`：Influx 分块读取与 MySQL RealBus 字段元数据过滤
- `src/chronaris/pipelines/partial_data_builder.py`：vehicle-only 窗口构建、每字段每窗口最多 `32` 点、`vehicle_only_feature_bundle.npz` 写出
- `src/chronaris/pipelines/partial_data.py`：兼容 re-export 入口
- `src/chronaris/access/influx_cli.py`：支持多 measurement 查询与 Flux 侧 `window + limit` 下推
- `scripts/run_stage_h_export.py`：partial-data sidecar 接入真实 Influx reader 与 MySQL metadata provider

实现约束：

- 不一次性拉取全天原始点。
- Influx 查询按 `300s` 分块执行。
- Flux 侧先按 `5s` window 分组，再对每个 `_measurement/_field/_start/_stop` 保留最多 `32` 行。
- Python builder 侧再次按同一上限保护窗口内字段点数。
- 只保留 RealBus 元数据可解析字段；缺映射 measurement 会进入 manifest 的 `measurement_statuses`。

## 4. 收口实跑资产

主运行：

- run manifest：`docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json`
- 主报告：`docs/reports/stage-h-closure-2026-04-27.md`
- 机器资产根目录：`docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/`

双流 Stage H `validation` 结果：

| sortie | view 数 | view |
| --- | ---: | --- |
| `20251005_四01_ACT-4_云_J20_22#01` | 1 | `20251005_四01_ACT-4_云_J20_22#01__pilot_10033` |
| `20251002_单01_ACT-8_翼云_J16_12#01` | 2 | `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035`, `20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033` |

下游读取验证：

- `load_stage_h_feature_run("docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json")`
- `generated_view_count=3`
- 三个 view 的 `fused_representation.shape` 均为 `(8, 16, 96)`

partial-data 结果：

- manifest：`docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/partial_data/partial_data_manifest.jsonl`
- window manifest：`docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/partial_data/vehicle_only_window_manifest.jsonl`
- feature bundle：`docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/partial_data/vehicle_only_feature_bundle.npz`
- `built_entry_count=1`
- `skipped_entry_count=0`
- `vehicle_only_window_manifest` 行数：`1478`
- `vehicle_only_feature_bundle.npz`：
  - `values.shape=(1478, 105, 823)`
  - `sample_ids=1478`
  - `feature_names=823`

## 5. 诊断说明

本次 `validation` 导出中：

- `20251005...__pilot_10033`：`PASS`
- `20251002...__pilot_10035`：`PASS`
- `20251002...__pilot_10033`：`WARN`

`WARN` 来自投影诊断阈值提醒，不代表导出失败。对应 view 的 `feature_bundle.npz`、`window_manifest.jsonl`、`projection_diagnostics_summary.json` 与 `causal_fusion_summary.json` 均已生成。

本轮曾观察到 vehicle-only partial 构建的初版实现 CPU / RSS 偏高。原因是窗口上限只在 Python 侧生效，Influx 仍返回了较大的 raw CSV。已修正为 Flux 侧 `window + limit` 下推后再构建，partial-only 实跑完成并写出 bundle。

## 6. 测试闭环

已新增或扩展覆盖：

- partial-data seed concrete entry 加载
- Influx vehicle-only provider scope 与 Flux 侧 window limit
- RealBus 字段过滤与 measurement skip reason
- 每窗口每字段最多 `32` 点
- `vehicle_only_feature_bundle.npz` 固定键与 shape
- fake Stage H validation run：3 view + partial sidecar + report + `load_stage_h_feature_run()`

已执行：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_h_export`

全量测试结果见本轮最终记录。

## 7. 阶段 I 入口

阶段 I 不应再直接读取 E/F/G 训练中间对象。默认入口应为：

- 双流标准化融合特征：`load_stage_h_feature_run(run_manifest_path)`
- vehicle-only partial 资产：`vehicle_only_feature_bundle.npz` + `vehicle_only_window_manifest.jsonl`

后续优先任务：

1. 设计最小下游任务消费协议。
2. 用 Stage H run manifest 作为标准输入，做认知负荷评估 / 空中失能风险分析 / 飞行事件复盘的最小对比或消融。
3. 继续轻量多架次 manifest 盘点，但不要把 vehicle-only partial 数据误当成双流融合样本。
