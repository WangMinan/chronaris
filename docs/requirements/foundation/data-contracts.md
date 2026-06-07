# Data Contracts

## 1. 设计目的

第一版先统一“对象层协议”，而不是急着统一所有上游字段名。

这样做的目的：

1. 让 `access` 和 `dataset` 解耦
2. 让后续真实数据库适配可以独立推进
3. 让模型层只面对统一对象，不直接依赖上游库表细节

## 2. 核心对象

### `SortieLocator`

表示一次架次定位条件。

关键字段：

- `sortie_id`
- `pilot_id`
- `aircraft_id`
- `start_time`
- `end_time`

### `RawPoint`

表示一条原始时序点。

关键字段：

- `stream_kind`
- `measurement`
- `timestamp`
- `clock_time`
- `timestamp_precision_digits`
- `values`
- `tags`
- `source`

约束：

- `timestamp` 必须来自同一时间语义体系
- 如果原始记录只有时分秒，则 `clock_time` 保存原始时间值，`timestamp` 保存补全后的完整时间
- `timestamp_precision_digits` 用于保留来源精度语义，例如生理流为 6、飞机流为 3
- `values` 不在 v1 强行约束为固定 schema
- 生理流时间戳默认保留微秒级精度
- 飞机流时间戳默认保留毫秒级精度

### `SortieMetadata`

表示架次元信息。

关键字段：

- `sortie_id`
- `flight_task_id`
- `flight_batch_id`
- `flight_date`
- `mission_code`
- `aircraft_model`
- `aircraft_tail`
- `pilot_code`
- `sortie_number`
- `batch_number`
- `extra`

### `SortieBundle`

表示同一架次下的原始输入聚合。

组成：

- `locator`
- `metadata`
- `physiology_points`
- `vehicle_points`

### `AlignedPoint`

表示带相对时间偏移的点。

关键字段：

- `point`
- `offset_ms`

### `AlignedSortieBundle`

表示已经完成统一参考时间校正的架次对象。

关键字段：

- `reference_time`
- `physiology_points`
- `vehicle_points`

### `WindowConfig`

表示窗口构建配置。

关键字段：

- `duration_ms`
- `stride_ms`
- `min_physiology_points`
- `min_vehicle_points`
- `allow_partial_last_window`

### `SampleWindow`

表示一个可直接用于后续训练或评测的窗口样本。

关键字段：

- `sample_id`
- `sortie_id`
- `window_index`
- `start_offset_ms`
- `end_offset_ms`
- `physiology_points`
- `vehicle_points`
- `labels`

### `DatasetBuildResult`

表示单架次数据构建结果。

组成：

- `aligned_bundle`
- `windows`

## 3. v1 时间语义约定

第一版采用下面的保守约束：

1. `access` 层输出的时间戳必须已经具备统一语义。
2. 不允许在一次构建中混用 aware datetime 和 naive datetime。
3. 时间参考策略由 `TimebasePolicy` 显式指定。
4. 内部窗口切分统一基于 `offset_ms`。
5. 飞机数据若原始只有时分秒，必须先结合 `flight_task.flight_date` 补足完整日期后再进入 `RawPoint.timestamp`。
   在当前真实库中，这个日期语义来自 `flight_batch.fly_date`。
6. 飞机数据的跨日补全逻辑应与 `TimeSequenceProcessor` 保持一致：当前参考时间小于前一条时，日期偏移加一。

## 4. 后续可扩展点

这些内容在 v1 不锁死，但接口预留：

- 多种标签源
- 事件流独立建模
- 多 sortie 拼接数据集
- 中间态导出协议
- 特征矩阵版本号
