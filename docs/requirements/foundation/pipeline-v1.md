# Pipeline V1

## 1. 输入

`pipeline-v1` 的输入是一个 `SortieLocator`。

它由 `access` 层负责转成：

- 生理点序列
- 航电点序列
- 架次元信息

## 2. 处理步骤

### Step 1. 读取架次数据

使用 `SortieLoader` 聚合同一架次的原始数据。

补充约束：

- 生理数据应直接解析为完整时间戳，保留微秒级精度
- 飞机数据若原始只有时分秒，应先按飞行日期语义组装完整日期时间
- 当前真实库中的飞行日期语义来自 `flight_batch.fly_date`
- 若出现跨日，应按参考时间回退规则追加日期偏移

输出：

- `SortieBundle`

### Step 2. 建立统一时间基准

使用 `TimebasePolicy` 选择参考时刻。

默认建议：

- `earliest_observation`

输出：

- `AlignedSortieBundle`

### Step 3. 构建样本窗口

基于 `WindowConfig` 切分为窗口样本。

输出：

- `SampleWindow[]`

### Step 4. 形成构建结果

输出：

- `DatasetBuildResult`

## 3. v1 非目标

下面这些不放在 v1：

- 直接查询真实 Influx/MySQL 表结构
- 完整特征工程
- 对齐模型训练
- 融合模型训练
- 服务化部署

## 4. v1 完成标准

满足以下条件即可认为 v1 打通：

1. 能通过统一接口装载单架次原始对象
2. 能稳定生成相对时间偏移
3. 能切出窗口样本
4. 能输出可读的结果摘要
5. 有基础单元测试覆盖主要时间与窗口逻辑
