# Architecture

## 1. v1 目标

`chronaris` 的第一版代码目标不是直接落模型训练，而是先打通一条最小可运行链路：

1. 读取指定架次的人机多源数据与元信息
2. 组织为统一样本对象
3. 统一时间基准并生成窗口
4. 输出可供后续对齐模型与融合模型消费的数据结构

## 2. 分层结构

### `access`

- 面向 InfluxDB / MySQL 的访问边界
- 负责“怎么拿到数据”
- 不负责样本构建和模型逻辑

### `schema`

- 定义统一数据契约
- 负责“数据长什么样”
- 是 `access` 和 `dataset` 之间的稳定接口

### `dataset`

- 负责时间基准、相对时间偏移、窗口切分、样本构建
- 负责“如何把原始点变成可训练样本”

### `pipelines`

- 负责把 `access -> dataset` 串起来
- 负责“这条链路怎么跑”

## 3. v1 时序处理边界

第一版只解决下面这件事：

- 把一个 `sortie` 的生理流、航电流和元信息组织成统一对象
- 基于一个明确的参考时刻生成相对时间轴
- 生成可复用窗口样本

第一版先不解决：

- 真实数据库驱动细节
- 物理一致性损失
- 因果掩码融合
- 训练器、特征导出格式定稿

## 4. 关键对象流

```text
SortieLocator
  -> SortieLoader
  -> SortieBundle
  -> align_sortie_bundle(...)
  -> AlignedSortieBundle
  -> build_sample_windows(...)
  -> DatasetBuildResult
```

## 5. 当前设计决策

1. `schema` 使用 Python dataclass，先保持依赖最小化。
2. `dataset` 以相对时间毫秒为统一内部表示，便于后续模型消费。
3. 时间参考策略做成显式配置，不把“取哪个起点”写死在代码深处。
4. `access` 先以协议接口和内存实现为主，真实 Influx/MySQL 适配后补。
5. 最小验证优先使用单架次数据构建链路，而不是直接开训练脚本。
