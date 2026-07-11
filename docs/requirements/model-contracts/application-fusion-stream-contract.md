# 应用下游评估双流与融合表示合同

日期：2026-07-10  
状态：后续编码、导出和公平性测试的强制接口

## 1. 目的

本合同把“原始异步双流”“模型编码器”“统一融合表示”和“下游算法”分开，避免任务头、标签或某个模型的内部维度污染公平比较。

主数据流固定为：

```text
原始生理流 + 原始航电流
  -> 双流观察批次
  -> 方法编码器
  -> [B, Tq, 64] 融合表示
  -> 冻结下游算法
  -> 任务预测与指标
```

## 2. 双流观察批次

逻辑接口如下；实现使用 `torch.Tensor`，数据加载层负责从既有 numpy/dataclass 合同转换。

```python
@dataclass(frozen=True, slots=True)
class DualStreamObservationBatch:
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    physiology_values: torch.Tensor
    physiology_timestamps_s: torch.Tensor
    physiology_point_mask: torch.Tensor
    physiology_feature_mask: torch.Tensor
    physiology_observation_age_s: torch.Tensor

    vehicle_values: torch.Tensor
    vehicle_timestamps_s: torch.Tensor
    vehicle_point_mask: torch.Tensor
    vehicle_feature_mask: torch.Tensor
    vehicle_observation_age_s: torch.Tensor

    query_timestamps_s: torch.Tensor
    source_sample_hashes: tuple[str, ...]
```

### 2.1 形状

| 字段 | 形状 | 类型 |
| --- | --- | --- |
| `physiology_values` | `[B, Tp, Fp]` | float32 |
| `physiology_timestamps_s` | `[B, Tp]` | float64 或 float32 |
| `physiology_point_mask` | `[B, Tp]` | bool |
| `physiology_feature_mask` | `[B, Tp, Fp]` | bool |
| `physiology_observation_age_s` | `[B, Tp, Fp]` | float32 |
| 航电对应字段 | `[B, Tv, Fv]` 等 | 同上 |
| `query_timestamps_s` | `[B, Tq]` | float64 或 float32 |

### 2.2 时间语义

- 时间为相对当前 30 秒上下文起点的秒数。
- 有效时间必须单调非降；相同时间允许出现不同 measurement 的观测。
- padding 时间值不参与计算，对应 point mask 必须为 false。
- query time 严格递增，主协议 `Tq=96`，即每个 5 秒窗口 16 个查询点、六个窗口组成 30 秒上下文。
- 原始微秒/毫秒精度保留在输入 manifest；模型内部不得先把两个流强制改成同一原始采样率。

### 2.3 mask 与观测年龄

- point mask 表示该时间位置是否存在任一有效字段。
- feature mask 表示具体字段是否真实观测；插值值不能标记为真实观测。
- observation age 表示距该字段最近真实观测的秒数；从未观测时使用 `+inf` 并由 mask 屏蔽。
- 整段模态缺失通过该模态 point mask 全 false 表示，不用全零数值冒充有效观测。

### 2.4 字段元数据

批次之外的 dataset manifest 必须记录：

- 生理/航电字段顺序、中文含义、单位和来源 measurement。
- 字段是原始观测、确定性派生、插值还是标签源。
- 训练折排除字段及原因。
- 每个字段在 train/test 的有效点数和缺失率。

### 2.5 鼎新长表的模型无关时间合并

鼎新输入在进入任何编码器或训练折预处理器之前，统一按 `100 ms` 固定因果时间箱合并。时间箱内同一字段取均值，输出时间戳取该箱最后一次真实观测，随后重算 mask 与 observation age。该函数不得接收方法名、标签或 split role，六种方法必须消费完全相同的结果；原始 snapshot 和 source hash 不因模型输入压缩而改变。

时间箱只延迟箱内较早事件，最大延迟小于 `100 ms`，不允许把观测前移。它不替代 96 点公共查询轴，也不允许在 outer-test 上重新拟合任何参数。正式 manifest 必须记录 `model_input_bin_width_s=0.1`，恢复 checkpoint 时该值变化视为协议不兼容。

## 3. 统一融合表示

```python
@dataclass(frozen=True, slots=True)
class FusionStreamBatch:
    sample_ids: tuple[str, ...]
    timestamps_s: torch.Tensor
    sequence_embedding: torch.Tensor
    valid_mask: torch.Tensor
    pooled_embedding: torch.Tensor
    method_name: str
    fold_id: str
    checkpoint_sha256: str
    source_sample_hashes: tuple[str, ...]
```

### 3.1 强制形状

| 字段 | 形状 |
| --- | --- |
| `timestamps_s` | `[B, Tq]` |
| `sequence_embedding` | `[B, Tq, 64]` |
| `valid_mask` | `[B, Tq]` |
| `pooled_embedding` | `[B, 64]` |

pooled embedding 按 valid mask 做均值池化；任何额外 delta、统计量或任务特征都不能追加到 64 维合同中。

### 3.2 禁止字段

融合表示文件和表示 manifest 不得包含：

- 分类或回归 logits、预测值、残差目标、类别概率。
- 真实标签、高响应标记或机动状态。
- 检索 rank、候选 ID 或任务名称。
- attention entropy、物理残差等仅对 Chronaris 可见的诊断量。

诊断量可以写入独立 diagnostics 文件，但下游 consumer 的输入加载器必须拒绝这些列。

## 4. 编码器协议

```python
class FusionStreamEncoder(Protocol):
    method_name: str
    output_dim: int = 64

    def forward(
        self,
        batch: DualStreamObservationBatch,
    ) -> FusionStreamBatch: ...
```

训练器负责优化与 checkpoint；编码器只负责从同一输入合同生成表示。任务头不得成为编码器导出的必要组成。

### 4.1 训练接口

每个 fold 的训练调用必须显式接收：

- `train_sample_ids`、`validation_sample_ids` 和 `held_out_sample_ids`。
- pretext objective 配置。
- method 配置、seed、最大 epoch、early stopping 规则。
- scaler/adapter 的拟合样本 ID。
- 是否使用仿真预训练以及对应生成器 manifest hash。

训练器输出：checkpoint、training protocol、curve summary、fit-sample hash 和可恢复状态。

### 4.2 OOF 导出

对每个外层 fold：

1. 仅在 train groups 训练编码器和所有预处理器。
2. 锁定 checkpoint。
3. 分别导出 train/validation/test 表示，并标记 export role。
4. 下游算法只在 train 表示拟合，在 test 表示评价。
5. 同一 test sample 只能对应一个 held-out checkpoint。

OOF 合并器必须检查样本无重复、无遗漏、顺序一致和 checkpoint lineage 完整。

## 5. 六种方法的适配规则

### 5.1 生理单流

- 只读取生理字段，航电 mask 置空但不删除批次字段。
- 使用与 ContiFormer 单流编码器相同的连续时间 block、64 维隐藏层和公共 pretext 预算。
- 作为生理响应任务的强单流基线。

### 5.2 航电单流

- 只读取航电字段。
- 编码器结构和训练预算与生理单流相同。
- 作为机动任务的强单流基线。

### 5.3 朴素时间同步

- 在 query time 上对每个流分别执行仅使用过去/当前观测的 forward-fill 或线性插值；默认不得使用未来观测。
- 追加 feature-valid mask 和 observation age。
- 标准化和 PCA 只在训练折拟合，保留 `min(64, 可用维数)` 个分量后零填充到 64 维。
- 不使用标签训练投影。

### 5.4 MulT

- 复用 vendored MulT cross-modal block 和现有 `sequence_embedding`。
- 为两个流增加相对时间和 mask 特征。
- 双向跨模态注意力后投影到 64 维；不得使用任务 head 的 pooled embedding 作为主表示。

### 5.5 ContiFormer

- 复用现有 ContiFormer encoder，分别消费原始时间和 mask。
- 双流时序状态在 query time 上重采样后融合并投影到 64 维。
- 导出的 sequence embedding 必须来自任务头之前。

### 5.6 Chronaris

固定链路：

```text
生理 ObservationEncoder -> ODE-RNN
航电 ObservationEncoder -> ODE-RNN
-> query time 连续潜态
-> 对齐与物理一致性约束
-> 生理查询航电历史的因果融合
-> 64 维表示投影
```

具体要求：

- 必须调用 `DualStreamODERNNPrototype` 或其等价重构后的生产实现，不能由任务评价 wrapper 替代。
- 固定点数 lag 改为按真实秒数建立的 0–5、5–15、15–30 秒三个可见域。
- 多尺度门控只能在各自因果可见域内归一化。
- 物理损失必须在 manifest 中逐项记录 active/unavailable；字段缺失时不得把未计算项写成 0 并宣称约束有效。
- 无因果掩码、无物理、无连续演化和单尺度 lag 作为固定消融，不另行搜索消融配置。

## 6. 公共 pretext 训练合同

MulT、ContiFormer、Chronaris 和两个单流编码器共享：

| 目标 | 权重 | 说明 |
| --- | ---: | --- |
| masked reconstruction | 1.0 | 只重构被训练增强遮挡的观测 |
| short-horizon prediction | 0.5 | 预测下一查询点或短窗口状态 |
| lag discrimination | 0.2 | 区分真实配对与错误时移 |

Chronaris 额外使用：

| 目标 | 最终权重 |
| --- | ---: |
| continuous alignment | 0.2 |
| physical consistency | 0.1 |
| causal direction regularization | 0.1 |

第 1–10 epoch 只启用公共目标；第 11–20 epoch 将三个 Chronaris 目标从 0 线性升至最终权重；之后保持不变。

## 7. 训练增强合同

增强仅作用于训练折：

- 单点随机缺失概率 0.10。
- 连续缺失段长度从 1–5 秒均匀采样，每个样本最多一段/模态。
- 整段模态 dropout 概率 0.05，两个模态不得同时 dropout。
- 时间抖动标准差取真实审计 p95 的一半，限制在 5–100 ms。
- 额外时钟偏移从 `[-250, 250] ms` 均匀采样。

所有深度融合方法使用同一增强 realization；训练批次通过 seed 派生共享 augmentation ID。

## 8. 表示 manifest

每个正式导出必须记录：

- 输入 snapshot 与 split manifest hash。
- method、配置、参数量、训练 seed 和训练时长。
- train/validation/test group 与 sample hash。
- scaler、PCA、投影层和 checkpoint hash。
- output dim、query point count 和 valid ratio。
- pretext 目标及损失权重。
- synthetic pretraining 状态和生成器版本。
- `label_used_for_encoder_training=false`；若端到端微调则必须另建表示族并写 true。

## 9. 合同失败行为

遇到下列情况必须 fail closed：

- 六方法样本 ID 或 query time 不一致。
- test group 出现在任一 fit-sample 列表中。
- 输出维度不是 64。
- 表示文件含禁止字段。
- checkpoint hash 缺失或 held-out lineage 不完整。
- Chronaris 声明物理项 active 但没有相应字段或 loss 记录。

失败 fold 写 `contract_violation` 和具体原因，不用零指标占位，不参与汇总排名。
