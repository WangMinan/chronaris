# Alignment Preview - 20251005 ACT-4 J20 22#01

## 范围

本报告记录 `阶段 E` 的首轮真实最小训练回归结果。

本次运行直接复用了服务器本机已配置好的 `influx` CLI 活动配置，未依赖 MySQL 元信息查询，而是基于当前仓库已确认的 overlap-focused 事实做最小实跑。

配置：

- sortie: `20251005_四01_ACT-4_云_J20_22#01`
- runtime device: `cuda`
- physiology measurements: `eeg`, `spo2`
- vehicle measurement: `BUS6000019110020`
- physiology tag filters:
  - `collect_task_id=2100448`
  - `pilot_id=10033`
- vehicle tag filter:
  - `sortie_number=20251005_四01_ACT-4_云_J20_22#01`
- query window:
  - start: `2025-10-05T01:35:00Z`
  - stop: `2025-10-05T01:38:01Z`
- sample window:
  - duration: `5000 ms`
  - stride: `5000 ms`
- split:
  - train: `60%`
  - validation: `20%`
  - test: `20%`
  - gap windows: `0`
- reference grid points: `16`
- epochs: `3`
- batch size: `8`
- optimizer: `Adam`
- learning rate: `1e-3`
- prototype:
  - hidden dim: `32`
  - embedding dim: `32`
  - projection dim: `16`
  - ODE method: `euler`

## 样本摘要

- generated E0 samples: `25`
- max physiology feature count: `12`
- max vehicle feature count: `21`
- split counts:
  - train: `15`
  - validation: `5`
  - test: `5`

## 训练指标

### Train History

Epoch 1:

- physiology reconstruction: `983877358.9333333`
- vehicle reconstruction: `2.8984689999859613e+26`
- alignment: `0.3034619867801666`
- total: `2.8984689999859613e+26`

Epoch 2:

- physiology reconstruction: `983865625.6`
- vehicle reconstruction: `2.8984689999859613e+26`
- alignment: `0.1871648758649826`
- total: `2.8984689999859613e+26`

Epoch 3:

- physiology reconstruction: `983857305.6`
- vehicle reconstruction: `2.8984689999859613e+26`
- alignment: `0.1080525000890096`
- total: `2.8984689999859613e+26`

### Validation History

Epoch 1:

- physiology reconstruction: `1175453696.0`
- vehicle reconstruction: `2.9050661213668713e+26`
- alignment: `0.20460203289985657`
- total: `2.9050661213668713e+26`

Epoch 2:

- physiology reconstruction: `1175444096.0`
- vehicle reconstruction: `2.9050661213668713e+26`
- alignment: `0.1101318821310997`
- total: `2.9050661213668713e+26`

Epoch 3:

- physiology reconstruction: `1175436416.0`
- vehicle reconstruction: `2.9050661213668713e+26`
- alignment: `0.0549350269138813`
- total: `2.9050661213668713e+26`

### Test Metrics

- physiology reconstruction: `1209078016.0`
- vehicle reconstruction: `2.908369010893269e+26`
- alignment: `0.0549350269138813`
- total: `2.908369010893269e+26`

## 当前结论

这次回归已经证明：

1. 真实 overlap-focused E0 样本可以直接进入 `AlignmentPreviewPipeline`
2. `split -> reference grid -> torch batch -> dual-stream ODE-RNN -> reconstruction/alignment loss -> train/validation/test` 链路已经真实跑通
3. `alignment loss` 在 train / validation 上都明显下降，说明共享参考时间轴上的最小对齐目标已经开始起作用

同时也暴露出当前最关键的新问题：

1. vehicle reconstruction loss 量级约为 `1e26`
2. 总损失几乎完全被 vehicle stream 主导
3. physiology reconstruction 和 alignment loss 的变化已经很难在总损失中体现

## 工程含义

当前阶段 E 的 blocker 已从“链路能不能跑通”转移为：

1. 导出并检查共享参考时间轴上的中间态
2. 处理 vehicle stream 的尺度问题
3. 决定先做：
   - 输入归一化
   - reconstruction loss reweighting
   - 或两者同时做

在这一步完成前，不宜把当前 total loss 的数量级直接当作“训练是否稳定”的唯一判断依据。
