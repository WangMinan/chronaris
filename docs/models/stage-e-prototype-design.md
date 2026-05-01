# Stage E Prototype Design

## 1. 文档目的

这份文档用于把“选题报告/基金申请书中的研究主线”映射到当前 `chronaris` 仓库的 `阶段 E` 实现边界，作为下一阶段编码的直接依据。

它要回答的是：

1. `阶段 E` 到底先做什么
2. 当前现有代码能直接承接什么
3. 研究文本里提到的物理约束与因果融合，哪些现在做，哪些先预留接口

## 2. 依据来源

本草案综合了下面几类来源：

1. `docs/planning/coding-roadmap.md`
2. `docs/models/e0-minimal-input.md`
3. `docs/models/alignment-batch-contract.md`
4. `docs/planning/stage-e-closure-2026-04-21.md`
5. `docs/选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx`
6. `docs/选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx`
7. `docs/models/stage-e-reference-repos.md`

## 3. 当前阶段判断

按照路线图与收口记录，当前结论是：

1. `阶段 E 已完成收口`
2. `阶段 F` 可在当前阶段 E 基线上启动
3. 当前仓库已具备稳定的 `E0ExperimentSample -> AlignmentBatch -> Stage E objective` 路径

也就是说，本文档当前主要作为阶段 E 设计存档；下一轮实现默认转向阶段 F。

## 4. 研究主线与当前实现边界的映射

选题报告和基金申请书描述的是一条完整研究链：

1. 双流连续潜态建模
2. 物理一致性约束时间对齐
3. 因果掩码跨模态融合
4. 标准化融合特征输出
5. 典型任务验证

但 `chronaris` 当前路线图为了降低实现风险，把它拆成了多个阶段：

| 研究文本中的模块 | 当前仓库阶段 | 当前决定 |
| --- | --- | --- |
| 双流连续潜态建模 | `阶段 E` | 已完成收口，默认冻结基线 |
| 物理一致性约束时间对齐 | `阶段 F` | 已完成完整物理约束族收口 |
| 因果掩码跨模态融合 | `阶段 G` | 已完成最小非对称因果融合收口 |
| 标准化融合特征输出 | `阶段 H` | 已完成收口，标准化 feature bundle 与 `load_stage_h_feature_run()` 入口已冻结 |
| 下游任务验证 | `阶段 I` | 已完成收口，公开数据双轨 baseline、Stage H case study 与 Phase 3 closure 已落盘 |

这意味着：

1. 选题和基金中的理论主线仍然成立
2. 代码实现继续按 `E -> F -> G -> H -> I` 的工程顺序分步落地
3. 本文档保留阶段 E 原型设计背景；当前阶段事实以 `docs/planning/coding-roadmap.md` 为准

## 5. 当前代码基底

截至目前，`阶段 E` 已经可以直接依赖下面这些契约：

### 输入侧

- `E0ExperimentSample`
- `NumericStreamMatrix`
- `AlignmentBatch`

### 输入语义

当前每条流已经具备：

1. 数值特征矩阵
2. 点级时间偏移 `offsets_ms`
3. padding 后的 `mask`
4. 统一后的特征名列表

其中，`AlignmentBatch` 的核心形状已经是：

1. `values: [B, T, F]`
2. `mask: [B, T]`
3. `offsets_ms: [B, T]`

### 当前已知数据规模

当前 overlap-focused preview 已确认：

1. 样本数约 `25`
2. physiology 最大数值特征数 `12`
3. vehicle 最大数值特征数 `21`

这说明阶段 E 首轮原型应优先追求：

1. 输入协议不再变化
2. 单架次 preview 数据可以稳定过拟合
3. 中间态和损失可以解释

## 6. 阶段 E 的目标与非目标

### 目标

`阶段 E` 首轮原型需要完成：

1. 基于 `AlignmentBatch` 的双流连续时间编码
2. 生理流与航电流各自的连续潜态推进
3. 最小观测重构损失
4. 最小时间对齐损失
5. 可导出的中间态结果

### 非目标

当前不在 `阶段 E v1` 一次性完成：

1. 全量物理残差约束
2. 因果掩码跨模态注意力
3. 最终融合特征导出规范
4. 完整服务化接口
5. 大规模多架次训练框架

## 7. 核心设计选择

### 选择什么主骨架

`阶段 E v1` 采用：

**确定性双流 ODE-RNN 原型**

而不是一开始直接采用：

1. 全量 variational `Latent ODE`
2. 全量 `Neural CDE`
3. 带随机扩散项的 `SDE`

### 这样选的原因

1. 当前输入是离散观测点，不是现成的连续控制路径
2. 选题/基金文本都强调“连续求解 + 观测更新”机制
3. `ODE-RNN / GRU-ODE` 更贴近当前 `E0 -> AlignmentBatch` 协议
4. 相比 `VAE + KL + posterior inference`，确定性原型更适合先做最小可训练实验
5. 该路线更容易在后续接入 `阶段 F` 的物理约束和 `阶段 G` 的因果融合

### 与研究文本的对应

这个选择与基金申请书中的这条逻辑是一致的：

1. 两条流分别建模
2. 在观测间隔内做连续演化
3. 在观测到达时做门控更新
4. 在统一参考时间轴上进行软对齐

## 8. 原型总体结构

### 8.1 输入适配层

新增一层 `torch` 侧输入适配，把 `AlignmentBatch` 转成模型前向需要的张量对象。

建议输出至少包含：

1. `values`
2. `mask`
3. `offsets_ms`
4. `offsets_s`
5. `delta_t_s`
6. `feature_valid_mask`

其中：

1. `delta_t_s` 由相邻有效观测时间差得到
2. `feature_valid_mask` 从 `values` 中的 `NaN` 推导
3. `offsets_ms -> offsets_s` 在适配层完成，不把时间单位转换塞进模型主体

### 8.2 双流观测编码器

每条流各自拥有独立编码器：

1. physiology encoder
2. vehicle encoder

每个编码器负责把观测向量 `x_t` 映射为观测嵌入 `e_t`。

首轮建议保持简单：

1. `Linear`
2. `LayerNorm`
3. `GELU/ReLU`
4. `Linear`

理由：

1. 当前特征规模不大
2. 首轮重点在连续推进，不在复杂前端特征工程
3. 编码器越简单，越容易判断问题出在时间建模还是输入侧

### 8.3 连续时间状态推进

每条流各自维护一个隐藏状态：

1. `h_phys(t)`
2. `h_vehicle(t)`

在两个观测时刻之间，状态通过 ODE 求解器推进：

1. `dh_phys / dt = f_phys(h_phys, t)`
2. `dh_vehicle / dt = f_vehicle(h_vehicle, t)`

在观测到达时，使用门控更新吸收新信息：

1. 先把旧状态推进到当前观测时刻
2. 再把当前观测嵌入送入更新单元
3. 更新单元输出当前时刻的新状态

首轮实现上建议采用：

1. `torchdiffeq` 作为求解器
2. `GRUCell` 风格的观测更新单元

这样更贴近“连续求解 + 观测更新”的实现形态。

### 8.4 统一参考时间轴

虽然两条流的观测时刻不同，但对齐损失不能只在各自原始采样点上算。

因此，`阶段 E v1` 需要在每个窗口内构造共享参考时间轴：

1. 以窗口起点为 `0`
2. 以窗口终点为 `window_duration`
3. 在其间均匀取 `K` 个参考点

默认建议：

1. `K = 16` 或 `K = 32`

模型在这组参考时间点上分别求出：

1. `h_phys(t_k)`
2. `h_vehicle(t_k)`

再进行投影和对齐。

### 8.5 对齐投影头

在共享参考时间轴上，不直接用原始隐藏状态做对齐，而是先投影到共享对齐空间：

1. `a_phys(t_k) = P_phys(h_phys(t_k))`
2. `a_vehicle(t_k) = P_vehicle(h_vehicle(t_k))`

这里的作用是：

1. 允许两条流保留各自动力学差异
2. 只在共享语义子空间里逼近
3. 为 `阶段 F` / `阶段 G` 继续扩展保留接口

## 9. 损失设计

### 9.1 重构损失

每条流都应保留最小重构目标。

建议：

1. physiology reconstruction loss
2. vehicle reconstruction loss

实现方式：

1. 在真实观测时刻，用 decoder 从隐藏状态重构观测值
2. 仅在 `mask=True` 且该特征非 `NaN` 的位置上计算 MSE

这一步的意义是：

1. 避免隐藏状态只为“跨流相似”而塌缩
2. 强迫每条流自己的连续潜态仍然对原始观测负责

### 9.2 最小对齐损失

`阶段 E v1` 的对齐损失建议先用：

**共享时间轴上的投影表示相似损失**

可采用：

1. `MSE`
2. `1 - cosine similarity`

首轮建议优先使用 `cosine + MSE` 中较稳的一种，不先把 `soft-DTW` 设为唯一主损失。

原因：

1. 当前样本很少
2. 首轮目标是先把训练过程跑稳
3. `soft-DTW` 更适合作为第二步增强或对比项

### 9.3 总损失

`阶段 E v1` 总损失建议写成：

`L = λ_p * L_recon_phys + λ_v * L_recon_vehicle + λ_a * L_align`

当前不把物理残差项硬塞进 `阶段 E v1`，但在代码接口上要预留：

1. `physics_loss_fn`
2. `regularization_terms`

供 `阶段 F` 接入。

## 10. 中间态输出

为了满足路线图里“能输出对齐中间态”的要求，模型至少应返回：

1. 两条流在观测时刻的隐藏状态
2. 两条流在共享参考时间轴上的隐藏状态
3. 两条流在共享对齐空间上的投影结果
4. 观测重构结果
5. 每个样本的对齐损失分量

这些中间态后续可直接用于：

1. 轨迹可视化
2. 损失诊断
3. `阶段 F` 的物理残差检查
4. `阶段 G` 的语义事件抽取

## 11. 最小训练/验证切分策略

当前路线图把“固化最小训练/验证切分方式”放在 `阶段 E` 前面，因此这里直接定一个默认方案。

### 默认原则

1. 按时间顺序切分
2. 不做随机打散
3. 避免相邻窗口信息泄漏

### 当前单架次 preview 默认切分

若样本数不少于 `15`，默认采用：

1. 前 `60%` 作为 train
2. 中间 `20%` 作为 validation
3. 最后 `20%` 作为 test

对当前约 `25` 个窗口，可默认切成：

1. train: `15`
2. validation: `5`
3. test: `5`

若未来窗口出现重叠，且 `stride < duration`，则在不同切分块之间额外留出至少 `1` 个窗口的 gap，减少泄漏风险。

## 12. 代码落位建议

建议在 `src/chronaris/models/alignment` 下新增：

1. `config.py`
2. `torch_batch.py`
3. `ode_cells.py`
4. `encoders.py`
5. `decoders.py`
6. `losses.py`
7. `reference_grid.py`
8. `prototype.py`

建议在 `src/chronaris/pipelines` 下新增：

1. `alignment_preview.py`

建议在 `tests` 下新增：

1. `test_alignment_torch_batch.py`
2. `test_alignment_prototype_forward.py`
3. `test_alignment_losses.py`
4. `test_alignment_preview_pipeline.py`

## 13. 依赖建议

当前 `pyproject.toml` 还没有深度学习依赖。

`阶段 E` 最小建议依赖：

1. `torch`
2. `torchdiffeq`

当前不建议首轮就加入：

1. `torchcde`
2. `torchsde`
3. 大量训练框架封装

原因是当前更需要先把原型可训练路径走通。

## 14. 推荐实现顺序

推荐按下面顺序推进，而不是一开始就把所有模块堆满。

### 第一步

完成：

1. 时间顺序切分函数
2. `AlignmentBatch -> torch batch` 适配
3. 基础张量与 mask 单元测试

### 第二步

完成单流原型：

1. 一条流的观测编码
2. 一条流的 ODE 推进
3. 一条流的重构损失

先让单流在一个极小样本上过拟合。

### 第三步

扩展为双流：

1. physiology stream
2. vehicle stream
3. 双流共享参考时间轴

### 第四步

加入最小对齐损失：

1. 共享投影头
2. alignment loss
3. 中间态导出

### 第五步

补最小实验脚本与评估摘要：

1. train / validation loss 曲线
2. 参考时间轴上的对齐轨迹
3. 单样本中间态导出

## 15. 阶段 E 的退出条件

满足下面条件，就认为 `阶段 E v1` 完成：

1. 能基于当前单架次 preview 数据完成一次真实训练
2. 双流重构损失可稳定下降
3. 最小对齐损失可稳定下降
4. 前向和反向过程中不出现大面积 `NaN`
5. 能导出一份中间态结果用于人工核验
6. 训练/验证切分方式已经固定，不再依赖手工挑窗

## 16. 当前结论

对 `chronaris` 来说，`阶段 E` 的正确打开方式不是直接实现整套论文方法，而是：

1. 用现有 `E0 -> AlignmentBatch` 协议作为稳定输入
2. 先落一个确定性的双流 ODE-RNN 连续对齐原型
3. 先把重构与最小对齐跑通
4. 给 `阶段 F` 的物理约束和 `阶段 G` 的因果融合显式留接口

这样既不背离选题报告和基金申请书中的研究主线，也符合当前仓库的工程推进顺序。
