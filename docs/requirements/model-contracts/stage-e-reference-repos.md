# Stage E Reference Repos

## 1. 目的

这份文档记录进入 `阶段 E：连续对齐模型原型` 时，适合优先参考的 GitHub 仓库。

它的作用不是现在就把这些仓库直接接成依赖，而是：

1. 明确哪些实现最贴近 `chronaris` 当前问题
2. 避免后续模型设计时重新做一轮零散搜索
3. 为阶段 E / F / G 的实现取舍提供统一参照

## 2. 选择原则

本清单按下面标准筛选：

1. 优先覆盖“不规则采样、多速率、连续时间、时间对齐、多模态融合”这些与当前课题直接相关的实现
2. 优先选择原始作者仓库或常用主仓库，而不是随机 fork
3. 区分“直接可借鉴的主骨架”和“可借鉴的损失/基线/工具库”
4. 不把“论文好看但与当前输入协议不贴合”的仓库放在最前面

## 3. 最值得优先看的仓库

| 仓库 | 链接 | 对 `chronaris` 的直接价值 | 当前建议用途 |
| --- | --- | --- | --- |
| `YuliaRubanova/latent_ode` | <https://github.com/YuliaRubanova/latent_ode> | 最贴近“不规则时间序列 + 连续潜态 + 重构训练”主线 | 参考 `阶段 E` 的双流连续建模主骨架 |
| `edebrouwer/gru_ode_bayes` | <https://github.com/edebrouwer/gru_ode_bayes> | 强调“连续演化 + 观测到达时门控更新”，和当前离散观测输入形式很贴近 | 参考观测更新机制和 ODE-RNN 风格实现 |
| `rtqichen/torchdiffeq` | <https://github.com/rtqichen/torchdiffeq> | PyTorch ODE solver 基础库 | 若阶段 E 采用 Neural ODE 路线，这是最直接的底层求解器 |
| `Maghoumi/pytorch-softdtw-cuda` | <https://github.com/Maghoumi/pytorch-softdtw-cuda> | 可微时间对齐损失 | 作为最小 `alignment loss` 的候选增强件 |

## 4. 重要的备选连续时间路线

| 仓库 | 链接 | 价值 | 当前判断 |
| --- | --- | --- | --- |
| `patrick-kidger/torchcde` | <https://github.com/patrick-kidger/torchcde> | 处理不规则采样、缺失值、连续控制路径非常强 | 适合作为阶段 E 的备选技术路线，不建议先于 ODE-RNN 落地 |
| `patrick-kidger/NeuralCDE` | <https://github.com/patrick-kidger/NeuralCDE> | `torchcde` 的论文级示例实现 | 更适合在第二轮原型时参考 |
| `google-research/torchsde` | <https://github.com/google-research/torchsde> | 连续时间不确定性建模 | 更偏 `阶段 F` 之后的增强，不是 `E v1` 首选 |
| `DiffEqML/torchdyn` | <https://github.com/DiffEqML/torchdyn> | 连续深度学习综合工具箱 | 功能很多，但当前原型期偏重，不建议首轮直接依赖 |

## 5. 重要的基线与对比参考

| 仓库 | 链接 | 价值 | 当前建议用途 |
| --- | --- | --- | --- |
| `ranakroychowdhury/mTAN` | <https://github.com/ranakroychowdhury/mTAN> | 不规则时间点的时间注意力建模 | 后续作为非 ODE 连续时间基线 |
| `mims-harvard/Raindrop` | <https://github.com/mims-harvard/Raindrop> | 不规则多变量时序表示学习，带多种 baseline | 后续做对比实验或吸收局部设计 |
| `tslearn-team/tslearn` | <https://github.com/tslearn-team/tslearn> | 传统时间序列工具箱，含 DTW/soft-DTW 等 | 做离线核验、传统基线和分析工具 |

## 6. 对当前仓库的实际建议

### 最推荐的首轮组合

针对 `chronaris` 当前已经稳定的 `E0ExperimentSample -> AlignmentBatch` 输入协议，首轮最推荐参考：

1. `latent_ode`
2. `gru_ode_bayes`
3. `torchdiffeq`
4. `pytorch-softdtw-cuda`

原因：

1. 当前输入是离散观测点、mask 和 offset，不是已经构造好的连续控制路径
2. 选题报告和基金申请书都明确强调“连续求解 + 观测更新”逻辑
3. ODE-RNN / GRU-ODE 风格比 CDE 路线更贴近当前代码状态
4. 首轮原型更需要“能训起来”，而不是一次性把连续时间方法做满

### 当前不建议直接作为首选主骨架的路线

1. 直接以 `torchcde` 为主骨架
2. 直接上 variational `Latent ODE` 全量 VAE 版本
3. 一上来就引入 `SDE` 或大而全连续学习工具箱

原因不是这些路线不好，而是它们与当前仓库的输入协议、依赖状态和阶段目标不完全一致。

## 7. 与阶段划分的对应关系

| 研究主题 | 更贴近的参考仓库 | 当前仓库阶段 |
| --- | --- | --- |
| 双流连续潜态建模 | `latent_ode`, `gru_ode_bayes`, `torchdiffeq` | `阶段 E` |
| 物理一致性约束 | `torchdiffeq`, `torchdyn`, `torchsde` | `阶段 F` |
| 因果掩码跨模态融合 | `mTAN`, `Raindrop` 仅可借鉴局部结构 | `阶段 G` |
| 传统对齐/核验基线 | `tslearn`, `pytorch-softdtw-cuda` | `阶段 E/I` |

## 8. 当前结论

进入 `阶段 E` 时，最务实的做法不是在所有连续时间路线中平均发力，而是先用：

- `ODE solver`
- `观测更新`
- `重构损失`
- `最小对齐损失`

把第一版 `双流连续对齐原型` 跑通。

在这个前提下，后续再按路线图把：

- 物理一致性约束
- 因果掩码融合
- 标准化特征导出

分别推进到 `阶段 F / G / H`。
