# Stage E Execution Plan

## 1. 目的

这份文档用于把 `阶段 E：连续对齐模型原型` 的实现顺序、环境策略和阶段性产物写死，降低跨对话和跨机器切换时的上下文丢失风险。

它重点回答：

1. 阶段 E 先做什么，后做什么
2. 哪些工作可以先在当前本地 CPU 环境完成
3. 什么时候切到 WSL Ubuntu + GPU 环境
4. 每完成一小步后，应该把哪些事实沉淀回仓库文档

## 2. 当前前提

截至当前，仓库已经具备：

1. `E0ExperimentSample`
2. `AlignmentBatch`
3. overlap-focused preview 的真实样本
4. 进入 `阶段 E` 的设计草案

当前仍未完成：

1. 导出并核验共享参考时间轴上的中间态
2. 处理 vehicle stream 的尺度问题或 loss weighting 问题
3. 固化最小实验摘要输出
4. 基于修正后的目标再跑一轮真实训练回归
5. 真实 GPU 环境下的训练与回归验证

当前环境探针（`2026-04-17`）已确认两组事实：

1. 本地 Windows `chronaris` 环境中 `numpy` / `torch` 已安装
2. 但当前本机相关二进制运行时并不稳定，不适合作为可靠的 `numpy/torch` runtime 验证环境
3. 实验室服务器 WSL Ubuntu 22.04 已完成 `chronaris` 训练环境准备
4. 当前服务器环境路径为 `/home/wangminan/env/anaconda3/envs/chronaris`
5. 在真实 shell 中已经确认：
   - `numpy 2.4.4`
   - `torch 2.11.0+cu130`
   - `torchdiffeq` 已安装在该 env 的 `site-packages`
   - `torch.cuda.is_available() == True`
   - 可见 GPU 为 `NVIDIA GeForce RTX 4090`
6. 当前服务器实际环境版本高于仓库中 `chronaris-stage-e-gpu.yml` 记录的目标版本
7. 如果后续需要严格复现，应补一次环境文件回写或重建校准

这意味着当前环境分工已经从“本地 CPU 开发 + 远程训练环境待准备”切换为“本地 CPU 辅助开发 + 服务器 GPU 主环境已就绪”。

当前代码探针（`2026-04-17` 夜）补充确认：

1. `ChronologicalSplitConfig / split_e0_samples_chronologically` 已实现并有单元测试
2. `ReferenceGrid` 构造器已实现并有单元测试
3. `AlignmentBatch -> TorchAlignmentBatch` 适配已实现并有单元测试
4. 最小确定性双流 `ODE-RNN` forward 原型已实现
5. 最小 reconstruction loss 已实现
6. 共享参考时间轴上的最小投影对齐损失已实现
7. 最小 preview `train / validation / test` pipeline 已实现并有单元测试
8. 已基于真实 overlap-focused E0 样本完成一轮最小训练回归
9. 当前已确认 `alignment loss` 下降，但 vehicle reconstruction loss 量级过大并主导 total loss

需要额外记录的是：

1. 当前这台 Windows 机器上，`numpy` / `torch` 的包安装虽然成功
2. 但相关二进制运行时并不稳定，至少在当前机器上不适合作为可靠的 `numpy/torch` runtime 验证环境
3. 因此仓库中的 `numpy/torch` 运行时测试默认只在显式启用时执行，避免本机环境把无关流程直接打崩

另外，本地机器上的 `conda` 配置显示：

1. `defaults` 使用清华镜像
2. `pytorch` / `conda-forge` 也通过清华镜像映射
3. shell 中存在 `HTTP_PROXY` / `HTTPS_PROXY`

因此如果后续再次出现安装卡顿，优先怀疑：

1. `conda-forge` 求解过慢
2. 镜像同步节奏导致的小包元数据延迟
3. 本地代理变量影响包管理器访问路径

针对 `torchdiffeq`，当前仓库默认目标仍然约定为：

1. `numpy` / `pytorch` 继续优先走 `conda`
2. `torchdiffeq` 通过环境文件中的 `pip` 子段安装

## 3. 环境策略

### 3.1 开发环境分工

当前默认采用“两阶段环境”：

1. **本地 Windows + `chronaris` conda 环境**
   负责纯 Python 模块、协议、切分、配置、测试和不依赖 GPU 的代码开发
2. **远程 WSL Ubuntu 22.04 + GPU**
   当前已完成依赖安装与 GPU 验证，后续负责 `torch` 前向开发、真实训练、长时间实验和性能回归

### 3.2 GPU 环境优先级

后续真实训练默认优先：

1. **实验室 4090**
2. 家里工作站 4070 Ti

原因：

1. 4090 更适合作为首个稳定训练环境
2. 4070 Ti 可作为备用环境或快速回归环境

### 3.3 环境切换原则

在下面这些工作完成前，默认不切 GPU：

1. 切分策略稳定
2. 前向张量协议稳定
3. 单元测试覆盖基础 shape / mask / offset 行为
4. 至少有一个最小前向原型能在 CPU 上跑通

只有在开始以下工作时，再切到 WSL Ubuntu：

1. 安装 `torch`
2. 安装 `torchdiffeq`
3. 开始真实训练 loop
4. 需要长时间 ODE 积分或中间态导出验证

当前状态补充（`2026-04-17` 晚）：

1. 上述切换条件中的依赖安装已经完成
2. 服务器环境已经不再是阶段 E 的阻塞项
3. 后续真实 preview 训练回归和中间态检查默认在服务器环境推进

### 3.4 跨机器同步原则

如果后续在服务器重新开会话，默认依赖：

1. 当前仓库中的 `docs`
2. 当前仓库中的 `AGENTS.md`
3. git 仓库中的最新代码与文档

因此每完成一个小里程碑，都应至少同步更新：

1. 相关实现代码
2. 对应测试
3. 一份简短说明性文档或路线图状态

## 4. 阶段 E 的执行顺序

### E-1：计划固化与文档沉淀

目标：

1. 固化阶段 E 的实现边界
2. 固化环境策略
3. 固化首轮里程碑顺序

当前状态：

- 已开始

对应文档：

1. [stage-e-prototype-design.md](D:/code/chronaris/docs/models/stage-e-prototype-design.md)
2. [stage-e-reference-repos.md](D:/code/chronaris/docs/models/stage-e-reference-repos.md)
3. 本文档

### E-2：CPU 安全基础模块

目标：

1. 固化时间顺序切分策略
2. 先写不依赖 `torch` 的工具模块
3. 用单元测试锁定行为

本阶段建议先完成：

1. `E0ExperimentSample` 的时间顺序切分工具
2. gap-aware split 行为定义
3. 参考时间轴生成的纯 Python / `numpy` 工具
4. 对应单元测试

退出条件：

1. 切分逻辑稳定
2. shape / 边界行为有测试
3. 不依赖 GPU 即可验证

### E-3：最小前向原型

目标：

1. 完成 `torch` 侧 batch 适配
2. 先做单流前向
3. 再扩展成双流前向

本阶段建议顺序：

1. `AlignmentBatch -> torch batch`
2. 单流 encoder
3. 单流 continuous evolve + observation update
4. 单流 decoder
5. 双流封装
6. 共享参考时间轴投影

退出条件：

1. CPU 上能跑通一轮前向
2. 中间态 shape 全部稳定
3. 没有大面积 `NaN`

### E-4：最小损失与训练 loop

目标：

1. 加入重构损失
2. 加入最小时间对齐损失
3. 搭建最小训练/验证脚本

本阶段建议顺序：

1. reconstruction loss
2. alignment loss
3. metrics summary
4. preview train loop
5. preview validation loop

退出条件：

1. 至少完成一次真实训练
2. loss 能下降
3. 可导出最小中间态

### E-5：WSL + GPU 回归验证

目标：

1. 在远程 Ubuntu 环境安装依赖
2. 完成真实训练
3. 输出第一轮实验记录

本阶段产物至少包括：

1. 依赖安装说明
2. 训练命令
3. 关键 loss 曲线或摘要
4. 一份中间态检查说明

## 5. 当前默认编码入口

为了避免一开始直接展开大量模型代码，当前默认先从下面两项开始：

1. **时间顺序切分**
2. **纯 CPU 安全工具与测试**

原因：

1. 这是路线图里明确优先项
2. 这部分不依赖 GPU
3. 这部分出错成本低
4. 这部分一旦稳定，后续训练代码就少一层不确定性

## 6. 当前不提前做的事

1. 不先写完整训练框架
2. 不先写 `阶段 G` 的因果融合
3. 不先写 `阶段 F` 的全量物理残差
4. 不先为了远程环境改写大量工程结构
5. 不先因为将来上服务器，就把当前代码写成“环境驱动”的复杂方案

## 7. 文档更新规则

从当前开始，阶段 E 每推进一个小里程碑，至少更新下面之一：

1. `docs/planning/coding-roadmap.md`
2. `docs/planning/implementation-summary-*.md`
3. 一份新的阶段说明文档

推荐原则：

1. **设计变化**，更新设计文档
2. **执行顺序变化**，更新本文档
3. **代码能力新增**，更新 implementation summary
4. **阶段状态变化**，更新 coding roadmap

## 8. 当前结论

阶段 E 的执行方式已经固定为：

1. 先文档沉淀
2. 再 CPU 安全模块
3. 再最小前向原型
4. 再训练 loop
5. 服务器 GPU 环境已完成验证，后续默认在该环境继续 E-3 / E-4

当前下一步就是：

**导出共享参考时间轴上的中间态，并处理 vehicle stream 的尺度或 loss weighting 问题。**
