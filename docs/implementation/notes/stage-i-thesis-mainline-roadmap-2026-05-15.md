# Stage I 论文主线重构 Roadmap

更新时间：2026-05-15

## 1. 目的

这份文档只回答一件事：

- 在 `Stage I historical closure`、`chronaris_opt` 和 `public opt closed` 都已经存在的前提下，下一阶段应该如何把仓库收敛成更贴近选题报告的论文主线

它不覆写历史收口，也不把 `proxy / adapter` 结果包装成已经完成的 thesis 本体。

## 2. 当前判断

当前仓库已经证明了两件事：

1. `chronaris` 已经是一个真实可跑、可导出、可复现实验的研究原型体系。
2. 但它还没有完全收敛成“统一骨干、受任务牵引、可部署推理、能够直接映射论文表述”的主线实现。

当前差距不在 `access / schema / dataset / timebase` 这些底座，而集中在下面七类问题：

1. 公开数据分支的第二模态是 `task_context / scenario_context` 代理流，不是论文里严格意义上的航电流。
2. `Stage H` 已补上 `fixed checkpoint inference -> export` 路径，但它还不是完整 runtime inference。
3. `E/F/G/H` 骨干与 `Stage I` 下游任务已经补上最小联合训练闭环，但仍停留在 weak-label Phase C 首轮。
4. `Stage F` 当前是 `weak physics family`，还不足以支撑“显式刚体动力学残差”这一强表述。
5. `Stage G` 当前是 `minimal causal attention`，还没有语义查询向量、事件 token 与事件级归因层。
6. 私有主线虽然已经补出 `risk_proxy / workload_proxy / event_replay_tag` weak-label builder，但仍和论文里的风险/负荷/复盘人工真值任务没有完全对齐。
7. `serving/runtime_demo.py` 仍是离线报告入口，不是实时或准实时推理引擎。

## 3. 下一阶段总目标

下一阶段不再追求“多跑几轮旧 benchmark”，而是把主线收敛成下面这条链：

1. 固定一个可复用的双流 backbone。
2. 用真实或弱标签 thesis task 牵引统一训练。
3. 通过 frozen checkpoint 做多 sortie 一致导出和离线/在线推理。
4. 在 support、benchmark、runtime 三条证据线上都明确区分：
   - `历史 closure`
   - `public adapter evidence`
   - `private proxy evidence`
   - `thesis mainline`

## 4. 分阶段推进

### Phase A：主线边界校准

目标：

- 先把“代码现在到底证明了什么”说清楚，再继续扩方法。

本阶段重点：

- 明确公开分支是 `context proxy / adapter evidence`，不与真实双流主线混写。
- 明确 `T1/T2/T3` 是 `proxy tasks`，不直接等价于论文任务本体。
- 清理 `planning / docs / reports / AGENTS` 中把旧计划当现行主线的表述。

退出条件：

- 论文口径、代码命名、报告索引三处一致。
- `planning` 根目录只保留现行路线图与 closure，旧 `stage-xxx-plan` 全部归档。

### Phase B：统一骨干与 checkpoint 导出

目标：

- 把 `Stage H` 从“每个 view 现场重训后导出”改成“固定骨干统一推理导出”。

本阶段重点：

- 新增 backbone 训练入口。
- 为 `Stage H export` 增加 `checkpoint_path + inference_only` 模式。
- 在 `feature_bundle` / `run_manifest` 中补足 checkpoint lineage。

退出条件：

- 同一个 checkpoint 能对多 sortie / 多 view 做一致导出。
- `Stage H` 主线导出不再依赖 per-view 训练。

### Phase C：联合训练与 thesis task builder

目标：

- 让 `E/F/G/H` 骨干和下游任务形成真正的训练闭环。

当前状态：

- 已完成首轮代码/测试收敛；当前已经具备 `task_heads`、`L_task + L_causal`、`thesis weak-label builder` 与最小 multitask smoke path。

本阶段重点：

- 新增统一 task heads 与 multitask trainer。
- 把 `proxy task` 和 `thesis task` builder 拆开。
- 先用私有弱标签任务打通 `L_recon + L_align + L_phy + L_causal + L_task`。

退出条件：

- frozen backbone 与 multitask backbone 的指标和导出 contract 都可复现。
- 至少有一组 `thesis task proxy` 可以直接由联合训练模型输出，而不是完全靠导出后外接 head。

### Phase D：方法体补强

目标：

- 把当前“最小原型”提升到更贴近论文方法表述的实现层级。

本阶段重点：

- `Stage F` 新增 `rigid_body_family` 和显式状态映射/残差。
- `Stage G` 新增 `SemanticQueryBank / EventTokenExtractor / CausalEventFusion`。
- support 报告开始输出事件级归因，而不是只保留时间步 attention。

退出条件：

- `physics` 和 `fusion` 两条 support 报告都能直接对应选题报告中的关键术语。

### Phase E：runtime inference

目标：

- 补齐离线飞参回放和在线/准在线传感器流的推理入口。

本阶段重点：

- 新增流式窗口缓存与 checkpoint 推理入口。
- 输出任务预测、attention/event 解释和关键贡献摘要。
- 保留 `runtime_demo` 作为论文展示工具，不再把它写成实时引擎。

退出条件：

- 至少一条 CLI replay/inference 路径可以对窗口流增量输出预测和解释。

## 5. 优先级顺序

默认按下面顺序推进：

1. `Phase A/B/C`：首轮代码收敛已完成，当前以回归维护为主。
2. `Phase D`：再强化 physics / semantic fusion。
3. `Phase E`：最后补 runtime inference。

原因很直接：

- `Phase A/B/C` 已经解决“当前证据是否在同一条主线、是否已有最小联合训练闭环”。
- `Phase D` 再解决“方法体是否真正按论文术语落在代码里”。
- `Phase E` 最后解决“是否具备部署/演示语义”。

## 6. 当前不做的事

- 不继续扩大 `UAB` 的 CPU-heavy 搜索空间。
- 不把 `UAB robust-prior adapter` 写成双流融合本体胜利。
- 不把 `T1/T2/T3` 直接写成人工真值任务最优。
- 不提前重写上游接收器、入库链路或原始大文件治理。
- 不删除 `E/F/G/H` 收口工件；它们仍是历史基线和导出依赖。

## 7. 执行入口

当前执行顺序应以这三份文档为准：

1. [coding-roadmap.md](coding-roadmap.md)
2. [stage-i-thesis-mainline-coding-plan-2026-05-15.md](stage-i-thesis-mainline-coding-plan-2026-05-15.md)
3. [thesis-coding-gap.md](thesis-coding-gap.md)
