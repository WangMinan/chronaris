# 毕业论文编码缺口评估

更新时间：2026-05-15

## 1. 目的

这份文档只回答一件事：

- 对照 `docs/requirements/选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx`、当前仓库实现和已落盘证据，论文主线在编码层面还差什么

它不覆写历史 closure，也不把 `proxy / adapter` 结果包装成已经完成的 thesis 本体。

## 2. 当前已经成立的事实

当前可以明确成立的结论是：

- `E / F / G(min) / H` 已完成真实数据链路、导出 contract 与测试闭环。
- `Stage I Phase 0 + Phase 1 + Phase 2 + Phase 3` 的公开 benchmark 历史收口已完成。
- `chronaris_opt` 已在鼎新 `proxy benchmark` 的 分类任务、回归任务和检索任务 三任务上达到当前对照矩阵最优，并固化 package。
- `alignment support`、`causal support`、`fixed six-path ablation` 三份 support 已落盘。
- `runtime/demo` 与 `anchor` 已经提供 thesis-facing 的离线展示入口。
- `public opt closed` 仍可作为公开支撑证据，但它主要是 `public adapter / calibration evidence`，不是 thesis 双流主线已经完全收口的证据。
- `Stage I thesis mainline Phase A` 已完成首轮代码/测试收敛：公开第二模态与鼎新 分类任务、回归任务和检索任务 的 thesis-facing 边界已经在 contract、metadata 与报告生成代码里统一。

因此，当前真正没完成的已经不是“系统完全跑不起来”，而是“现有实现是否已经和论文主线一一对应”。

## 3. 当前七类关键缺口

### 3.1 公开分支的第二模态仍是弱监督流

当前 `UAB` 与 `NASA` 公共分支使用的是 `task_context / scenario_context`，而不是论文里严格意义上的航电/战术环境连续流。

这意味着：

- 它可以作为 `public adapter evidence`。
- 但它不能单独证明“生理流 + 航电流”的 thesis 双流本体已经被公开数据严谨复现。

### 3.2 Stage H 已补上 frozen backbone inference export 基础

当前仓库已经补上：

- `stage_i_backbone_train` 可复用骨干训练入口
- `AlignmentPreviewPipeline` 的 checkpoint 保存 / 加载与 `inference_only` 运行
- `Stage H export` 的 `checkpoint_path + inference_only`
- `run/view manifest` 与 `load_stage_h_feature_run()` 的 `export_mode / backbone_lineage`

这意味着：

- `Stage H` 已不再必须绑定 `per-view training` 才能导出。
- 论文主线已经具备 `frozen checkpoint inference export` 的代码基础。
- 当前剩余缺口不再是“能不能做 checkpoint inference”，而是“backbone 是否已经和真实 thesis task 联合训练闭环”。

### 3.3 骨干和下游任务已补上最小联合训练闭环

当前已落地：

1. `src/chronaris/models/alignment/task_heads.py` 已统一 `classification / regression / retrieval` head contract。
2. `build_stage_e_objective()` 已支持把 `L_task` 与 `L_causal` 汇总进总目标。
3. `src/chronaris/dataset/stage_i_real_task_builders.py` 已补出 `risk_proxy / workload_proxy / event_replay_tag` 三类 thesis-facing weak-label tasks。
4. `src/chronaris/pipelines/stage_i/training/multitask_train.py` 已可在最小 synthetic/Dingxin-style 样本上跑通一次 `backbone + causal fusion + task heads` 端到端训练并落盘。
5. `stage_i_private_benchmark` 已把 weak_label_task_evidence 与 `thesis_task_evidence` 分层写入 summary/manifest。

这意味着：

- 论文里“任务牵引的统一损失”已经有了第一版代码落点。
- 但当前 thesis-facing 任务仍然是 `weak-label / proxy thesis task`，不是人工真值闭环。
- 当前很多历史最优结果仍来自外接 head、residual 和 calibration，而不是已经完成 Phase D/E 之后的 backbone 本体。

### 3.4 Stage F 仍是 weak physics family

当前 `Stage F` 已经有价值，但核心仍是：

- `speed <-> acceleration`
- `altitude <-> vertical_speed`
- `attitude <-> angular_rate`
- smoothness / envelope / pairwise / latent fallback

它足以支撑“物理一致性约束”，但还不够支撑“显式刚体动力学残差 / rigid-body family”的强表述。

### 3.5 Stage G 仍停留在 minimal causal attention

当前 `Stage G` 已经实现：

- 当前生理状态作为 `Query`
- 历史第二流状态作为 `Key/Value`
- 非对称因果 mask
- 基于相邻 hidden state 变化范数的 event score

但它还没有：

- `SemanticQueryBank`
- `EventTokenExtractor`
- `query-to-event attention`
- 事件原型级解释

### 3.6 鼎新真实数据任务已补出 thesis weak-label layer，但仍不能替代人工真值任务

当前同时存在两层鼎新真实数据任务：

- 分类任务、回归任务和检索任务：`private proxy benchmark / proxy tasks`
- `risk_proxy / workload_proxy / event_replay_tag`：`thesis weak-label tasks`

但这两层都不能直接替代：

- 空中失能风险分析
- 认知负荷评估
- 飞行事件复盘

### 3.7 serving 仍是 demo，不是 runtime inference

当前 `runtime_demo.py` 的职责是读取已冻结资产并生成总结，不负责：

- 流式接入
- 增量分窗
- checkpoint 推理
- 在线解释输出

所以它可以支撑展示和报告，但还不能支撑“近实时推理引擎”这层论文表述。

## 4. 当前优先级

### 第一优先级：主线边界校准已完成

已完成三件事：

1. 公开分支降格为 `public adapter evidence`
   当前状态：已完成首轮代码/测试收敛，后续新报告与导出默认沿用该口径
2. `Stage H` 改为 `checkpoint inference export`
   当前状态：已完成首轮代码/测试收敛，后续 thesis-facing 导出默认沿用该口径
3. `proxy task` 和 `thesis task` 明确拆开
   当前状态：已完成 `proxy contract / weak-label thesis-task builder / summary layering` 首轮收敛

当前最重要的未完成项已经前移到第二优先级：补 `Stage F rigid-body family` 与 `Stage G semantic event fusion`，而不是继续停留在 Phase C skeleton。

### 第二优先级：再补方法体

然后再完成三件事：

1. `Stage F rigid_body_family`
2. `Stage G semantic event fusion`
3. 把 weak-label thesis task 从最小 smoke 提升到更强的真实资产/评测路径

这是把研究原型收敛成论文方法主线的关键阶段。

### 第三优先级：最后补 runtime inference

最后再做：

1. 流式窗口缓存
2. checkpoint 推理
3. 预测 + attention/event 解释输出

## 5. 当前不应再沿用的旧判断

下面这些判断已经不再准确：

- “为了毕业论文，剩余工作主要是图表整理和 fairness confirm。”
- “`public opt closed` 基本等于论文公开主线已经完成。”
- “`chronaris_opt` 鼎新最优 package 已经足以证明 thesis 主线 fully closed。”
- “`runtime_demo` 已经可以视作实时推理入口。”

更准确的判断是：

- 当前已经有很强的历史证据和支撑资产。
- 但 thesis mainline 仍需要一次结构化重构与收口。

## 6. 当前执行入口

当前应以这三份文档为准：

1. [coding-roadmap.md](coding-roadmap.md)
2. [stage-i-thesis-mainline-roadmap-2026-05-15.md](stage-i-thesis-mainline-roadmap-2026-05-15.md)
3. [stage-i-thesis-mainline-coding-plan-2026-05-15.md](stage-i-thesis-mainline-coding-plan-2026-05-15.md)

历史 `stage-xxx-plan` 已统一归档到：

- [archive/stage_i/README.md](archive/stage_i/README.md)
