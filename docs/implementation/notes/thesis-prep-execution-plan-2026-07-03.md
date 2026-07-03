# 毕业论文阶段执行计划

日期：2026-07-03

状态：文档细化版，尚未启动清理、删除、历史改写或新实验。

## 0. 固定前提

后续毕业论文阶段默认采用以下前提：

- 不再把鼎新新增一手数据或新增专家评价数据作为执行依赖。
- 若答辩或论文中提到继续争取外部数据，只能写成“若后续可获得则作为附加验证”，不能作为主线闭合条件。
- 现有私有真实双流数据仍是主证据来源：2 个 sortie、3 个双流 view、111 个窗口、333 条 weak-label task。
- 公开 UAB/NASA 仍是 public adapter / context-proxy 泛化与校准证据，不等价于私有真实航电流。
- 仿真数据允许进入附录型实验，但必须标注为 synthetic stress-test，不替代真实数据或专家真值。
- LLM 允许扩展，默认复用 DeepSeek v4-pro 现有链路；LLM 输出只能作为 preprocessing context、rubric、semantic hints、runtime explanation、synthetic scenario card 或 review packet，不直接作为专家真值。
- 当前先更新文档，不执行清理、不删除 artifact、不改写 git history、不跑新实验。

## 1. 总体顺序

建议采用以下执行顺序：

1. `P38` 论文协议冻结：统一实验矩阵、结果表 schema 和 claim boundary。
2. `P42` 仓库收敛清理：先审计、备份和记录，再删除/拆分/历史瘦身。
3. `P39` 仿真数据集：实现 synthetic generator、审计和附录型 stress test。
4. `P40` T3/public 指标提升：在清理后的仓库中定点优化 retrieval 与 public route。
5. `P41` 论文级消融统一：把 private/public/model/ablation 聚合为统一长表、短表和图。
6. `P43` 论文材料化：整理方法、实验、图表、边界、答辩问答和复现实验包。

此顺序的理由是：如果先继续跑实验，会继续扩大已有 nested artifact 和日志膨胀；如果先清理但没有协议矩阵，容易误删当前论文需要的证据。因此先冻结协议，再清理，再实验。

## 2. P38 论文协议冻结

目标：把现有 P30/P31/P32/P34/P35/P36/P37 证据冻结为论文可引用的统一协议矩阵。

应产出：

- `docs/artifacts/assets/stage_i_thesis_protocol/<run_id>/experiment_registry.csv`
- `docs/artifacts/assets/stage_i_thesis_protocol/<run_id>/result_matrix_long.csv`
- `docs/artifacts/assets/stage_i_thesis_protocol/<run_id>/result_matrix_summary.csv`
- `docs/artifacts/assets/stage_i_thesis_protocol/<run_id>/claim_boundary_table.csv`
- `docs/artifacts/stage_i/stage-i-thesis-protocol-<run_id>.md`

矩阵字段：

| field | 含义 |
| --- | --- |
| `evidence_quadrant` | private_model_comparison / private_component_ablation / public_model_comparison / public_component_ablation |
| `dataset_role` | private_real_dual_stream / private_proxy / public_context_proxy / synthetic_stress_test |
| `task` | T1 / T2 / T3 / NASA / UAB n_back / UAB heat_the_chair |
| `split_protocol` | leave_one_view_out / leave_one_sortie_out / LOSO / fixed_public_split / synthetic_domain_shift |
| `model_or_component` | Chronaris variant、baseline 或 ablated component |
| `metric` | macro-F1、balanced accuracy、RMSE、MAE、top-k、MRR 等 |
| `value` | 指标值 |
| `baseline` | 对照方法或 reference artifact |
| `delta_positive_is_better` | 统一方向后的变化量 |
| `seed_count` | 随机种子数 |
| `artifact_path` | 可追溯 CSV/JSON/manifest |
| `claim_boundary` | 论文可写边界 |

P38 不重跑实验，只读已有 artifact。验收标准是：任何论文图表都能从 P38 的 registry 或 matrix 反查到当前 artifact root。

## 3. P42 仓库收敛清理

目标：先把仓库恢复到适合长期论文实验的结构，再跑新实验。

清理分四层执行，必须每层先审计再删除。

### 3.1 清理前审计

应产出：

- `docs/artifacts/cleanup/20260703-thesis-prep-cleanup-inventory.md`
- tracked artifact size table
- current-entry reference table
- deletion candidate table
- external backup manifest
- `.git` / LFS footprint report

审计命令方向：

- `git status --short`
- `git log --oneline --decorate -8`
- `du -sh docs/artifacts docs/artifacts/assets src scripts tests .git`
- `find docs/artifacts/assets -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 du -sh | sort -hr`
- `git count-objects -vH`
- `du -sh .git/lfs .git/objects`
- `git lfs status`
- `git lfs migrate info --include-ref=refs/heads/main --include='docs/**'`

### 3.2 可直接清理项

这些项只要审计确认即可删除：

- `__pycache__`
- `.pytest_cache`
- 临时 scratch
- 已由当前 summary 接管、且不被 docs/code 引用的 partial CSV
- 已外置备份且 manifest 标记为 pruned 的 checkpoint binary

### 3.3 需要备份后删除的 tracked artifact

候选包括但不限于：

- P37 nested public/private 中可由 top-level summary、CSV、figures 复现的 child logs / training curves / dense rows。
- P35/P31/P28 中已经由上层 summary 接管的 per-candidate child outputs。
- 202605 历史 private/public opt 包中不再作为当前入口的 raw-ish payload。
- 重复 task manifest / training curves，若 checksum 与 canonical 文件一致，可替换为说明或仅保留 canonical。

删除前必须：

1. 用 `rg` 查引用。
2. 将被删路径复制到 `/home/wangminan/projects/chronaris-local-artifacts/cleanup-20260703/`。
3. 写 backup manifest。
4. 更新 `docs/artifacts/cleanup/` 记录。
5. 更新 `ARTIFACTS.md` 和 `stage_i/README.md`，把 current link 改为 summary/manifest 而不是 deleted path。

### 3.4 Git history / LFS 瘦身

只有在当前树清理后仍确认 `.git` 或 publishable docs history 异常膨胀，才执行历史改写。

执行前置条件：

- 工作树改动已审阅。
- 外置备份完成。
- 远端 SHA 已记录。
- 用户已允许 force-with-lease 形态发布。
- `git filter-repo` 可用或已确认安装方式。

验证标准：

- `git lfs migrate info --include-ref=refs/heads/main --include='docs/**'` 显示 docs 历史体积合理。
- `.git/lfs` 本地缓存已 prune 到合理水平。
- `git diff --check` 通过。
- `git lfs status` 正常。
- 若后续 push，必须使用 `--force-with-lease` 而不是裸 force。

本计划阶段不执行历史改写。

## 4. P39 仿真数据集与附录实验

目标：构造可解释、可审计、可复现的 synthetic stress-test dataset，用于验证机制而非替代真实证据。

### 4.1 数据生成原则

- 以 Stage H schema 为上限，不创造论文主线没有定义的输入接口。
- 使用脚本生成连续数值时序；LLM 不直接生成高频数值流。
- 显式模拟多速率、不规则采样、缺失、噪声、飞行阶段、事件滞后、生理响应延迟和个体差异。
- 为 synthetic 数据生成 oracle labels，但标签只能称为 simulation oracle，不称为专家真值。
- 所有输出写入新 run_id，并带 `synthetic=true`、`generator_version`、`seed`、`source_stats_path`、`known_causal_graph`。

### 4.2 建议生成器模块

建议新增：

- `src/chronaris/pipelines/stage_i/synthetic/`
- `scripts/stage_i/synthetic/build_synthetic_dataset.py`
- `scripts/stage_i/synthetic/audit_synthetic_dataset.py`

生成器组件：

- `schema_sampler`：读取现有 Stage H feature schema。
- `flight_phase_simulator`：生成 climb / cruise / maneuver / event-like phase。
- `vehicle_dynamics_simulator`：生成速度、高度、加速度、垂向速度和约束残差。
- `physiology_response_simulator`：生成滞后响应、个体基线、噪声和疲劳项。
- `event_context_simulator`：生成语义事件 token 和 context proxy。
- `oracle_task_builder`：生成 T1/T2/T3 和 public-like labels。
- `domain_shift_builder`：生成跨 sortie、跨 pilot、跨传感器缺失的 stress split。

### 4.3 仿真数据审计

必须输出：

- marginal distribution check
- autocorrelation / cross-correlation check
- physical residual check
- lag-response plausibility check
- missingness/noise profile
- label-feature overlap audit
- train/test domain-shift summary
- synthetic-vs-real discriminator smoke

论文写法：附录或补充实验中写“仿真压力测试表明方法在已知机制下能恢复/区分某类结构”，不能写“仿真证明真实飞行数据结论”。

## 5. P40 T3/public 指标提升

目标：在清理后的仓库上做定点优化，不做无边界大搜索。

### 5.1 T3 retrieval

当前状态：

- P37 T3 未超过 P34。
- 后续若继续优化，应集中在 retrieval 机制。

候选方向：

- hard negative mining
- InfoNCE temperature / hard-negative weight 小网格
- event-semantic positive pair
- candidate pool 分层
- split-aware retrieval calibration
- retrieval-specific projection head
- 防止 sample identity / window position leakage 的 audit

验收：

- 不只看 top1；同时看 top3、top5、MRR。
- 保留失败结果。
- 不改变 P34/P37 reference。
- 若无提升，论文中把 T3 写成方法边界和 future work。

### 5.2 public route

当前状态：

- P37 public route accepted。
- public 仍是 context-proxy。

候选方向：

- adaptive context gate confirm
- UAB n_back loss weighting / target transform
- NASA/UAB route-specific calibration
- context-only vs physiology-only vs fused route 的统一对比

验收：

- NASA macro-F1 / balanced accuracy 和 UAB RMSE / MAE 分开看。
- 不把 public context proxy 写成真实航电流。
- 所有结果进入 P38/P41 矩阵，而不是孤立报告。

## 6. P41 论文级消融统一

目标：把现有消融从“工程诊断”收束为“论文实验章节”。

统一消融主题：

1. 连续时间双流对齐 vs 朴素同步 / 单流。
2. 物理残差对 T2 的贡献。
3. 因果掩码和 lag window 对方向约束的贡献。
4. 语义事件对 T3 候选排序的贡献。
5. task-aware heads 对 T1/T2 的贡献。
6. stream-role gate 对 private/public route 分离的贡献。
7. synthetic stress-test 对机制恢复能力的补充说明。

输出：

- 统一长表。
- 论文短表。
- 每个组件一张小图或一组子图。
- 每个结论一条 claim boundary。

## 7. P43 论文材料化

目标：把仓库证据转化为毕业论文材料，而不是继续堆实验目录。

应产出：

- 方法章节变量表。
- 模型结构图和公式说明。
- 数据与任务定义表。
- 真实私有数据实验表。
- 公开 context-proxy 实验表。
- 仿真附录实验表。
- 消融与案例分析图。
- 可复现性说明。
- 答辩问答口径。

写作边界：

- 私有真实双流：主线证据。
- weak-label：任务原型和机制验证，不是人工真值。
- public：adapter / calibration / context-proxy。
- synthetic：附录型 stress test。
- LLM：preprocessing / review / explanation / rubric，不是专家真值。

## 8. LLM 扩展边界

可扩展用途：

- 字段语义归一。
- weak-label 规则复核。
- synthetic scenario card 生成。
- expert rubric 草案。
- runtime explanation。
- case-study narrative draft。
- human review packet 生成。
- 论文图表说明草稿。

禁止或需要人工确认后才可做的用途：

- 直接生成高频数值时序并写成真实数据。
- 直接生成专家标签并写成专家真值。
- 改写现有 weak-label 而不保留 `label_unchanged` 或 change audit。
- 把 LLM hints 写成核心因果证据。
- 未审计 payload 就发送原始全量高频数据。

## 9. 执行开关

当前只更新文档。后续若用户要求开始执行，建议第一条明确指令是：

> 启动 P38 + P42，只做协议冻结和清理审计，不删除任何 tracked artifact。

完成审计后，再由用户确认是否进入删除、外置备份和历史改写。
