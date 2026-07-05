# 毕业论文准备评估

日期：2026-07-03

## 评估依据

- 当前仓库 HEAD：`6228787 feat: add stage i optimized final polish`。
- 初始工作区状态：`git status --short` 无输出；后续仅新增/更新毕业论文准备文档。
- 本轮读取：`docs/STATE.md`、`docs/implementation/TASKS.md`、`docs/requirements/SPEC.md`、`docs/artifacts/ARTIFACTS.md`、`docs/artifacts/stage_i/README.md`、`docs/midterm/claims-matrix-2026-06-13.md`、`docs/midterm/boundaries-and-risks-2026-06-13.md`。
- 附件读取：`王旻安-西北工业大学硕士学位论文中期考核表.docx` 经 DOCX XML 段落抽取；PDF 未作为主来源。
- 本轮未修改源码、未删除产物、未重跑实验。

## 总判断

中期前的代码和产物已经足够支撑“原型闭环”和“中期报告叙事”，但还不足以直接支撑毕业论文最终实验章节。主要缺口不在于缺一个大而全的新模型，而在于：论文证据边界需要重新收束，实验矩阵需要统一成同一 task / split / leakage-safe protocol 下的可复查表格，模型优化要集中打 分类任务和检索任务 和 public route 的短板，消融需要从工程诊断提升为论文级因果论证，仓库需要在继续实验前做一次收敛清理。

新的前提必须进入后续计划：不要把“继续从鼎新获取更多飞行员生理、飞机总线或专家评价数据”作为主路径。它可以作为答辩措辞中的外部机会，但毕业论文实际执行应默认不会拿到新的一手数据。因此，主线应从“等待新增真实数据”改为“现有鼎新真实双流证据 + 公开 context-derived second-stream 泛化证据 + 严格边界内的模拟/弱标签补强”。

模拟数据是可行的，但只能承担两类角色：方法机制验证和压力测试。它不能替代鼎新真实双流数据，也不能伪装成专家评价真值。可接受做法是基于现有 Stage H schema、字段统计、飞行阶段/事件约束、物理残差和生理响应滞后构造脚本化 synthetic dataset；LLM 可以辅助生成场景说明、规则复核和专家评价 rubric 草案，但不应直接生成高频数值时序或被写成专家真值。

## 用户确认后的执行前提

2026-07-03 用户已确认以下前提，后续毕业论文阶段计划应以此为准：

1. 内部执行不再期待鼎新新增一手数据，也不再期待新增专家评价数据；答辩口径可以保留“若后续可获得则作为附加验证”。
2. 允许且需要引入仿真数据集；仿真数据可以进入附录型实验，但必须明确标注为 synthetic / simulation / stress-test evidence。
3. 允许调用 LLM，并优先复用现有 DeepSeek v4-pro 调用链路；可以扩展 LLM 用途，但仍需遵守数据外发、切片、审计和真值边界。
4. 当前时间充足，后续模型工作优先进一步提升 检索任务与 public route 指标，而不是只做论文包装。
5. 接受先清理仓库再跑新实验；清理目标是彻底收敛。允许删除 tracked 历史 artifact、允许外置备份、允许在必要时改写 git 历史。
6. 不修改附件中的中期考核表；后续只在仓库 docs、论文写作材料和实验计划中更新口径。

这些确认改变了后续优先级：先做文档化的执行计划和清理方案，再由用户确认启动实际清理；清理完成后再进入 检索任务与公开路线 指标提升、仿真数据和论文级消融。

## 当前成果能支撑什么

1. 鼎新真实双流主线已经具备论文原型证据：2 个 sortie、3 个双流 view、111 个窗口、333 条 weak-label task，覆盖数据组织、双流连续潜态、物理残差、因果掩码、语义事件、runtime contract。
2. 公开支撑线已经具备可引用价值：公开模型对比与公开融合刷新/公开融合消融/流角色融合/最终指标打磨 覆盖 NASA/UAB 的 public adapter / context-derived second-stream model comparison、fusion refresh、ablation 和 public route calibration。
3. 鼎新真实数据第三方模型对比、公开融合消融、跨证据矩阵、任务感知头优化、流角色融合、优化模型再评估与最终指标打磨 已经形成当前最接近论文实验矩阵的资产：Dingxin third-party comparison、public fusion ablation、cross-evidence matrix、task-aware heads、stream-role fusion、optimized summary 和 final polish。
4. 最终指标打磨 的局部结论是可写但必须限域：分类任务校准 accepted，public route accepted，检索任务 rejected 并沿用 任务感知头优化 retrieval；public 第二流仍是 context-derived second stream。

## 还缺什么

### 1. 论文证据协议冻结

需要先冻结毕业论文使用的 2x2 证据矩阵：

| 维度 | 鼎新真实双流 | 公开泛化/context-derived second-stream |
| --- | --- | --- |
| 模型比较 | 鼎新真实数据第三方模型对比 + 任务感知头优化与流角色融合/最终指标打磨 的 Dingxin 分类任务、回归任务和检索任务 | 公开模型对比与公开融合刷新 + 流角色融合/最终指标打磨 的 NASA/UAB |
| 组件消融 | leakage-safe Dingxin ablation + task-head/stream-role ablation | 公开融合消融 + public route/gate/context ablation |

这个矩阵必须统一字段：`task`、`dataset_role`、`evidence_role`、`split`、`protocol`、`metric`、`baseline`、`delta`、`seed_count`、`claim_boundary`。后续所有论文表格和图都应从这个矩阵派生，不再让读者穿透多层 nested artifact。

### 2. 模型优化要从广撒网改为定点补短板

当前最值得继续做的不是无限扩 candidate，而是三个定点：

- 鼎新 分类任务：最终指标打磨 已有小幅提升，可作为 calibration 收束；后续只需要稳定性/置信区间，不宜再大规模追指标。
- 鼎新 回归任务：任务感知头优化 已经给出明显改善，应以误差分解、物理残差贡献和跨 split 稳定性为主，不必优先继续调参。
- 鼎新 检索任务：仍是最弱项，最终指标打磨 没超过 任务感知头优化。若还要做模型工作，应集中在 hard negative、InfoNCE 温度/采样、事件语义 positive pair、候选池分层和 retrieval 指标稳定性，而不是同时改所有模块。
- 公开 route：最终指标打磨 public route 在 NASA/UAB 形成 accepted improvement，但仍是 context-derived second-stream。后续可围绕 adaptive context gate、loss weight、UAB n_back 误差来源做小范围 confirm。

### 3. 消融实验需要升级成论文级论证

现有消融已经很多，但论文说服力仍需要三件事：

- 同一协议：同一 task、同一 split、同一 leakage-safe 边界、同一随机种子数量，不混用历史满分 weak-label 与防泄漏结果。
- 同一表格：把 Dingxin/public、model comparison/component ablation 聚合成一个短表和一个长表。
- 同一结论边界：每个组件只回答对应问题，例如物理残差回答 回归任务 短期动态误差，语义事件回答 检索任务 候选排序，context gate 回答公开 context-derived second-stream route，不互相替代。

### 4. 模拟数据可以做，但要写成 synthetic stress test

建议设计一个单独 synthetic workflow，而不是把模拟数据混入主实验：

- generator 以现有 Stage H 字段规范为约束，输出 physiology stream、vehicle stream、event/context stream、oracle weak label、missingness/noise profile。
- 物理部分由脚本控制速度/高度/加速度/垂向速度残差，不能让 LLM 直接生成数值时序。
- 生理响应用可解释滞后函数和个体差异参数构造，使 分类任务、回归任务和检索任务 的 oracle 关系已知。
- 验证项包括边际分布、缺失模式、自相关/互相关、物理残差、事件滞后、label-feature overlap audit、训练/测试 domain shift。
- 报告中只写成“机制验证、鲁棒性和泄漏检查”，不能写成“新增真实飞行数据”或“专家评价数据”。

LLM 在这里的合适位置是：生成任务场景文字、字段语义描述、专家评价 rubric 草案、异常案例解释和复核 packet。LLM 不应作为专家标签真值来源，除非后续由真人确认并留下人工审核记录。

## 仓库收敛与清理评估

当前仓库已经明显膨胀。粗略盘点：

- `docs/artifacts/assets` 约 242 MB，当前最大目录包括 `stage_i_optimized_final_polish`、`stage_h`、`stage_i_stream_role_fusion`、`stage_i_private`。
- `.git` 约 8.2 GB，说明历史对象/历史 LFS 或曾经提交的大文件仍是主要膨胀来源；这需要单独确认后才能做历史清理。
- 源码与测试共约 85k 行，多个文件超过 800 行，违反当前协作说明中的拆分建议或硬约束。
- 本地仍存在 `__pycache__`，可清理但收益主要是卫生项。
- 仍有较多 tracked `run.log`、training curves、task manifest 和 nested summary，部分可能已经被上层 summary 接管。

优先清理顺序建议如下：

1. 文档事实同步：`docs/STATE.md` 与 `docs/implementation/TASKS.md` 的 Git/HEAD 描述仍有局部滞后，需先修正为 `6228787` 后再开展新任务。
2. 本地生成缓存：删除 `__pycache__`、`.pytest_cache`、临时 scratch，不影响历史。
3. 代码拆分：优先拆 `thirdparty_comparison.py`、`fusion_ablation.py`、`fusion_refresh.py`、`deep_baseline_runtime.py`、`model_comparison.py`、`stage_i_sequences.py`、`runtime_inference.py` 等超长文件；拆分目标是保持 CLI 不变、核心实现按 data/config/train/report/render 分层。
4. Artifact 收敛：对 最终指标打磨 nested public/Dingxin、流角色融合/公开融合消融 训练曲线、历史 `stage_i_private/202605*`、旧 public opt 包做引用审计；只保留 summary、metrics、figures、manifest、resume command 和必要 log，删除可重建的 row-level/nested payload。
5. Git 历史清理：只有在确认备份和远端协作影响后，再考虑 `git filter-repo` / LFS 历史瘦身；这不是只读评估阶段应直接执行的动作。

## 建议后续工作包

1. `P38` 论文协议冻结：生成一个论文级 experiment registry / result table，固定 鼎新真实数据第三方模型对比、公开融合消融、任务感知头优化、流角色融合与优化模型再评估/最终指标打磨 引用，明确每个 claim 的证据层级。
2. `P39` synthetic stress-test 可行性实现：先做小规模脚本生成器和审计，不进入主结果，只验证方法机制与泄漏边界。
3. `P40` 检索任务 定点优化：只围绕 retrieval 做 hard negative / contrastive / semantic pair 小范围 confirm，并保留失败结果。
4. `P41` 消融统一化：把 Dingxin/public 消融重新聚合到同一长表、短表和图，不混用不一致协议。
5. `P42` 仓库收敛清理：先清缓存和 docs 状态漂移，再拆超长文件，最后审计 artifact 与 git history。
6. `P43` 论文材料化：把方法公式、变量表、实验表、图、边界说明和答辩问答从当前 docs 中抽成毕业论文写作包。

## 已确认事项与执行开关

上述六个问题已由用户在 2026-07-03 明确确认：

- 内部执行不再依赖鼎新新增一手数据或专家评价数据。
- synthetic dataset 被允许且需要进入附录型实验。
- LLM 可调用，优先复用 DeepSeek v4-pro 既有链路，并可扩展用途。
- 后续仍希望继续提升 检索任务与公开路线 指标。
- 用户接受先清理仓库再跑新实验，并允许 tracked 历史 artifact 删除、外置备份和必要时 git history 改写。
- 不修改附件中的中期考核表。

当前剩余的不是原则确认，而是执行开关。用户已要求“先不要执行，先更新文档，进一步细化文档”。因此，下一步只有在用户明确要求启动时才进入清理审计或实验执行。

## 执行建议

建议先不急着重跑大实验。第一步应做 `P38 + P42` 的轻量版本：冻结论文协议矩阵、修正文档状态漂移、列出可删 artifact 的引用审计表。这样可以避免继续在膨胀仓库上叠加新 run，也能让后续模型优化和 synthetic stress test 有明确的论文位置。

用户在 2026-07-03 进一步要求：当前先不要执行清理或新实验，先更新并细化文档。因此，后续执行细化已另行落到 [thesis-prep-execution-plan-2026-07-03.md](thesis-prep-execution-plan-2026-07-03.md)。
