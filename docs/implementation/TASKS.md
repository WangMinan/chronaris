# Chronaris 当前任务

更新时间：2026-07-03

## 文档定位

本文件是当前唯一主动执行入口。总路线、阶段结构、当前任务队列和默认工作方式统一维护在这里；历史计划与阶段笔记保留在 `notes/`。

## 总路线

1. 读取指定架次的人机多源数据及元信息。
2. 建立统一 schema、统一时间参考和统一样本组织。
3. 实现双流连续潜态建模。
4. 实现物理一致性约束时间对齐。
5. 实现因果掩码跨模态融合。
6. 输出标准化融合特征与中间态接口。
7. 面向典型任务开展对比、消融和案例验证。

## 阶段结构

- `Stage A/B/C`：已完成。
- `Stage D`：后置，保留为后续数据集工程化工作。
- `Stage E0/E/F/G(min)/H`：已完成并收口，作为历史基线与后续依赖。
- `Stage I`：
  - `legacy public benchmark`：保留早期 `Phase 0/1/2/3` 公开 UAB/NASA benchmark 与 closure，当前代码入口归入 `src/chronaris/pipelines/stage_i/legacy/` 与 `scripts/stage_i/legacy/`。
  - `thesis mainline training`：覆盖 `Phase A/B/C` 的统一骨干、真实 Stage H weak-label 联合训练和 checkpoint export，当前代码入口归入 `src/chronaris/pipelines/stage_i/training/`。
  - `private proxy branch`：覆盖 `chronaris_opt`、`T1/T2/T3`、private benchmark 与组件诊断，当前代码入口归入 `src/chronaris/pipelines/stage_i/private/`，诊断报告编排归入 `evidence/`。
  - `public adapter branch`：覆盖公开 UAB/NASA adapter、calibration、transfer boundary 与 public mainline 报告，当前代码入口归入 `src/chronaris/pipelines/stage_i/public/`。
  - `evidence closure`：覆盖 P10-P18 的 evidence runner、weak-label sweep、support、rotation audit、thesis materials、midterm pack 和 runtime schema 边界说明，当前代码入口归入 `src/chronaris/pipelines/stage_i/evidence/`。
  - `runtime/service`：核心推理服务位于 `src/chronaris/serving/`，Stage I 入口脚本归入 `scripts/stage_i/runtime/`。
  - `LLM preprocessing`：P20 DeepSeek 在线时序数据预处理只作为 preprocessing context / rule review / semantic hints / runtime explanation，当前代码入口归入 `src/chronaris/pipelines/stage_i/llm/` 与 `scripts/stage_i/llm/`。

## 默认工作方式

每轮实现默认按下面顺序收敛：

1. 目标锁定。
2. 代码实现。
3. 测试闭环。
4. 文档回写。
5. 冗余清理。

运行 Python 脚本、测试、基准或阶段命令前，默认使用：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

## Stage I 代码与脚本组织

当前 Stage I 源码不再继续堆放在单层 `stage_i_*.py` 文件里：

- `src/chronaris/pipelines/stage_i/common/`：跨 public/private/evidence 复用的 observer、baseline split、deep model helper。
- `src/chronaris/pipelines/stage_i/training/`：backbone 与 multitask 训练。
- `src/chronaris/pipelines/stage_i/public/`：公开 UAB/NASA adapter、public opt、public fusion、mainline report。
- `src/chronaris/pipelines/stage_i/private/`：private benchmark、`chronaris_opt`、代理任务和优化包。
- `src/chronaris/pipelines/stage_i/evidence/`：P10-P18 主动证据、support、rotation audit、thesis materials 和 midterm pack。
- `src/chronaris/pipelines/stage_i/llm/`：P20 LLM preprocessing pipeline、harness、slicing、reporting。
- `src/chronaris/pipelines/stage_i/legacy/`：历史公开 Phase 0/1/2/3 与 baseline closure。

脚本入口统一放到 `scripts/stage_i/<category>/`。根目录不再保留 `run_stage_i_*.py` / `build_stage_i_*.py` 旧脚本文件；需要执行旧命令时，应改用 `scripts/README.md` 里列出的 canonical 路径。Python 模块层保留旧 `chronaris.pipelines.stage_i.stage_i_*` import 的包级兼容映射，以便历史 notebook 或外部调用迁移时不需要立刻重写全部 import。

## 当前工作区与推送状态

当前事实：

- 当前分支为 `main`，当前 HEAD 为 `6228787 feat: add stage i optimized final polish`，且 `origin/main` 指向同一提交；P28-GPUOPT 已完成 chronaris_public_fusion 训练效率 profiling，P30/P31/P32 已完成 private third-party comparison、public fusion ablation 和 cross-evidence matrix，P34/P35/P36 与 optimized model summary r4 confirm20 已进入当前历史，P37 optimized final polish 也已进入当前历史。当前工作树只包含毕业论文准备相关文档改动；尚未执行清理、删除 artifact、改写历史或启动新实验。本轮不改变 P27/P28/P30/P31/P32/P34/P35/P36/P37 confirmed metrics。
- `P10-P15` 主动证据工具、测试、报告、索引和可引用汇总资产已经进入远端历史，主体功能与资产提交为 `70b651a feat: add stage i evidence closure tools`。
- `Phase D/E/F` 代码、文档与资产已经进入历史基线；当前最新主动证据入口为 `docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`。
- 已进入 git 历史的最新 Stage I 真实 replay/support/ablation 资产包括：
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/`
  - `docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/`
  - `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/`
  - `docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/`
  - `docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
  - `docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
  - `docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`

中期后毕业论文准备确认前提：

- 内部执行不再期待鼎新新增一手数据或专家评价数据；答辩口径可保留“若后续可获得则作为附加验证”。
- 允许并需要引入仿真数据集；仿真数据可以进入附录型实验，但必须标注为 synthetic stress-test / simulation oracle。
- 允许调用 LLM，并优先复用现有 DeepSeek v4-pro 链路；可扩展到 synthetic scenario、expert rubric 草案、case explanation、review packet 和论文图表说明，但不替代专家真值。
- 当前时间充足，清理完成后优先继续提升 T3 retrieval 与 public route 指标。
- 用户接受先清理仓库再跑新实验；允许删除 tracked 历史 artifact、允许外置备份、允许必要时改写 git history。
- 当前用户要求先不要执行清理或新实验，先更新并细化文档。

下一轮待执行队列（尚未启动）：

1. `P38` 论文协议冻结：统一 private/public、model comparison/component ablation 的 2x2 证据矩阵、result registry 和 claim boundary。
2. `P42` 仓库收敛清理：先审计、备份、记录，再删除/拆分/历史瘦身；第一步只做 inventory，不删除 tracked artifact。
3. `P39` 仿真数据集：构建 synthetic generator、审计与附录型 stress-test，不混入真实数据主结果。
4. `P40` T3/public 指标提升：清理后定点优化 T3 retrieval 与 public route。
5. `P41` 论文级消融统一：聚合现有消融为统一长表、短表和图。
6. `P43` 论文材料化：整理方法章节、实验表、图、边界说明、复现实验包和答辩问答。

文档细化入口：

- `docs/implementation/notes/thesis-prep-readiness-assessment-2026-07-03.md`
- `docs/implementation/notes/thesis-prep-execution-plan-2026-07-03.md`

后续收敛顺序：

1. P21 已完成 LLM preprocessing 融入 Stage I 数据融合管线的对比实验，并把结果落到 `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`，可供中期报告直接引用。
2. P20 已完成 DeepSeek 在线时序数据预处理、agent-style prompt/harness v2、切片整合和小样本真实 r3-sliced run；P21 已在此基础上比较 baseline vs LLM-context、semantic query bank vs LLM hints、runtime report with/without LLM explanation，以及小样本人工复核 packet。
3. 中期报告写作优先从 `docs/midterm/` 的事实清单、边界风险说明和 claims matrix 进入；P20/P21 只能写成 LLM preprocessing context 与对比证据，不能写成真值标注、OpenAI 默认接入或核心因果证据。
4. 保留并维护 `P18` 的 `P11 stable/partial` 与 `P17 schema-contract` 当前入口，避免后续继续回退到纯手工解释或旧 r1 demo。
5. 持续维护 evidence runner 的 `skip-heavy / reuse-existing` 策略；如需更大 `live_influx` 网格，先明确预算，再从当前 `2` 组合稳定版扩展。
6. 若后续发现可用角速度字段，在 `rotation audit` 基础上复跑 `minimal / full / rigid_body`；若没有，继续保持 `rotation disabled` diagnostics 口径。
7. 若后续要把 runtime/service 继续收紧到“exact schema only”，优先围绕当前 `native_feature_schema_status=aligned` 的 missing vehicle groups 做采样契约补齐，而不是重建上游接收器。
8. 展开文献检索前，先用 `docs/midterm/claims-matrix-2026-06-13.md` 约束论文 claim 强度，再按异构时序对齐、连续潜态、物理约束、因果融合、航空人因 weak-label、LLM 辅助时序预处理六组关键词搜索。
9. 中期图表当前入口为 `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json`；旧 `r2-p18` 图包已清理，仅保留 `runtime_semantic_case.csv` 作为 P20/P21 LLM preprocessing 历史输入表，r4 runtime case refresh 已由 r5 接管并清理，r5 已由 r6 报告重绘图包接管并在 2026-06-21 深度清理中从 docs 产物目录删除。
10. Public-P27/P28 已完成 public model comparison 与 chronaris_public_fusion refresh；中期公开模型对比优先引用 `docs/artifacts/stage_i/stage-i-public-model-comparison-20260701T-stage-i-public-model-comparison-r1.md`、`docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/improvement_summary.csv` 和 P28 `fusion_refresh_summary.json`。
11. P28-GPUOPT 已完成代表性 fold GPU efficiency profiling；效率证据优先引用 `docs/artifacts/stage_i/stage-i-public-fusion-gpu-optimization-20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1.md` 与 `docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/optimization_summary.json`。该结果只说明训练吞吐、显存和 resume 状态，不替代 P28 full LOSO confirmed metrics。
12. P30 已完成 private real dual-stream Stage H third-party comparison；入口为 `docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md` 与 `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/evidence_manifest.json`。该结果是 private T1/T2/T3 proxy task 的第三方对比，不能写成论文最终人工真值任务。
13. P31 已完成 public fusion ablation；入口为 `docs/artifacts/stage_i/stage-i-public-fusion-ablation-20260702T-stage-i-public-fusion-ablation-gpuopt-r1.md` 与 `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/public_fusion_ablation_summary.json`。该结果属于 public adapter / context-proxy component ablation，不替代 P28 confirmed refresh，也不证明 public 第二模态等价于私有航电流。
14. P32 已完成 cross-evidence matrix；入口为 `docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md` 与 `docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/evidence_manifest.json`。中期报告可用它说明 private/public/proxy/component 四层证据分工。
15. P34/P35/P36 与 optimized model summary 已新增 optimized Chronaris CUDA confirm / aggregation，当前入口为 P34 r3 + P35/P36 r4：
   - P34：`docs/artifacts/stage_i/stage-i-task-aware-heads-20260702T-stage-i-task-heads-optimization-r3-confirm20.md`，资产根 `docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/`。
   - P35：`docs/artifacts/stage_i/stage-i-stream-role-aware-fusion-20260702T-stage-i-stream-role-fusion-r4-v3-confirm20.md`，资产根 `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/`。
   - P36：`docs/artifacts/stage_i/stage-i-optimized-chronaris-reevaluation-20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20.md`，资产根 `docs/artifacts/assets/stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20/`。
   - Optimized model summary：`docs/artifacts/stage_i/stage-i-optimized-model-summary-20260702T-stage-i-optimized-model-summary-r4-v3-confirm20.md`，资产根 `docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/`。
   - 当前状态：P34 `status=completed`，为 CUDA 20-epoch confirm（3 seeds / `leave_one_view_out` + `leave_one_sortie_out` / 20 epochs），T1 两个 split 小幅改善，T2 明显改善，T3 混合；P35 `status=completed`，完成 requested private/public v3 confirm，private 分支比较 `chronaris_v3_stream_role_fusion` / `v3_no_role_gate` / `v3_fixed_causal_lag`，public 分支比较 `v3_stream_role` / `v3_no_role_gate` / `v3_force_private_causal` / `v3_context_adapter_only`，public 24 个 dataset/variant/seed 组合无缺失；P36 与 optimized model summary 为 `completed` aggregation，读取固定 P30/P31/P32 reference 并汇总 P34/P35/P36 的关键指标、GPU runtime、gate 和 claim boundary。后续论文表述仍需保留 public context proxy、private proxy benchmark、T3 mixed 和“不是全面胜出”边界。
16. 2026-07-02 P34/P35/P36 后处理已完成：基于既有 CSV/JSON 重绘本轮柱状图并加入短数值标签；新增 `checkpoint_policy` future-run 输出策略；独立 CUDA profiling 记录表明当前 public context-proxy 小折 tensor cache 约 `0.115 GB`、auto batch 选 `2048/1024`、memory pressure 低于 `0.21`；清理过渡 run、dense predictions、partial CSV、checkpoint 和大日志，dense prediction CSV / checkpoint 已外置备份，重复 weak-label manifest 副本已回指 canonical。清理与 profiling 边界见 `docs/artifacts/cleanup/20260702-p34-p36-gpu-and-docs-cleanup.md`。本项不改变 P34/P35/P36 或 P30/P31/P32 confirmed metrics。
17. P37 optimized final polish 已完成，入口为 `docs/artifacts/stage_i/stage-i-optimized-final-polish-20260702T-stage-i-optimized-final-polish-r1.md`，资产根 `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/`。
   - 当前状态：`status=completed`，P30/P31/P34/P35/P36 固定为 reference，未重跑或改写；CUDA required，tensor cache `auto`，AMP `bf16`，auto batch，torch compile `default`，heartbeat/progress/resume/skip-completed 已落盘。
   - T3：`p37_t3_info_nce_temp0p05_hardw2` 与 P34 在 `top1=0.0315315`、`top3=0.0855856`、`top5=0.139640`、`mrr=0.117939` 上持平，P37 T3 rejected，后续仍引用 P34 confirmed retrieval。
   - T1：accepted；`leave_one_view_out` macro-F1 `0.204614` vs P34 `0.187489`（+`0.017125`），`leave_one_sortie_out` macro-F1 `0.220099` vs P34 `0.216065`（+`0.004034`）。
   - public route：accepted；`p37_public_force_adaptive_context_gate` NASA combined macro-F1 `0.443919` vs P35 `0.432193`（+`0.011726`），UAB mean RMSE `3.191811` vs P35 `3.374779`（改善 `0.182968`）。该分支仍是 public adapter / context-proxy evidence。
18. 2026-07-01 src/docs artifact prune 已删除 `src/tests/scripts` 下本地 Python 编译缓存和 archive-only 早期 public torch/mainline/fusion screen 旧产物；当前删除边界见 `docs/artifacts/cleanup/20260701-src-docs-artifact-prune.md`。

引用边界：P37 final polish 只在固定 P30/P31/P34/P35/P36 reference 上做局部收束；T1 calibration 与 public route calibration 可写成 accepted improvement，T3 retrieval 仍沿用 P34 confirmed retrieval，public 分支继续写成 public adapter / context-proxy evidence。

验收：

- `git diff --check` 通过。
- 状态文档中的 `HEAD`、远端同步状态、当前任务队列和编码缺口一致。
- 新增报告路径和 asset 路径必须能从 `ARTIFACTS.md` 或本文件追溯。
- 清理后 `git lfs status` 需保持正常，且若 `git lfs migrate info --include-ref=refs/heads/main --include='docs/**'` 未显示异常历史膨胀，则不做历史重写。

## 已完成 P0：冻结 Phase D/E/F 工作区并提交

结果：`Phase D/E/F` 主代码、测试与 runtime sample exporter 已经进入本地 git 历史。

本轮复查范围：

- 复查 `git status --short --untracked-files=all` 中全部未提交项，区分三类改动：
  - `Phase D/E/F` 代码、测试和文档同步。
  - 为 runtime checkpoint feature schema 做的必要兼容改动。
  - 与本阶段无关的临时改动或新产物。
- 保留并复查 `Phase D rigid_body` 关键文件：
  - `src/chronaris/models/alignment/physics_state_mapping.py`
  - `src/chronaris/models/alignment/physics_residuals.py`
  - `src/chronaris/models/alignment/physics.py`
  - `src/chronaris/models/alignment/physics_features.py`
  - `src/chronaris/models/alignment/__init__.py`
  - `src/chronaris/pipelines/alignment_preview.py`
  - `scripts/run_stage_e_relative_preview.py`
  - `tests/test_alignment_model_losses.py`
- 保留并复查 `Phase E semantic event fusion` 关键文件：
  - `src/chronaris/models/fusion/semantic_event.py`
  - `src/chronaris/models/fusion/__init__.py`
  - `src/chronaris/pipelines/causal_fusion.py`
  - `src/chronaris/pipelines/stage_i/evidence/support_builders.py`
  - `src/chronaris/pipelines/stage_i/evidence/support_reporting.py`
  - `tests/test_stage_i_support.py`
- 保留并复查 `Phase F runtime inference` 关键文件：
  - `src/chronaris/dataset/streaming_windows.py`
  - `src/chronaris/dataset/__init__.py`
  - `src/chronaris/serving/runtime_inference.py`
  - `src/chronaris/serving/__init__.py`
  - `scripts/stage_i/runtime/run_inference.py`
  - `src/chronaris/pipelines/stage_i/training/multitask_train.py`
  - `tests/test_runtime_inference.py`
- 确认新增代码默认输出路径统一落到 `docs/artifacts/assets/...`，Markdown 报告统一落到 `docs/artifacts/stage_i/...` 或既有阶段报告目录。
- 本轮补跑并保留结果，完整覆盖 torch runtime 用例时需要显式开启测试开关：

```bash
CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest \
  tests.test_alignment_model_losses \
  tests.test_alignment_pipeline \
  tests.test_stage_i_support \
  tests.test_stage_i_multitask_train \
  tests.test_runtime_inference
```

当前最近一次结果：`Ran 33 tests in 7.962s`，`OK`。

本轮提交：

- `0f4db72 feat: add stage i runtime sample exporter`
- `9ef4f64 feat: add rigid-body physics semantic event runtime inference`
- `890a315 docs: record stage i runtime semantic rigid-body artifacts`
- `57ca739 feat: expand stage i rigid-body support and runtime service`

## 已完成 P1：补真实资产上的 Stage I multitask 联合训练证据

结果：Phase C 已经不再停留在合成烟测，而是在真实 Stage H all-window clean 资产上形成了可引用证据。

前置编码任务：

- 已新增 `scripts/stage_i/training/train_multitask.py`。
- 已复用 `collect_stage_i_backbone_samples()` 路线并补 `view_id::raw_window_sample_id` sample-id contract，生成真实 `E0ExperimentSample`。
- 已复用 `load_aligned_private_records()` 与 `build_stage_i_real_task_payload()` 构造 weak-label task entries。
- 已修正 workload proxy 的归一化尺度，并把 event replay pair 调整为优先近邻配对，避免验证/测试分区丢失 retrieval 监督。
- 默认输出已切到 `docs/artifacts/assets/stage_i_multitask/<run_id>/`。

建议真实资产入口：

- `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
- `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`

本轮输出：

- Stage I multitask checkpoint binary 已从 git 保留范围移出；当前仅在远程开发机仓库外备份或由训练重新生成，repo 内引用 `multitask_summary.json` / report / schema contract。
- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- `docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`

报告边界：

- 已明确写成 `thesis weak-label evidence`。
- 已保持“不写成人工真值任务”的边界。
- 当前产物按 `mainline closure evidence` 使用，不宣称人工真值最优。

## 已完成 P2：刷新 private benchmark 分层资产

结果：已用当前 Phase C 代码重跑 private benchmark，使历史 `T1/T2/T3` proxy evidence 与 thesis weak-label evidence 在资产层明确拆开。

建议命令形态：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/stage_i/private/run_benchmark.py \
  --e-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json \
  --f-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json \
  --output-root docs/artifacts/assets/stage_i_private \
  --report-root docs/artifacts \
  --enable-optimized-chronaris \
  --export-optimized-package
```

本轮验收：

- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_manifest.jsonl`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_summary.json`。
- 已生成 `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/thesis_task_summary.json`。
- `private_benchmark_summary.json` 已包含 `evidence_layers.proxy_evidence` 和 `evidence_layers.thesis_task_evidence`。
- CLI 已打印 thesis task manifest / summary 路径。

## 已完成编码 P3：补 Stage F 刚体运动物理约束

结果：`rigid_body` physics family 已进入工作区，并保持旧 `minimal / full` 路线不回归。

实际落点：

- `src/chronaris/models/alignment/physics_state_mapping.py`
- `src/chronaris/models/alignment/physics_residuals.py`
- `src/chronaris/models/alignment/physics.py`
- `src/chronaris/models/alignment/physics_features.py`
- `src/chronaris/pipelines/alignment_preview.py`
- `scripts/run_stage_e_relative_preview.py`
- `tests/test_alignment_model_losses.py`

本轮验收：

- `rigid_body` family 可被显式选择。
- 已新增刚体状态映射与残差模块，能诊断启用/缺失项。
- `tests.test_alignment_model_losses` 已覆盖 family 选择、缺失诊断和垂直/姿态残差。

## 已完成编码 P4：补 Stage G 语义事件融合

结果：`G(min)` 之上已新增 `SemanticQueryBank / EventTokenExtractor / CausalEventFusion`，并把事件级归因摘要接到 support 报告链路。

实际落点：

- `src/chronaris/models/fusion/semantic_event.py`
- `src/chronaris/models/fusion/__init__.py`
- `src/chronaris/pipelines/causal_fusion.py`
- `src/chronaris/pipelines/stage_i/evidence/support_builders.py`
- `src/chronaris/pipelines/stage_i/evidence/support_reporting.py`
- `tests/test_stage_i_support.py`

本轮验收：

- 已输出 event token。
- 已输出 query-to-event attention。
- 已输出事件级归因摘要。
- `tests.test_stage_i_support` 已验证 support 报告能区分“时间步注意力”和“事件级归因”。

## 已完成编码 P5：补 runtime inference

结果：已新增真正的 checkpoint-backed runtime inference 入口，支持样本序列化、本地 replay 输入和流式窗口缓存。

实际落点：

- `src/chronaris/dataset/streaming_windows.py`
- `src/chronaris/dataset/__init__.py`
- `src/chronaris/serving/runtime_inference.py`
- `src/chronaris/serving/__init__.py`
- `scripts/stage_i/runtime/run_inference.py`
- `src/chronaris/pipelines/stage_i/training/multitask_train.py`
- `tests/test_runtime_inference.py`

本轮验收：

- mock stream 或本地回放流可以增量产出窗口。
- 可以加载 checkpoint 做风险/负荷/事件预测。
- 可以输出 attention / semantic event attribution 解释。
- `tests.test_runtime_inference` 已覆盖窗口缓存、样本序列化和端到端推理闭环。

## 已完成主体 P6：把 Phase D/E/F 从代码完成推进到产物闭环

结果：runtime replay、语义事件融合 support、刚体约束 smoke / ablation 都已经落盘；并且在切到本地 `127.0.0.1:3306` MySQL 后，`rigid_body` 已经不再只是 fallback evidence，`vehicle_rigid_body_translation` 已经在真实链路上启用。

本轮完成：

- 已补 runtime sample exporter：
  - `scripts/stage_i/runtime/export_runtime_samples.py`
- 已完成 runtime replay：
  - 首轮 r1 产物已由 service r2 接管并清理，仅保留在 git 历史中。
  - 当前入口：`docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_summary.json`
  - 当前报告：`docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
  - raw `runtime_samples.jsonl` 已按 `docs/artifacts/cleanup/20260619-lfs-docs-prune.md` 从 docs/LFS 清理，需要复跑时重新生成。
  - 当前 replay 规模：`111` 个样本、`3` 个 view、`2` 个 sortie，`sample_id_mode=view_prefixed`。
  - 当前 runtime task heads：`risk_proxy` 分类、`workload_proxy` 回归、`event_replay_tag` 检索。
- 已完成语义事件融合 preview + support：
  - `docs/artifacts/stage_i/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1.md`
  - `docs/artifacts/stage_i/assets/stage-i-semantic-event-20260607T-stage-i-semantic-event-r1/causal_fusion_summary.json`
  - 首轮 support r1 产物已由 r2 接管并清理，仅保留在 git 历史中。
  - 当前 support summary：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
  - 当前 support 报告：`docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
  - `support_summary.json` 已包含 `alignment_support`、`causal_support.semantic_event`、`main_ablation_rows` 和 overview plot。
- 已完成 `minimal / full / rigid_body` 真实 smoke / ablation：
  - 首轮 rigid-body r1 产物已由 r2 接管并清理，仅保留在 git 历史中。
  - 当前 rigid-body summary：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json`
  - 当前 rigid-body 报告：`docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`
  - 本次更新后 `vehicle_field_metadata.status=loaded`、`field_count=96`。
  - `rigid_body` 的 `vehicle_rigid_body_translation=1.413177490234375`，`vehicle_rigid_body_vertical=0`，`vehicle_rigid_body_rotation=0`。

## 中期前主动推进顺序

1. 先建立统一 evidence runner，避免后续五条线靠手工命令散跑。
2. 深挖现有私有 Stage H 数据与论文本体模型：围绕 `risk_proxy / workload_proxy / event_replay_tag` 做小网格和消融表。
3. 拆解 `chronaris_opt` 机制贡献：把它从“private proxy 最优结果”推进成“表示学习、对齐、因果掩码贡献可诊断”。
4. 有界补做 CPU-heavy `sklearn` / UAB torch 候选：只作为 public adapter baseline / calibration baseline，不扩写成本体闭环。
5. 整理 NASA/UAB 公开适配器迁移边界：用表格说明公开代理数据与私有双流数据的模态、标签、任务粒度和时间基准差异。
6. 核验 `rigid_body rotation` 字段：能找到成对角速度字段就补 rotation ablation；找不到则固化为真实字段缺口诊断。

## 已完成 P7：扩充 rigid_body 的字段语义覆盖

结果：`rigid_body` 已经不再只启用 `translation`，本轮通过真实 MySQL label + token 扩充，`vertical` 也已经在真实链路上启用。

本轮完成：

- `vehicle_field_metadata` 已确认可通过本地 `127.0.0.1:3306` 正常加载。
- `BUS6000019110020` 当前加载字段数为 `96`。
- 已扩充中文 token，并新增刚体字段映射诊断导出。
- 新的 `rigid_body` 真实 run：

```bash
CHRONARIS_MYSQL_HOST=127.0.0.1 CHRONARIS_MYSQL_PORT=3306 \
CHRONARIS_MYSQL_USER=wangminan CHRONARIS_MYSQL_PASSWORD=... \
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_e_relative_preview.py \
  --enable-physics-constraints \
  --physics-constraint-family rigid_body \
  --input-normalization-mode zscore_train \
  --epoch-count 1 \
  --batch-size 8 \
  --strict-mysql-field-labels \
  --device cpu \
  --report-path docs/artifacts/stage_i/stage-i-rigid-body-rigid-body-20260607T-stage-i-rigid-body-r2.md
```

本轮验收：

- `vehicle_field_metadata.status=loaded`
- `enabled_residuals=['translation','vertical']`
- `vehicle_rigid_body_translation=1.133332371711731`
- `vehicle_rigid_body_vertical=3.9466116428375244`
- 汇总：`docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json`
- 报告：`docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`

## 已完成 P8：放大 semantic event support 证据

结果：已经从单 preview summary 扩到当前 Stage H `validation` profile 的 3 个双流 view，并生成了 view-level semantic ranking。

本轮完成：

- 新增 runner：`scripts/stage_i/evidence/build_semantic_event_support.py`
- 新增多 view summary：
  - `docs/artifacts/assets/stage_i_semantic_event_support/20260607T-stage-i-semantic-support-r2/semantic_event_support_summary.json`
- 新增 support 聚合：
  - `docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- 新增报告：
  - `docs/artifacts/stage_i/stage-i-semantic-event-support-20260607T-stage-i-semantic-support-r2.md`
  - `docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`

本轮验收：

- 覆盖 `3` 个 view、`111` 个样本。
- `support_summary.json` 已包含 `causal_support.semantic_event.view_rows`。
- support 报告已经能回答：
  - top view：`20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
  - dominant query：`risk_proxy`
  - top offset：`0.626197s`

## 已完成 P9：runtime inference 服务化补强

结果：runtime inference 现在支持 batch / incremental / both 三种 replay 模式、schema 对齐诊断和 JSONL 导出。

本轮完成：

- `StreamingWindowBuffer` 已支持：
  - `max_cached_points_per_stream`
  - `allow_out_of_order`
  - `diagnostics`
- `runtime_inference.py` 已支持：
  - `replay_mode=batch|incremental|both`
  - `batch_size`
  - `max_windows`
  - `strict_feature_schema`
  - `input_normalization_stats` 作为旧 checkpoint 的 schema fallback
  - latency / throughput / chunk_count / feature_schema_status diagnostics
- CLI 已支持：
  - `--max-windows`
  - `--batch-size`
  - `--emit-jsonl`
  - `--strict-feature-schema`
  - `--replay-mode`
- 新产物：
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_summary.json`
  - `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_predictions.jsonl`
  - `docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`

本轮验收：

- `replay_mode=both`
- `batch_sample_count=40`
- `incremental_sample_count=40`
- `sample_count_match=True`
- `feature_schema_status=aligned`
- `feature_schema_source=input_normalization_stats`

## 已完成 P10：统一论文闭环评测 harness

目标：把 Phase C/D/E/F 以及中期前新增的五条证据线整合成一个可重复跑的 evidence runner，减少后续论文补图、补表时的手工步骤。

建议新增：

- `src/chronaris/pipelines/stage_i/evidence/closure_runner.py`
- `scripts/stage_i/evidence/run_closure.py`
- `tests/test_stage_i_evidence_runner.py`

职责：

- 串联 Stage I multitask checkpoint、rigid_body ablation、semantic support、runtime replay、private proxy diagnostics、public adapter calibration summary。
- 统一 run id 规则、artifact root、report root 和 manifest 输出。
- 生成 `evidence_manifest.json`，记录输入、输出、命令参数、git commit、测试结果摘要和 `evidence_layer`。
- 支持 `--skip-heavy`、`--reuse-existing`、`--only multitask|rigid_body|semantic|runtime|private_proxy|public_adapter|rotation|all`。
- 对 heavy public adapter 分支默认只登记已有产物；只有显式打开 `--run-heavy-public-adapter` 才真实执行。

本轮结果：

- 统一入口：
  - `src/chronaris/pipelines/stage_i/evidence/closure_runner.py`
  - `scripts/stage_i/evidence/run_closure.py`
  - `tests/test_stage_i_evidence_runner.py`
- 稳定 manifest：
  - `docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/evidence_manifest.json`
- 稳定报告：
  - `docs/artifacts/stage_i/stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md`
- 当前 `r2` 已纳入：
  - `multitask / rigid_body / semantic / runtime / private_proxy / public_adapter / rotation`
- `skip-heavy` 已切到 bounded 路线：
  - `multitask` 使用 `stage_h_window_stats_proxy`
  - `rigid_body / semantic / runtime` 复用稳定资产

## 已完成 P11：私有 Stage H 论文本体模型深挖

目标：围绕现有私有 Stage H 双流数据和 Phase C multitask checkpoint，把 `risk_proxy / workload_proxy / event_replay_tag` 做成更充分的 thesis weak-label evidence，而不是只保留一轮训练结果。

建议新增或扩展：

- `src/chronaris/pipelines/stage_i/evidence/weak_label_sweep.py`
- `scripts/stage_i/evidence/run_weak_label_sweep.py`
- `tests/test_stage_i_multitask_sweep.py`

实验设计：

- 小网格，不做无边界扩搜：
  - `physics_constraint_family=minimal|full|rigid_body`
  - `causal_weight=0|0.05|0.1`
  - `task_loss_weight=0.5|1.0`
  - `causal_lag_window_points=None|3`
- 固定输入：
  - `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
  - `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
- 每个 run 输出 `multitask_summary.json`、`thesis_task_manifest.jsonl`、`checkpoint_metadata`、训练/验证/测试指标。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/evidence/weak_label_sweep.py`
  - `scripts/stage_i/evidence/run_weak_label_sweep.py`
  - `tests/test_stage_i_multitask_sweep.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/multitask_sweep_summary.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/thesis_weak_label_multitask_ablation.csv`
  - `docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md`
- 当前 `skip-heavy` 路线说明：
  - `sample_source=stage_h_window_stats_proxy`
  - 仍明确写成 `thesis weak-label evidence`
  - 不包装成人工真值任务

## 已完成 P11+：补一轮 `live_influx` thesis weak-label sweep

目标：在当前 bounded `stage_h_window_stats_proxy` sweep 之外，使用本地 `127.0.0.1` 的 MySQL / InfluxDB CLI 形成 `live_influx` sample collection 证据，并与 proxy 路线并排展示。

本轮结果：

- 真实 `live_influx` child run 已完成并用于稳定汇总：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-01-minimal-cw0p00-tlw0p50-lagnone/multitask_summary.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/runs/20260613T-stage-i-p11-live-influx-r1-02-minimal-cw0p00-tlw0p50-lag3/multitask_summary.json`
- 稳定汇总产物：
  - r2 汇总已由 r3 resume 接管并清理，仅保留在 git 历史中。
  - 当前 summary：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
  - 当前表：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/thesis_weak_label_multitask_ablation.csv`
  - 当前报告：`docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md`
- 当前对比口径：
  - `sample_source=live_influx`
  - `sample_count=111`
  - `task_entry_count=333`
  - `combination_count=2`
  - `best_test_total=1153.8985701851223`
- 当前 blocker 已保留但未伪造成稳定证据：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/progress.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/run.log`
  - `20260613T-stage-i-p11-live-influx-r1` 的 `4` 组合尝试在 `run_index=3/4` 处因 runtime cost 手动中断；稳定 `r2` 只复用其中已完成的两个 live child run，与 proxy 的 `2` 组合口径对齐。

## 已完成 P12：`chronaris_opt` 机制诊断

代码落点：

- `src/chronaris/pipelines/stage_i/private/optimization.py`
- `src/chronaris/pipelines/stage_i/private/benchmark.py`
- `scripts/stage_i/private/run_benchmark.py`
- `tests/test_stage_i_private_optimization.py`

目标：

- 把 `chronaris_opt` 从“private proxy benchmark 最优候选”推进成“机制贡献可诊断”的证据。
- 在 `T1/T2/T3` 上拆解：
  - 去掉因果掩码。
  - 去掉时间残差或 lag-aware residual。
  - 去掉 task-aware head。
  - 仅保留 E/F/G/H 既有 baseline。
- 输出 `chronaris_opt_component_ablation.csv/json` 和中文机制诊断报告。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/evidence/private_component_ablation.py`
  - `scripts/stage_i/evidence/run_private_component_ablation.py`
  - `tests/test_stage_i_private_component_ablation.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.json`
  - `docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.csv`
  - `docs/artifacts/stage_i/stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md`
- 当前报告已拆出：
  - `remove_causal_mask`
  - `remove_time_residual`
  - `remove_task_aware_head`

## 已完成 P13：public adapter 有界校准 baseline

目标：中期前补做有限预算的 CPU-heavy `sklearn` / UAB torch 候选，用于公开代理数据的 adapter baseline / calibration baseline，而不是改写成双流本体闭环。

建议范围：

- UAB：
  - 默认仍以 torch heat-specialist / robust-prior 公开线为主。
  - 只允许 3 到 5 个候选组合，固定 seed、固定 LOSO、固定 `selected_subset`。
  - CPU-heavy `sklearn uab_hybrid` 必须显式 `--allow-cpu-heavy-sklearn`，并在 manifest 写入 `heavy_reason` 与 runtime。
- NASA：
  - 继续以 `NASA enhanced round 1` 为主。
  - 只补必要的 calibration 对照，不扩成新的深度模型竞赛。

建议落点：

- 扩展 `scripts/stage_i/public/run_opt.py` 的 run manifest metadata。
- 新增 `src/chronaris/pipelines/stage_i/evidence/public_adapter_calibration.py` 或复用 public mainline report builder。
- 新增 `tests/test_stage_i_public_opt.py` 中的 evidence layer / heavy guard 回归。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/evidence/public_adapter_calibration.py`
  - `scripts/stage_i/evidence/run_public_adapter_calibration.py`
  - `tests/test_stage_i_public_transfer_boundary.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/public_adapter_calibration_summary.json`
  - `docs/artifacts/stage_i/stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md`
- 当前 summary 已区分：
  - `public_adapter_baseline`
  - `calibration_baseline`
  - `legacy_public_opt`
  - `torch_uab`

## 已完成 P14：NASA/UAB 迁移与校准边界报告

目标：把 NASA/UAB 公开适配器结果整理成论文可引用的“迁移边界”证据，说明公开代理数据和私有真实双流数据之间的差异。

建议新增：

- `src/chronaris/pipelines/stage_i/evidence/public_transfer_boundary.py`
- `scripts/stage_i/evidence/build_public_transfer_boundary.py`
- `tests/test_stage_i_public_transfer_boundary.py`

报告内容：

- 数据边界表：
  - 私有 Stage H：真实生理流 + 真实飞机时序流 + sortie/pilot/view。
  - UAB：公开 physiology + task/context proxy + subjective workload。
  - NASA：公开 physiology + scenario/context proxy + attention state。
- 任务边界表：
  - `risk_proxy / workload_proxy / event_replay_tag` 属于 thesis weak-label。
  - `T1/T2/T3` 属于 private proxy。
  - UAB/NASA 属于 public adapter / calibration evidence。
- 性能表：
  - 引用 `stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`。
  - 引用新的 P13 calibration summary。

本轮结果：

- 新增：
  - `src/chronaris/pipelines/stage_i/evidence/public_transfer_boundary.py`
  - `scripts/stage_i/evidence/build_public_transfer_boundary.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/public_transfer_boundary_summary.json`
  - `docs/artifacts/stage_i/stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md`
- 当前报告已覆盖：
  - 数据边界表
  - 任务边界表
  - public adapter/calibration 性能引用表

## 已完成 P15：`rigid_body rotation` 字段核验与消融

目标：中期前补完 `rigid_body` 的最后一个关键缺口：`rotation`。能找到真实成对角速度字段就补实验；找不到就把缺口固化成可引用诊断。

建议落点：

- `src/chronaris/models/alignment/physics_state_mapping.py`
- `scripts/run_stage_e_relative_preview.py`
- `tests/test_alignment_model_losses.py`

核验顺序：

1. 从 MySQL label 和 Stage H feature schema 中复查 `真航向`、俯仰、横滚、角速度、航向角速度等字段。
2. 若存在可用 rate field，扩充 token 并重跑 `minimal / full / rigid_body`。
3. 若只存在角度、没有 rate field，保留 `rotation` disabled，并输出 `missing_requirements.rotation` 诊断。

本轮结果：

- 代码与测试：
  - `src/chronaris/models/alignment/physics_state_mapping.py`
  - `src/chronaris/pipelines/stage_i/evidence/rigid_body_rotation_audit.py`
  - `scripts/stage_i/evidence/run_rigid_body_rotation_audit.py`
  - `tests/test_alignment_model_losses.py`
  - `tests/test_stage_i_rotation_audit.py`
- 稳定产物：
  - `docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-rotation-audit-r2/rigid_body_rotation_audit_summary.json`
  - `docs/artifacts/stage_i/stage-i-rigid-body-rotation-audit-20260607T-stage-i-rotation-audit-r2.md`
- 当前结论：
  - `BUS6000019110020.code1031 = 真航向` 已归入 `yaw`
  - `yaw_rate` 仍缺失
  - `rotation_status=disabled`

## 已完成 P16：论文案例、消融表与说明图稳定化

目标：把现有和 P11-P15 新增 evidence 转成论文可直接引用的表格、案例材料和说明图件，减少后期靠手工复制指标或临时画图。

代码落点：

- `src/chronaris/pipelines/stage_i/evidence/thesis_materials.py`
- `scripts/stage_i/evidence/build_thesis_materials.py`
- `docs/artifacts/stage_i/`
  - 输出论文案例报告，保留中文解释、边界说明、图表引用路径。
- `docs/artifacts/assets/stage_i_thesis_figures/<run_id>/`
  - 输出 `.png` 图件和对应 `figure_manifest.json`，记录每张图的源数据、用途、证据层级和可复现命令。

本轮结果：

- 首轮图包已被 P22/r3、P23/r4、P25/r5 和 P26/r6 逐步替代，并已从 docs 产物目录清理。
- 当前中期图表入口只使用：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/figure_manifest.json`
  - `docs/artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md`
- P20/P21 LLM preprocessing 仍复用的 runtime case 输入表保留在：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`

## 已完成 P17：系统封装与部署边界

目标：为毕业设计系统实现章节补齐“离线/准实时推理服务”的工程闭环和说明图，而不是只停留在训练脚本和报告。

代码落点：

- `src/chronaris/serving/`
  - `src/chronaris/serving/runtime_service_smoke.py`
  - `src/chronaris/serving/__init__.py`
- `scripts/stage_i/runtime/run_smoke.py`
- `tests/test_runtime_service_smoke.py`

本轮结果：

- 真实单 view 输入：
  - raw `input_view_runtime_samples.jsonl` 已按 `docs/artifacts/cleanup/20260619-lfs-docs-prune.md` 从 docs/LFS 清理，需要复跑时重新生成。
  - 当前 `view_id=20251005_四01_ACT-4_云_J20_22#01__pilot_10033`
  - 当前 `sample_count=37`
- 稳定服务 smoke root：
  - 首轮 r1 已由 r2-contract 接管并清理，仅保留在 git 历史中。
  - 当前入口：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/`
- 关键输出：
  - `runtime_service_smoke_summary.json`
  - r1 runtime inference payload 已清理；当前 runtime summary 保留在 r2-contract 入口。
  - `runtime_error_cases.json`
  - `figure_manifest.json`
  - `docs/artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md`
- 三张系统说明图：
  - `runtime_service_flow.png`
  - `runtime_payload_schema.png`
  - `runtime_error_cases.png`
- 当前错误样例均已固化为 `expected_failure`：
  - `missing_checkpoint`
  - `missing_fields`
  - `empty_window`
  - `schema_mismatch`
- 当前真实 smoke 结论：
  - checkpoint 冷启动成功
  - 单 view replay JSONL 成功输出 predictions JSONL 与 summary JSON
  - `feature_schema_status=aligned`
  - 说明当前 runtime facade 仍依赖 `input_normalization_stats` 做 schema 对齐，若后续要收紧到 exact schema，需要继续补输入契约而不是重建上游接收器

## 已完成 P18：P11/P17 风险收口优化

目标：把 P11 的“部分完成但不中断证据链”和 P17 的“aligned 但还不是 exact schema”变成可复现、可解释、可验收的工程能力，而不是靠人工口头说明。

### 已完成 P18-A：P11 live_influx sweep partial/resume

当前事实：

- `r1` 的更大 `4` 组合尝试没有伪造成成功结果，blocker 已保留在 `progress.json / run.log`。
- 当前已新增 stable resume 版与 partial blocked 版：
  - stable resume：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/`
  - partial blocked：`docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/`

代码落点：

- `src/chronaris/pipelines/stage_i/evidence/weak_label_sweep.py`
  - 每个 child run 完成后即时写入 `partial_summary.json` 和临时 CSV。
  - 中断或失败时输出 `status=partial_blocked`、`completed_child_runs`、`blocked_at_run_index`、`blocker_log_path`。
  - 支持从已有 child run 恢复汇总，避免重复跑已完成组合。
- `scripts/stage_i/evidence/run_weak_label_sweep.py`
  - 增加 `--resume-existing` 与 `--resume-run-root`，允许从指定历史 run root 复用已完成 child run。
  - 增加 `--max-runtime-seconds` 或明确的预算 guard，避免 live_influx 大网格无限拖住。
- `tests/test_stage_i_multitask_sweep.py`
  - 覆盖 partial summary、resume、blocker 不伪造成 completed。

本轮结果：

- stable resume summary：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
  - `docs/artifacts/stage_i/stage-i-thesis-weak-label-multitask-sweep-20260613T-stage-i-p11-live-influx-r3-resume.md`
  - 当前已包含：
    - `derived_from_run_id=20260613T-stage-i-p11-live-influx-r1`
    - `completed_child_run_paths`
    - `blocked_attempt_log_paths`
    - `blocked_at_run_index=3`
    - `evidence_layer=thesis_weak_label`
- partial blocked summary：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/partial_summary.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/thesis_weak_label_multitask_ablation.partial.csv`
  - 当前 `status=partial_blocked`
  - 当前 `completed_child_runs=2`
  - 当前 `blocked_at_run_index=3`
  - 当前 `blocker_log_path=docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r4-partial/run.log`
- blocker 继续保留：
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/progress.json`
  - `docs/artifacts/assets/stage_i_multitask_sweep/20260613T-stage-i-p11-live-influx-r1/run.log`
- 当前没有伪造更大 live_influx 网格成功；`r4-partial` 只是一个 bounded resume/blocker smoke，用来固化证据链，而不是把未完成组合包装成 completed。

### 已完成 P18-B：P17 runtime schema contract 与 exact 边界

当前事实：

- P17 真实 smoke 成功，但 `feature_schema_status=aligned`，`feature_schema_source=input_normalization_stats`。
- 当前 checkpoint 没有显式 `feature_schema`，runtime fallback 到 `input_normalization_stats`。
- 当前单 view 输入为 `965` 个 vehicle features；checkpoint 期望 `1930` 个 vehicle features。缺口集中在另一半 BUS measurement：`BUS6000019110021` 到 `BUS6000019110026`。

代码落点：

- 新增 `src/chronaris/serving/runtime_schema_contract.py`
  - 从 checkpoint 导出 expected schema、schema hash、stream feature counts、measurement group counts。
  - 对比 runtime JSONL 输入 schema，输出 missing/extra feature 分组和 `exact_possible`。
  - 输出 `runtime_schema_contract.json`，作为 P17 后续部署契约事实源。
- 扩展 `src/chronaris/serving/runtime_service_smoke.py`
  - `StageIRuntimeSmokeConfig` 增加 `strict_feature_schema: bool`。
  - summary 增加 `schema_contract_path`、`native_feature_schema_status`、`canonical_feature_schema_status`。
  - error cases 继续保留 `schema_mismatch`，但错误摘要优先输出分组统计，避免几百个字段刷屏。
- 扩展 `scripts/stage_i/runtime/run_smoke.py`
  - 增加 `--strict-feature-schema`。
  - 增加 `--export-canonical-payload` 或等价参数，验证 canonical payload 路线；raw JSONL 已在 LFS 清理中移出 docs。
- 测试：
  - `tests/test_runtime_service_smoke.py`
  - 新增 `tests/test_runtime_schema_contract.py`

本轮结果：

- 新增 schema contract root：
  - `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/`
- 关键产物：
  - `runtime_service_smoke_summary.json`
  - `runtime_schema_contract.json`
  - `runtime_inference/20260613T-stage-i-runtime-service-smoke-r2-contract-canonical/runtime_inference_summary.json`
  - `docs/artifacts/stage_i/stage-i-runtime-service-smoke-20260613T-stage-i-runtime-service-smoke-r2-contract.md`
  - raw `canonical_runtime_samples.jsonl` 已按 `docs/artifacts/cleanup/20260619-lfs-docs-prune.md` 清理，contract 与 canonical runtime summary 保留。
- 当前 native 单 view 输入已清楚记录：
  - `native_feature_schema_status=aligned`
  - `expected_vehicle_feature_count=1930`
  - `input_vehicle_feature_count=965`
  - `missing_vehicle_feature_count=965`
  - `missing_vehicle_measurement_group_counts` 覆盖：
    - `BUS6000019110021`
    - `BUS6000019110022`
    - `BUS6000019110023`
    - `BUS6000019110024`
    - `BUS6000019110025`
    - `BUS6000019110026`
- strict native smoke 已作为 expected failure 写入：
  - `runtime_error_cases.json`
  - `strict_native_feature_schema_probe`
- canonical payload 路线已生成：
  - `runtime_schema_contract.json`
  - `canonical_feature_schema_status=exact`
  - raw canonical JSONL 已从 docs/LFS 清理；需要复跑时由 `run_smoke.py` 重新导出。
- 当前报告口径已固定：
  - `native aligned` 是当前真实部署边界
  - `canonical exact` 是服务层契约化 payload 能力
  - 不把 canonical exact 写成“原始上游输入 exact”
  - 不重建上游接收器，不做原始大文件入仓

### 已完成 P18-C：P16 图表同步刷新

目标：P18 完成后，刷新 P16 thesis materials，让论文图表同步反映 P11/P17 的真实边界。

本轮结果：

- 刷新后的 thesis materials 曾被 P25/r5 图包接管，当前入口已由 P26/r6 接管：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/`
  - `docs/artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md`
- 旧 r2-p18 图、manifest 和报告已清理；仅保留 LLM preprocessing 历史输入表：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`
- 当前 `weak_label_sweep_ablation.png/csv` 已增加：
  - `summary_status`
  - `derived_from_run_id`
  - `blocked_at_run_index`
  - `blocked_attempt_log_path_count`
  - partial resume/blocker 注记
- 当前 `runtime_semantic_case.png/csv` 已由 r6 重新纳入 thesis materials 生成链路，并体现：
  - `native_feature_schema_status=aligned`
  - `canonical_feature_schema_status=exact`
  - `expected_vehicle_feature_count=1930`
  - `input_vehicle_feature_count=965`
  - `missing_vehicle_feature_count=965`
  - `native_missing_measurement_group_count=6`
- 当前 `figure_manifest.json` 已记录：
  - `runtime_schema_contract.json`
  - `stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json`
  - `stage-i-p11-live-influx-r4-partial/partial_summary.json`
  - r5 `runtime_semantic_case` 图表替代说明与 `figure_quality_audit.csv`

## 已完成 P19：中期报告材料冻结与 docs 入口清理

目标：在搜索论文和正式展开中期报告前，把当前可写事实、证据边界、风险说明和报告 claim 强度冻结成文档，供后续本地 clone 后直接作为中期报告写作基础。

本轮结果：

- 新增中期写作入口：
  - `docs/midterm/README.md`
  - `docs/midterm/midterm-fact-sheet-2026-06-13.md`
  - `docs/midterm/boundaries-and-risks-2026-06-13.md`
  - `docs/midterm/claims-matrix-2026-06-13.md`
- 清理过时入口：
  - `docs/artifacts/mid-term/README.md` 已从旧 `20260509` r2 中期包改指向当前 `20260607` r3 和 `docs/midterm/`。
  - `docs/artifacts/mid-term/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md` 兼容链接已删除。
  - `docs/artifacts/mid-term/stage-i-midterm-20260607T-stage-i-midterm-r3.md` 兼容链接已新增。
- 更新导航与状态：
  - `docs/README.md` 已新增 `midterm` 目录说明。
  - `docs/artifacts/ARTIFACTS.md` 已新增中期事实清单、边界说明和 claims matrix。
  - `docs/artifacts/stage_i/README.md` 已把 P16/P17 当前入口改为 `r2-p18 / r2-contract`，首轮 r1 降为历史入口。
  - `docs/STATE.md` 已把 P16/P17 r1 标为首轮历史，并补入 P18/p18 和中期写作材料入口。

验收口径：

- 中期报告写作先读 `docs/midterm/README.md`。
- 当前事实引用先读 `docs/midterm/midterm-fact-sheet-2026-06-13.md`。
- 边界和答辩风险先读 `docs/midterm/boundaries-and-risks-2026-06-13.md`。
- 文献检索和正文 claim 先用 `docs/midterm/claims-matrix-2026-06-13.md` 约束证据强度。

## 已完成 P20：DeepSeek 在线时序数据预处理

目标：在现有 MySQL / InfluxDB 私有数据链路和 Stage H / Stage I 资产之上，接入 DeepSeek 在线大模型，形成中期可写的 LLM 辅助时序数据预处理能力。

当前状态：

- 已完成文档计划：`docs/implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md`。
- 已新增 DeepSeek/OpenAI-compatible provider contract、strict response contract、agent-style prompt/harness v2、schema-repair harness、切片整合和下游消费 helper。
- 已完成 mock provider 测试，并覆盖 schema repair retry、prompt protocol、payload slicing 与 local merge。
- 已完成 DeepSeek v4-pro 小样本真实切片 run：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_summary.json`。
- 当前真实 run `request_count=8`、`error_count=0`、`field_semantic_count=24`、`weak_label_review_count=3`、`semantic_query_hint_count=4`、`runtime_explanation_count=4`。
- 当前 harness `prompt_version=stage_i_llm_preprocessing.agent_guardrails.v2`、`schema_version=stage_i_llm_preprocessing_context.v2`、`schema_repair_attempt_count=0`、`final_invalid_task_count=0`。
- 当前 slicing `field_semantics=2`、`schema_gap_policy=2`、`runtime_explanations=2`，大 payload 按 stable identifier 本地合并，不把全量高频时序一次外发。

默认 provider：

- `CHRONARIS_LLM_PROVIDER=deepseek`
- `CHRONARIS_LLM_MODEL=deepseek-v4-pro`
- 不默认使用 OpenAI，除非用户后续明确解除信息安全顾虑。

实际代码落点：

- `src/chronaris/llm/provider.py`
- `src/chronaris/llm/schemas.py`
- `src/chronaris/llm/prompts.py`
- `src/chronaris/pipelines/stage_i/llm/preprocessing.py`
- `src/chronaris/pipelines/stage_i/llm/harness.py`
- `src/chronaris/pipelines/stage_i/llm/slicing.py`
- `src/chronaris/pipelines/stage_i/llm/reporting.py`
- `scripts/stage_i/llm/run_preprocessing.py`
- `tests/test_stage_i_llm_preprocessing.py`

建议输入：

- MySQL 字段 label、measurement id、code id、sortie / view 元信息。
- InfluxDB 派生的 Stage H 窗口统计摘要。
- 当前 `2` 个 sortie、`3` 个双流 view、`111` 个窗口样本的 weak-label task summary。
- runtime schema contract、runtime semantic case 和 schema gap summary。

建议输出：

- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_context.json`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_field_semantics.jsonl`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/field_semantic_dictionary.csv`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_weak_label_review.jsonl`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/weak_label_llm_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_schema_gap_policy.json`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/runtime_llm_explanations.jsonl`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_harness_summary.json`
- `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_request_response_audit.jsonl`
- `docs/artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md`

验收：

- mock provider 测试通过，离线环境不依赖真实 API。
- DeepSeek 小样本真实切片 run 完成并落盘。
- 请求、响应、prompt version、input hash、harness verdict、slicing summary、错误样例、成本/延迟摘要可追溯。
- 输出只作为字段语义、预处理建议、weak-label 复核和 runtime 解释证据，不替代人工真值或核心因果证据。
- 文档回写 `docs/STATE.md`、本文件、`docs/artifacts/ARTIFACTS.md` 和 `docs/midterm/claims-matrix-*.md`。

下一步对比实验：

- `A0 baseline`：不接 LLM context，复用当前 Stage I task entries 与内置 semantic query bank。
- `A1 llm_context`：只把 `llm_preprocessing_context` attach 到 task entries，标签值保持不变，检查训练/eval 指标和报告可解释性变化。
- `A2 llm_semantic_hints`：在 event fusion support 中加入 whitelisted LLM semantic query hints，对比 query coverage、view ranking 和 attribution 分布。
- `A3 runtime_explanation`：对比 runtime replay 报告中有/无 LLM explanation 的 schema gap、weak-label boundary 和 semantic attribution 完整性。
- `A4 human_review`：抽取小样本字段/规则人工复核，统计 LLM 是否减少人工查表和规则解释成本。

## 已完成 P21：LLM preprocessing 融入管线的对比实验与中期结果落地

目标：在 P20 `llm_preprocessing_context` 已真实生成的基础上，用 A0-A4 对比实验说明 LLM 接入现有 Stage I 数据融合管线的增量价值，并把结果写成中期报告可直接引用的 summary。

结果：

- A1 `llm_context`：`333/333` 条 Stage I weak-label task entries 已 attach P20 context，代码逐 entry 检查 `label_changed_count=0`、`label_unchanged=true`。
- A2 `llm_semantic_hints`：内置 semantic query bank 从 `3` 条扩展到 `7` 条，新增 `4` 条 P20 hints 均通过 recipe whitelist；本轮没有从现有 summary 伪造 view ranking/top attribution 重算。
- A3 `llm_runtime_explanation`：`12` 条 runtime semantic cases 中 `4` 条有 P20 LLM explanation，解释子集四项完整性 `model_prediction / semantic_attribution / schema_gap_note / weak_label_boundary = 1.0`。
- A4 `human_review_packet`：生成 `15` 条人工复核材料，覆盖字段语义 `6`、weak-label rule `3`、schema gap policy `6`；`human_review_completed=false`，人工未填写前不写成验证完成。

计划入口：

- `docs/midterm/llm-preprocessing-comparison-plan-2026-06-14.md`
- 下一轮工作 prompt：`docs/implementation/notes/goal-prompt-stage-i-p21-llm-comparison-2026-06-14.md`

默认输入：

- P20 context：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_context.json`
- Stage I task manifest：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- semantic support summary：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- runtime schema contract：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
- runtime semantic case table：`docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`

实际新增实现：

- `src/chronaris/pipelines/stage_i/llm/comparison.py`
- `src/chronaris/pipelines/stage_i/llm/comparison_reporting.py`
- `scripts/stage_i/llm/run_preprocessing_comparison.py`
- `tests/test_stage_i_llm_comparison.py`

实验条件：

- `A0 baseline`：不接 LLM context；复用当前 Stage I task entries 与内置 semantic query bank。
- `A1 llm_context`：attach P20 context 到 Stage I task entries；已验证 `label_unchanged=true`。
- `A2 llm_semantic_hints`：只通过 recipe whitelist 接入 LLM semantic hints；已完成 query coverage 对比，ranking/attribution 重算需后续基于 Stage H tensor 复跑。
- `A3 llm_runtime_explanation`：已对比 runtime cases 有/无 LLM explanation 的报告完整性。
- `A4 human_review_packet`：已生成字段/规则/schema gap 小样本人工复核表；没有人工填写前只能写成 review packet，不写成人工验证完成。

必须落盘的工程产物：

- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/condition_manifest.json`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/task_context_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/semantic_hint_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/runtime_explanation_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/human_review_packet.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/midterm_claims_payload.json`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/progress.json`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/run.log`
- `docs/artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md`

必须落到 `docs/midterm/` 的写作产物：

- `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`
- 已更新 `docs/midterm/README.md`，加入 P21 result summary。
- 已更新 `docs/midterm/claims-matrix-2026-06-13.md`，新增 “LLM 接入带来可解释性/复核效率增量” 限域 claim。
- 已更新 `docs/midterm/boundaries-and-risks-2026-06-13.md`，保持 “LLM 不替代人工真值” 与 “human review 未完成” 边界。

验收：

- A0-A4 已生成本地可追溯记录。
- `label_unchanged=true` 已由代码检查得出。
- LLM semantic hints 已走 whitelist；自由文本 prompt 未进入融合模块。
- runtime explanation completeness 已检查 `model_prediction / semantic_attribution / schema_gap_note / weak_label_boundary` 四项。
- `human_review_packet.csv` 只作为人工复核材料；人工未填写前不得写成验证结论。
- `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md` 已存在，并能直接给中期报告引用。
- 相关测试通过；最终 `git diff --check` 作为本轮收口门禁。

## 已完成 P22：中期 thesis materials r3 图表质量刷新

目标：修复 P16/P18 thesis materials 中低信息或易误读图表，先生成 r3 图表包验证新版构图，同时保持 evidence layer、runtime schema、rotation、LLM 边界清楚；该图包后续由 r4/r5 接管，当前已由 P26/r6 接管。

结果：

- 新增/刷新代码：
  - `src/chronaris/pipelines/stage_i/evidence/thesis_materials.py`
  - `src/chronaris/pipelines/stage_i/evidence/thesis_materials_data.py`
  - `src/chronaris/pipelines/stage_i/evidence/thesis_materials_figures.py`
  - `src/chronaris/pipelines/stage_i/evidence/thesis_materials_report.py`
  - `scripts/stage_i/evidence/build_thesis_materials.py`
  - `tests/test_stage_i_thesis_materials.py`
- r3/r4 run 已由 r5 接管并从 docs 产物目录清理；当前 thesis materials 入口见 P26 的 `20260621T-stage-i-thesis-materials-r6-report-figure-polish`。
- 本轮输出 `8` 张 PNG 与 `8` 张 CSV：
  - `evidence_layer_overview`：从 artifact present=1 改为证据层级矩阵。
  - `runtime_payload_schema`：改为 native replay payload 与 canonical service payload 字段契约对照。
  - `rigid_body_rotation_audit`：改为 family loss log-scale 对比 + pitch/roll/yaw angle/rate 可用性矩阵。
  - `weak_label_sweep_ablation`：改为 proxy/live best metrics、completed/partial 状态和小网格 lag heatmap。
  - `chronaris_opt_component_ablation`：改为 T1/T2/T3 分任务面板 + normalized delta 贡献。
  - `public_transfer_boundary`：完全中文化，改为公开适配、私有弱标注主线、私有代理消融的正向分工图。
  - `semantic_event_fusion_overview`：改为双流输入、causal mask、event token、semantic query、query-to-event attribution 和 view-level attribution。
  - `llm_comparison_a0_a4`：纳入 P20/P21 A0-A4 对比，明确 LLM 仅作为 preprocessing context / whitelisted hints / runtime explanation / pending review packet。

增强项实际执行：

- 已重新运行 rotation metadata audit：
  - `docs/artifacts/assets/stage_i_rotation_audit/20260619T-stage-i-rotation-audit-r3-figure-refresh/rigid_body_rotation_audit_summary.json`
  - 结果仍为 `rotation_status=disabled`；pitch/roll/yaw angle 有候选，pitch_rate/roll_rate/yaw_rate 仍缺失。
- private component ablation 使用已有 `27` 行 r2 artifact 重绘，未重跑。
- weak-label sweep 使用已有 stable/partial artifacts 重绘，未扩大 live_influx 网格。
- semantic 与 LLM 使用已有 Stage I semantic support、P20/P21 artifacts 重绘；未从 summary 倒推 attribution 改善，也未声称人工复核完成。

验收：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_thesis_materials tests.test_stage_i_support tests.test_runtime_service_smoke tests.test_stage_i_llm_preprocessing tests.test_stage_i_llm_comparison`：`Ran 16 tests in 2.878s`，`OK`。
- `build_thesis_materials.py` 已用 r3 run_id 实跑；r3/r4 输出已由 P25/r5 图包替代并清理。
- PIL/pandas 验证已确认 `8` 个 PNG 非空且可读取、`8` 个 CSV 有行且关键字段存在、`figure_manifest.json/table_manifest.json` 均为 `8` 项、报告中图表路径可解析。
- 2026-06-19 复核后，旧 r1、r2-p18 thesis-materials PNG/manifest/report 与 r3 thesis-materials 图包已清理，避免中期写作误用；`r2-p18/runtime_semantic_case.csv` 因仍被 P20/P21 LLM preprocessing 复现实验引用而保留。

## 已完成 P23：runtime semantic case r4 图表质量刷新

目标：把 r2-p18 仅保留为 P20/P21 历史输入的 `runtime_semantic_case.csv` 重新纳入当前 thesis materials 生成链路，并替换旧 runtime/semantic case 中近乎平坦折线、重复归因柱和大号 965 vs 1930 字段数对比。

结果：

- r4 run 已完成验证，随后被 P25/r5 接管并从 docs 产物目录清理；当前 run 见 P26/r6：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/`
  - `docs/artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md`
- r4 输出过 `9` 张 PNG 与 `9` 张 CSV；r5 输出 `11` 张 PNG 与 `11` 张 CSV；当前 r6 输出 `12` 张 PNG 与 `12` 张 CSV，并保留修复后的 `runtime_semantic_case.png/csv`。
- `runtime_semantic_case` 图改为：
  - 顶部 case card：view_id、展示窗口数、native/canonical 状态、vehicle 字段数、missing groups。
  - 中部 lollipop/dot plot：按窗口展示 `semantic_top_event_attribution`，只标注变化点和最高值，并按 `semantic_top_query_name` 着色。
  - 侧边 query 类型分布。
  - 底部 risk/workload/event score 范围 chip 与 schema 状态摘要。
- 当前 `figure_manifest.json` 与 `table_manifest.json` 均包含 `runtime_semantic_case`。
- 当前 `figure_quality_audit.csv` 记录 12 张当前图的 issue/action/replacement/qa_status、宽高、最小字号和内部词检查。

边界：

- r2-p18 `runtime_semantic_case.csv` 继续保留为 P20/P21 LLM preprocessing 历史输入表；r6 只是把该表作为当前中期图表生成输入，不改写历史 LLM summary。
- runtime 仍写成 `native aligned / canonical exact`；不声称原生 replay payload 已 exact，也不声称生产级在线服务。
- LLM 仍只作为 preprocessing context / whitelisted hints / runtime explanation / pending review packet，不写成人工真值或核心因果证据。

## 已完成 P24：leakage-safe private component ablation

目标：在不改写历史 private proxy 结果的前提下，新增 `protocol=leakage_safe_v1` 的 T1/T2/T3 组件消融协议，审计标签来源字段与输入特征同源风险，并把模型骨干结构消融和任务适配层消融拆分为独立产物。

结果：

- 新增代码：
  - `src/chronaris/pipelines/stage_i/private/leakage_audit.py`
  - `src/chronaris/pipelines/stage_i/private/leakage_safe_ablation.py`
  - `scripts/stage_i/private/run_leakage_safe_ablation.py`
  - `tests/test_stage_i_leakage_safe_ablation.py`
- 新 run：
  - `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/`
  - `docs/artifacts/stage_i/stage-i-private-leakage-safe-ablation-20260619T-stage-i-leakage-safe-ablation-r2.md`
- 输出包括：
  - `ablation_summary.json`
  - `label_feature_overlap_audit.json/csv`
  - `seed_metrics.csv`
  - `split_manifest.json`
  - `cross_view_metrics.csv`
  - `cross_sortie_metrics.csv`
  - `model_backbone_ablation.csv/json/png`
  - `task_adapter_ablation.csv/json/png`
  - `t2_error_distribution.png`
  - `t3_similarity_distribution.csv/png`

当前结果：

- `audit_status=pass`，未发现直接标签源重叠、标签确定性派生特征输入或身份/时间位置字段输入。
- records：`sample_count=111`、`view_count=3`、`sortie_count=2`。
- T1/T2/T3 seed rows 均完成；T3 对单飞行员视图记录 skipped fold，不生成伪正样本。
- 完整防泄漏任务输入：
  - T1 macro-F1=`0.17333333333333334`，balanced accuracy=`0.3333333333333333`。
  - T2 RMSE=`862.6941748579226`，NRMSE=`0.3695127256456088`，persistence RMSE=`201.4895651832178`，相对持久性基线改进率=`-3.281582393973904`。
  - T3 Top-1=`0.0`，Top-3=`0.0`，Top-5=`0.0`，MRR=`0.0114187054292705`，valid query=`74`，candidate=`8140`。
- `t3_similarity_distribution.csv` 对原始 `488400` 行进行确定性压缩，保留全部正样本并抽样负样本，写出 `20000` 行。

边界：

- 历史 private component ablation r2 仍保留为历史 private proxy 诊断；论文实验章节优先引用本轮 `leakage_safe_v1` 结果。
- T3 严格协议下已完成评价，但当前安全向量 Top-1/Top-3/Top-5 均为 `0.0`，不能写成已解决配对检索。

## 已完成 P25：thesis materials r5 leakage-safe refresh

目标：在 r4 runtime case refresh 基础上，把论文图件全部改为 A4 中期报告可用的中文 300dpi 图，并接入 P24 的 leakage-safe 消融结果。

结果：

- r5 曾输出 `11` 张 PNG 与 `11` 张 CSV，并已在 2026-06-21 深度清理中由 P26/r6 接管后从 docs 产物目录删除；如需追溯 r5，请从 git 历史读取。
- r5 输出内容包括：
  - `evidence_layer_overview`
  - `runtime_payload_schema`
  - `runtime_semantic_case`
  - `rigid_body_rotation_audit`
  - `weak_label_sweep_ablation`
  - `chronaris_opt_component_ablation`
  - `model_backbone_ablation`
  - `task_adapter_ablation`
  - `public_transfer_boundary`
  - `semantic_event_fusion_overview`
  - `llm_comparison_a0_a4`
- `figure_quality_audit.csv` 曾确认 11 张 PNG 均存在、非空、DPI 约 `300`，并与 table/figure manifest 数量匹配；当前图表 QA 以 P26/r6 的 `figure_quality_audit.csv` 为准。
- r5 图件可见文字已中文化；内部 protocol、run id、source path 和枚举保留在 CSV/JSON/manifest 中用于追溯。

边界：

- `semantic_event_fusion_overview` 没有伪造完整 view-query 归因热力图；当前源产物缺少完整“数据视图 × 查询类型”矩阵时，图中显示缺源需求。
- `runtime_semantic_case` 只展示当前真实覆盖的风险与工作负荷查询，不生成不存在的事件复盘查询。
- `chronaris_opt_component_ablation` 作为兼容总览图保留；正式组件消融推荐使用 `model_backbone_ablation.png` 与 `task_adapter_ablation.png`。

## 已完成 P26：thesis materials r6 report figure polish

目标：在 r5 基础上按中期报告 A4 落版可读性重绘 Chronaris 侧图件，不处理本地论文工作区的 imagegen 框图；r6 验证通过后，r5 已在 2026-06-21 深度清理中从 docs 产物目录移除。

结果：

- 新 run：
  - `docs/artifacts/assets/stage_i_thesis_figures/20260621T-stage-i-thesis-materials-r6-report-figure-polish/`
  - `docs/artifacts/stage_i/stage-i-thesis-materials-20260621T-stage-i-thesis-materials-r6-report-figure-polish.md`
- 当前输出 `12` 张 PNG 与 `12` 张 CSV：
  - `evidence_layer_overview`
  - `runtime_payload_schema`
  - `runtime_service_flow`
  - `runtime_semantic_case`
  - `rigid_body_rotation_audit`
  - `weak_label_sweep_ablation`
  - `chronaris_opt_component_ablation`
  - `model_backbone_ablation`
  - `task_adapter_ablation`
  - `public_transfer_boundary`
  - `semantic_event_fusion_overview`
  - `llm_comparison_a0_a4`
- `figure_quality_audit.csv` 已确认 12 张 PNG 均存在、非空、DPI 约 `300`，并记录宽高、中文标签策略、最小字号、内部词检查和长 ID 处理策略，所有 `qa_status=pass`。
- `figure_manifest.json` 不再写入 `leakage_safe` 路径串；消融图图内与 manifest 使用“严格评价协议/严格评价”口径，源产物路径仍保留在 CSV/summary 层用于追溯。

边界：

- `semantic_event_fusion_overview` 仍不伪造完整视图 × 查询类型归因数值；当前缺完整归因矩阵时只画覆盖/支撑状态。
- `weak_label_sweep_ablation` 未补不存在的风险阈值/负荷阈值网格；图中只展示已落盘的滞后窗口趋势与参数组合覆盖。
- `runtime_service_flow` 基于现有 runtime service summary 与 schema contract 重绘；不依赖已从 docs/LFS 清理的原始 sample JSONL。

## 已完成维护 P27：源码、脚本与 docs/artifacts 深度清理

目标：对 `src/` 中无用源码缓存、Stage I 脚本入口和 `docs/artifacts/` 冗余产物做一次保守深度清理，并把 docs 入口统一到当前 r6 图包。

结果：

- 清理记录：`docs/artifacts/cleanup/20260621-deep-cleanup.md`。
- 已删除被 P26/r6 接管的 P25/r5 thesis figure 图包与报告；当前中期图表入口只保留 r6。
- 已删除 P11 stable/partial 目录下两个空 `runs/` 子目录。
- 已清理本地 `src/`、`tests/`、`third_party/` 的 Python 编译缓存；这些缓存由 `.gitignore` 覆盖，不属于可提交源码。
- 已完成 tracked `src/chronaris` 与 `scripts/stage_i` 引用扫描；当前 canonical 源码与脚本仍被 tests、docs 或 evidence runner 依赖，本轮不删除 tracked canonical 入口。

保留边界：

- `20260613T-stage-i-p11-live-influx-r1` 的 child run summary/log 被 r3 resume 和 r4 partial summary 引用，保留；checkpoint binary 仅保留在远程开发机本地备份或重新生成，不进入 git。
- `20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv` 是 P20/P21 LLM preprocessing 历史输入表，保留。
- public adapter 的 `20260508T125651Z` sklearn baseline 与 `20260508T090700Z` torch baseline 被 public calibration/transfer boundary 与代码常量引用，保留。

## 已完成 Public-P27/Public-P28：public model comparison and fusion refresh

目标：面向中期报告补齐公开 UAB/NASA 模型对比、chronaris_public_fusion 长程 refresh、resume/partial/log/progress contract，并在 P28 后重新运行 P27 生成最终图表。

代码入口：

- P27 builder：`src/chronaris/pipelines/stage_i/public/model_comparison.py`
- P27 CLI：`scripts/stage_i/public/build_public_model_comparison.py`
- P28 refresh：`src/chronaris/pipelines/stage_i/public/fusion_refresh.py`
- P28 CLI：`scripts/stage_i/public/run_public_fusion_refresh.py`
- 长程训练日志与 checkpoint 补强：`src/chronaris/pipelines/stage_i/public/deep_baseline.py`、`src/chronaris/pipelines/stage_i/public/deep_baseline_runtime.py`

P28 输出：

- asset root：`docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/`
- report：`docs/artifacts/stage_i/stage-i-public-fusion-refresh-20260701T-stage-i-public-fusion-refresh-r1.md`
- status：`completed`
- core files：`fusion_refresh_summary.json`、`screen_leaderboard.csv`、`confirm_leaderboard.csv`、`fold_metrics.csv`、`training_curves.csv`、`best_by_dataset_task.json`、`evidence_manifest.json`、`run.log`、`progress.json`
- figures：`fig_public_fusion_refresh_confirm_vs_baselines.png`、`fig_public_fusion_refresh_delta_heatmap.png`、`fig_public_fusion_config_sensitivity.png`、`fig_public_fusion_training_curves_best.png`、`fig_public_fusion_win_summary.png`

P27 输出：

- asset root：`docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/`
- report：`docs/artifacts/stage_i/stage-i-public-model-comparison-20260701T-stage-i-public-model-comparison-r1.md`
- core files：`model_comparison_long.csv`、`model_comparison_wide.csv`、`improvement_summary.csv`、`evidence_manifest.json`
- figures：`fig_public_model_leaderboard_nasa_macro_f1.png`、`fig_public_model_leaderboard_nasa_balanced_accuracy.png`、`fig_uab_subjective_rmse_comparison.png`、`fig_uab_objective_macro_f1_comparison.png`、`fig_public_model_delta_heatmap.png`、`fig_public_model_win_summary.png`
- manifest：`p28_refresh_included=true`、`missing_metrics=[]`

实际命令：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/stage_i/public/run_public_fusion_refresh.py \
  --run-id 20260701T-stage-i-public-fusion-refresh-r1 \
  --prepare-sequences \
  --datasets nasa_csm uab_workload_dataset \
  --screen-epochs 5 \
  --confirm-epochs 20 \
  --screen-max-folds 2 \
  --screen-candidate-limit 2 \
  --confirm-top-k 1 \
  --confirm-seeds 42 \
  --device cuda

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/stage_i/public/run_public_fusion_refresh.py \
  --run-id 20260701T-stage-i-public-fusion-refresh-r1 \
  --nasa-root /tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/nasa_csm \
  --uab-root /tmp/chronaris_stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/uab_workload_dataset \
  --datasets nasa_csm uab_workload_dataset \
  --screen-epochs 5 \
  --confirm-epochs 20 \
  --screen-max-folds 2 \
  --confirm-max-folds none \
  --screen-candidate-limit 2 \
  --confirm-top-k 1 \
  --confirm-seeds 42 \
  --device cuda \
  --resume \
  --skip-completed \
  --allow-partial \
  --confirm-only \
  --heartbeat-seconds 60 \
  --batch-log-interval 20

/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/stage_i/public/build_public_model_comparison.py \
  --run-id 20260701T-stage-i-public-model-comparison-r1 \
  --refresh-summary docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/fusion_refresh_summary.json
```

验收：

- `compileall src/chronaris/pipelines scripts/stage_i/public` 通过。
- `unittest tests.test_stage_i_public_model_comparison tests.test_stage_i_public_fusion_refresh tests.test_stage_i_deep_pipeline` 通过：`Ran 22 tests`，`OK (skipped=2)`。
- P28 `missing_figures=[]`，P27 `missing_metrics=[]`。
- public 数据集仍按 `public adapter / calibration / context-proxy evidence` 引用，不改标签、不改 split、不使用 test fold 统计量做训练归一化或调参。

## 中期前边界管理

下面几类工作现在纳入中期前主动任务，但必须按证据分层写清楚，不能因为加做实验就改变论文边界。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前允许有限预算复现/补跑；只能作为公开 adapter baseline / calibration baseline。
- NASA/UAB 公开数据适配器结果：中期前要整理成 public adapter evidence 和 transfer boundary；不能改写成论文双流本体闭环。
- `chronaris_opt` 与 `T1/T2/T3`：中期前要补机制诊断；仍只能写成 private proxy benchmark evidence，不能写成人工真值 thesis task fully closed。
- `risk_proxy / workload_proxy / event_replay_tag`：中期前要补小网格与消融；仍只能写成 thesis weak-label evidence。
- DeepSeek 在线 LLM 预处理：P20 已接入字段语义归一、weak-label 复核、schema gap policy、runtime 解释和切片整合；仍不能写成 OpenAI 默认接入、人工真值替代、原始全量数据外发或核心因果证据。
- `rigid_body rotation`：中期前必须核验字段；启用或缺失都要以 diagnostics 形式固化。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。

## 历史计划入口

- [notes/coding-roadmap.md](notes/coding-roadmap.md)
- [notes/stage-i-thesis-mainline-roadmap-2026-05-15.md](notes/stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md](notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [notes/thesis-coding-gap.md](notes/thesis-coding-gap.md)
- [notes/iteration-playbook.md](notes/iteration-playbook.md)
