# 毕业论文编码缺口评估

更新时间：2026-05-07

## 1. 目的

这份文档只回答一件事：

- 结合 `docs/选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx` 与当前仓库/报告现状，为了支撑毕业论文，编码层面还差什么

它不重写阶段历史 closure，也不把 proxy 标签包装成人工真值。

## 2. 当前已经完成到什么程度

对照选题报告里的三段主线：

1. `基于连续时间动力学的异步数据对齐`
2. `基于语义解耦的非对称因果融合`
3. `融合管线的系统级实现与效能验证`

当前仓库已经可以成立的结论是：

- `E / F / G(min) / H` 已完成真实数据链路、导出 contract 与测试闭环。
- `Stage I Phase 0 + Phase 1 + Phase 2 + Phase 3` 的历史公开 benchmark closure 已完成。
- `chronaris_opt` 已在鼎新私有 proxy benchmark 的 `T1 / T2 / T3` 三任务上达到当前对照矩阵最优，并完成 package 固化。
- `alignment support`、`causal support`、`fixed six-path ablation` 三份论文证据 support 已落盘。
- thesis-facing `runtime/demo` 入口已补齐并已正式落盘，支持 `Stage H run manifest` 与 `optimized_candidate_package.json` 两类输入。
- `关键工况锚定` 导出 contract 已补齐并已正式落盘，可稳定产出 `anchor_manifest.json / anchor_windows.csv / anchor_report.md`。
- 当前公开主线不是“全赢”，而是 `NASA closed, UAB partial`。

因此，当前真正没完成的已经不是“E/F/G/H 还没做出来”，而是论文面向的最后一层编码收束。

## 3. 仍未完成的编码工作

### 3.1 必须至少做出取舍的项

1. 文档口径与论文主线状态同步
   - 现状：`coding-roadmap.md` 的历史 closure 语义，容易被误读成“论文主线已经全部编码完成”。
   - 当前更准确的事实应拆成三层：
     - `私有主线已闭合`
     - `公开主线 NASA closed / UAB partial`
     - `论文 support 已补齐，但不等于公开/私有所有口径都能写成全面 superiority`
   - 这项工作本轮已经开始回写到 `coding-roadmap.md`。

2. 公开 quantitative 主线仍未闭合
   - 当前公开主线结论仍然是 `NASA closed, UAB partial`。
   - 因此如果论文要写成“Chronaris 在公开数据上全面优于 `MulT / ContiFormer`”，编码和实跑工作都还没有结束。
   - 当前剩余的最主要不确定性已经从“有没有入口”转成“UAB clean win 能不能真正跑出来”。

### 3.2 只在想增强论文定量说服力时继续做的项

1. 公开主线的 `UAB clean win` 还没有完成
   - 当前公开主线结论来自 `docs/reports/stage_i/stage-i-public-mainline-20260507T024112Z-stage-i-public-mainline.md`。
   - 当前事实是：
     - `NASA combined macro-F1 = 0.4550`，相对 `MulT / ContiFormer` 已闭合
     - `UAB subjective` 仍未同时超过 `ContiFormer`
   - 所以：
     - 若论文只需要“公开数据有可复现实证，且至少有一条主线闭合”，当前可以成立
     - 若论文想写成“Chronaris 在公开数据上全面优于 `MulT / ContiFormer`”，则编码工作仍未完成

2. `chronaris_public_fusion` 仍只是 exploratory branch
   - 当前 confirm：`NASA combined macro-F1 = 0.3348 < 0.40`
   - 它可以继续作为一次性强化尝试，但当前不能替代 paper-facing public mainline
   - 若继续做，建议严格按 `NASA-first confirm` 再跑一次；若仍不过 gate，就停止该支线

### 3.3 当前不建议继续扩的项

1. 不继续新增更多公开数据集来拖大 Stage I 边界。
2. 不把私有 `proxy / weak-label` 任务写成真实 `G-LOC` 或人工真值最优。
3. 不为了论文措辞去重写上游 receiver / 入库链路。
4. 不在这一轮把最小 runtime 入口误膨胀成完整服务化系统。

## 4. 推荐的编码收束顺序

1. 先同步 `coding-roadmap.md` 与文档索引，明确“历史收口”和“当前论文主线”是两件事。
2. 当前 `runtime/demo` 与 `anchor` 入口已经补齐并完成正式资产落盘，论文正文可以继续保留“系统级推理入口/关键工况锚定”表述，但要注意把它写成“最小研究原型”而不是完整在线服务。
3. 接下来如果还要增强 quantitative 说服力，主线只剩公开数据继续冲线。
4. 当公开主线口径定稿后，再冻结编码主线，转入论文整编与图表整理。

## 5. 当前判断

当前仓库已经足够支撑论文的核心方法链路：

- 连续对齐
- 物理约束
- 非对称因果融合
- 标准化导出
- 真实 sortie case study
- 私有 proxy 最优性
- 公开数据的可复现实证

但如果把要求提高到下面这两个版本，则编码工作还不能算结束：

1. `公开数据上全面优于 MulT / ContiFormer`
2. `公开 quantitative 主线已经完全 closed`

换句话说，当前剩余工作主要是 thesis-facing closure，而不是底层模型还没搭起来。

## 6. 本轮核查

本轮已直接核查：

- 选题报告 `.docx`
- `docs/planning/stage-i-public-opt-win-plan-2026-05-06.md`
- `docs/reports/stage_i/` 当前主报告与 support 报告
- `src/chronaris/serving/` 现状
- 新增 `runtime/demo` 与 `anchor` 代码入口及其正式资产输出：
  - `docs/reports/assets/stage_i_runtime_demo/20260506T165435Z-stage-i-runtime-demo/`
  - `docs/reports/assets/stage_i_anchor/20260506T165435Z-stage-i-anchor/`

本轮已运行并通过的相关测试：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_h_export tests.test_stage_i_public_opt tests.test_stage_i_private_optimization tests.test_stage_i_support
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_deep_pipeline
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_case_study
```

当前残余技术风险：

- `stage_i_metrics.py` 出图时仍有 Matplotlib 中文缺字 warning；不影响指标，但论文图件出图前最好处理。
