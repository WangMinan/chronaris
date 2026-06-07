# Stage I 主线迁移计划

更新时间：2026-05-04

> 说明：本文件保留为 `chronaris_opt` 升级为鼎新私有主线时的历史节点快照。
> 当前现行入口改为 `docs/planning/stage-i-thesis-mainline-roadmap-2026-05-15.md` 与 `docs/planning/stage-i-thesis-mainline-coding-plan-2026-05-15.md`。

## 1. 目的

本文件只回答一件事：

- 在 `chronaris_opt` 已经完成鼎新私有 proxy benchmark full LOSO 的前提下，仓库当前主线应该如何调整

本文件不重写历史收口事实，也不把 proxy 标签改写成人工真值。

## 2. 当前判断

当前可以成立的最强表述是：

1. `chronaris_opt` 已在鼎新私有 proxy benchmark 的 `T1/T2/T3` 三任务上达到当前对照矩阵最优。
2. 该对照矩阵已经覆盖 `naive_sync / E / F / G(min) / no-mask / MulT / ContiFormer`。
3. `chronaris_opt` 不是独立于 `E/F/G/H` 的平替模块，而是建立在下面三层依赖上的当前最优候选：
   - `F full` hidden/projection
   - `G` causal fusion 配置与 no-mask 配对诊断
   - `Stage H all-window` 导出 contract

因此主线迁移应表述为：

- `chronaris_opt` 升级为“当前鼎新私有任务验证主线”
- `E/F/G/H` 降级为“仍需保留的历史基线与导出依赖”

而不是：

- 删除 `E/F/G/H` 收口文档
- 覆盖 `E/F/G/H` 的历史结论
- 把 `chronaris_opt` 写成脱离 `Stage H` contract 的全新底座

## 3. 可替换与不可替换

### 可替换

1. 当前主线叙述：
   - 从“Stage I 增强实验 / 私有优化候选”改为“鼎新私有主线”
2. 当前优先级叙述：
   - 从“继续扩公开数据或第三方 baseline”改为“先围绕 `chronaris_opt` 整编主线，再推进 public opt”
3. 当前论文任务定位：
   - 将 `chronaris_opt` 作为当前自有数据最强证据入口

### 不可替换

1. `E/F/G/H` 的历史收口文档与工件
   - 它们是 `chronaris_opt` 的前置依赖，不应删除
2. `Stage H` 导出 contract
   - 当前私有 benchmark 直接消费该 contract，不能废除
3. 2026-04-30 的 Stage I 公开数据 closure
   - 应保留为公开 benchmark 历史主线，不与私有 proxy 最优性结论混写

## 4. 本轮调整目标

1. 在 `coding-roadmap.md` 中把 `chronaris_opt` 明确提升为当前鼎新私有主线。
2. 在 `AGENTS.md` 中把默认工作方式改成：
   - 保留 `E/F/G/H` contract
   - 优先维护 `chronaris_opt` 主线
   - 下一步进入 `chronaris public opt`
3. 在 `docs/README.md` 中明确：
   - 当前鼎新私有最优性依据看 `archive/stage_i/stage-i-private-benchmark-plan-2026-05-02.md`
   - `thesis-support-assessment-2026-05-01.md` 是 private-opt 之前的快照
4. 在 `tests/README.md` 中补齐当前鼎新主线回归入口。
5. 补一份真实 `chronaris_opt` package 固化产物，避免当前最佳工件只停留在 summary / metrics。

## 5. 下一轮编码入口

当前文档口径同步完成后，下一轮不再继续讨论“要不要保留 `MulT / ContiFormer` 为当前鼎新主线”。

当前这一步已经完成真实 package 固化：

- `docs/reports/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`

下一轮直接进入：

1. 设计 `chronaris public opt` 的 sequence contract 对应物
2. 先以 `UAB subjective regression` 为首个目标任务
3. 再决定是否把同一思路扩到 `NASA attention_state`

## 6. 验证门槛

本轮主线迁移只在以下条件同时满足时成立：

1. `coding-roadmap.md`
2. `AGENTS.md`
3. `docs/README.md`
4. `tests/README.md`

四处口径一致。

建议最小验证命令：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_h_export tests.test_stage_i_deep_pipeline tests.test_stage_i_private_optimization
```
