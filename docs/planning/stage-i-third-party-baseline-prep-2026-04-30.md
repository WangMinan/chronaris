# 阶段 I 第三方基线准备记录（2026-04-30）

## 1. 结论

本轮已完成 Stage I 横向基线对比的前置准备：

1. 已将 `MulT` 以上游快照形式 vendor 到 `third_party/mult/`。
2. 已将 `SeqML/ContiFormer` 子目录以上游快照形式 vendor 到 `third_party/contiformer/`。
3. 已确认本地公开数据根目录 `/home/wangminan/dataset/chronaris` 中，当前主线所需的 `uab_workload_dataset` 与 `nasa_csm` 都已在位。

本轮不采用 `git submodule`。

## 2. 为什么不用 submodule

- `MulT` 与 `ContiFormer` 后续都需要按 Chronaris 的 `window_v2 + LOSO + Stage I report contract` 做适配。
- 直接 vendor 更适合在当前仓库内修改 wrapper、修补依赖与补测试。
- `ContiFormer` 上游并不是单独仓库，而是 `SeqML` 下的子目录；为一个很小的子目录引入整仓 submodule 不划算。

## 3. 已落地的上游快照

### MulT

- local path: `third_party/mult`
- upstream: `https://github.com/yaohungt/Multimodal-Transformer`
- commit: `a670936824ee722c8494fd98d204977a1d663c7a`
- snapshot scope: full repository
- size: `452K`

### ContiFormer

- local path: `third_party/contiformer`
- upstream: `https://github.com/microsoft/SeqML/tree/main/ContiFormer`
- SeqML commit: `1ecaa5b28fd14fa30eabf5c7de9fe11444e315ce`
- snapshot scope: `ContiFormer/` subtree only
- size: `36K`

## 4. 本地数据根目录核验

数据根目录：`/home/wangminan/dataset/chronaris`

当前看到的关键目录：

- `uab_workload_dataset`：`971M`
- `nasa_csm`：`7.3G`
- `matb_ii_workload`：`1.7G`
- `eeg-mat`：`176M`
- `ds007262`：`3.0M`
- `on_site`：`29G`

对本轮 Stage I 横向基线对比的直接判断：

- `uab_workload_dataset`：已满足当前主线准备条件
- `nasa_csm/extracted`：已满足当前主线准备条件
- `Stage H` 真实双流资产：继续沿用既有 `run_manifest.json`，不需要额外挪动数据
- `matb_ii_workload / eeg-mat / ds007262`：暂不作为本轮第一优先级

## 5. 下一步边界

本轮只完成“第三方代码快照落地 + 数据根目录核验”。

下一步才进入真正的适配实现：

1. 设计 `MulT / ContiFormer` 的 Chronaris sequence export contract。
2. 先跑 `UAB window_v2 + MulT`。
3. 再补 `NASA attention_state + MulT`。
4. 最后再评估 `ContiFormer` 是否值得接入同一批对比。
