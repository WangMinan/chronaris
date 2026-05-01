# Third-Party Baselines

本目录存放阶段 I 横向基线对比所需的上游代码快照。

当前策略：

- 不使用 `git submodule`
- 直接 vendor 上游快照到仓库内
- 用固定 commit 记录来源，后续如需 patch 直接在当前仓库完成

这样做的原因：

- `MulT` / `ContiFormer` 都需要按 Chronaris 的 `UAB/NASA + LOSO + window_v2` contract 做适配，直接 vendor 比 submodule 更容易修改与回归测试。
- `ContiFormer` 在上游并不是独立仓库，而是 `SeqML` 下的一个子目录；为一个小子目录引入整仓 submodule 成本过高。
- 当前快照体积很小，vendor 不会给仓库带来明显负担。

当前已落地的上游来源：

- `third_party/mult`
  - upstream: `https://github.com/yaohungt/Multimodal-Transformer`
  - commit: `a670936824ee722c8494fd98d204977a1d663c7a`
  - scope: full repository snapshot
- `third_party/contiformer`
  - upstream: `https://github.com/microsoft/SeqML/tree/main/ContiFormer`
  - seqml commit: `1ecaa5b28fd14fa30eabf5c7de9fe11444e315ce`
  - scope: `ContiFormer/` subdirectory snapshot plus upstream MIT license copy

注意：

- 本目录下的代码是上游快照，不代表已经完成 Chronaris 适配。
- 后续如需运行或 patch，应优先在 `src/chronaris/pipelines` / `src/chronaris/features` / `src/chronaris/dataset` 中写 wrapper，而不是直接把 Chronaris 逻辑塞进第三方源码。
