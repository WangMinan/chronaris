# 2026-07-01 src/docs artifact prune

本记录对应 2026-07-01 的 `src/` 过时脚本与 `docs/` 过时产物清理。

## 清理范围

- `src/`、`tests/`、`scripts/` 下的本地 Python `__pycache__` 生成缓存已删除。
- `src/chronaris` 下没有删除源码模块：本轮复查确认 Stage I public/private/evidence/legacy 模块仍被脚本、测试、兼容 import map 或当前报告链路引用。
- `docs/artifacts/assets/` 下删除 archive-only 的旧 public 迭代资产：
  - `stage_i_public_opt_torch/20260506T063146Z-stage-i-public-opt-uab-torch/`
  - `stage_i_public_opt_torch/20260507T133000Z-stage-i-public-opt-uab-torch-mainline-r2/`
  - `stage_i_public_opt_torch/20260507T134500Z-stage-i-public-opt-uab-torch-sessionmean-r3/`
  - `stage_i_public_opt_torch/20260507T141500Z-stage-i-public-opt-uab-torch-sessionpooled-r4/`
  - `stage_i_public_opt_torch/20260507T142500Z-stage-i-public-opt-uab-torch-sessionpooled-scalar-r5/`
  - `stage_i_public_mainline/20260506T064302Z-stage-i-public-mainline/`
  - `stage_i_public_mainline/20260507T024112Z-stage-i-public-mainline/`
  - `stage_i_public_mainline/20260507T143000Z-stage-i-public-mainline-uab-iteration/`
  - `stage_i_public_mainline/20260508T091000Z-stage-i-public-mainline-uab-heat-specialist-r1/`
  - `stage_i_public_opt/20260506T124500Z-stage-i-public-opt-nasa/`
  - `stage_i_public_fusion_screen/20260506T-stage-i-public-fusion-screen-round1/`
- 同步删除上述 archive-only 资产对应的 `docs/artifacts/stage_i/archive/public_history/` 报告。

## 保留边界

- 保留 P27/P28 当前公开对比与 fusion refresh 产物：
  - `stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/`
  - `stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/`
- 保留仍被代码或当前报告直接引用的 public adapter 基线：
  - `stage_i_public_opt/20260506T121000Z-stage-i-public-opt-uab/`
  - `stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1/`
  - `stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/`
  - `stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch/`
  - `stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch-gpu/`
  - `stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/`
  - `stage_i_public_fusion_screen/20260506T-stage-i-public-fusion-screen-round2/`
- 保留 P11 live child run、Stage H 输入、P20/P21 LLM 输入表、P26 r6 thesis figures、leakage-safe private ablation 与当前中期写作入口。

## 体积与 LFS 结论

- 当前树清理后，`docs/` 从约 `364M` 降到约 `299M`，`src/` 从约 `5.1M` 降到约 `2.4M`。
- `git lfs migrate info --include-ref=refs/heads/main --include='docs/**'` 审计显示 `docs/**` 历史 LFS objects 约 `165 MB`，与 2026-06-19 清理后的量级接近；本轮没有发现需要立即重写 git 历史的异常膨胀项。
- 因此本轮只做当前树清理、普通 commit 和普通 push，不执行 `git filter-repo` 或 force push。
