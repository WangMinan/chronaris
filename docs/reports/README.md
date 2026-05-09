# Reports Index

更新时间：2026-05-09

`docs/reports` 现按一级目录分为五组，顶层不再堆放阶段报告：

- `alignment/`
  - 阶段 E/F/G 的连续对齐与因果融合主报告
- `preview/`
  - preview / 可用性 / overlap 验证类报告
- `stage_h/`
  - Stage H 导出与 all-window 清洗报告
- `stage_i/`
  - Stage I 公共 benchmark、case study、deep baseline、support、midterm evidence 与 unified public mainline 报告
  - 目录内再分 `archive/`，见 `stage_i/README.md`
- `private/`
  - 鼎新私有 benchmark、`chronaris_opt` 支撑与 package 固化报告
  - 目录内再分 `archive/`，见 `private/README.md`

当前引用规则：

- 阶段状态判断仍以 `docs/planning/coding-roadmap.md` 为准。
- 当前论文式中期整编优先看 `stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md`。
- 当前公开主线统一结论优先看 `stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`。
- 当前私有最优性与 package 固化优先看 `private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md`。
- 阶段 I / private 的历史快照默认不再留在顶层，统一下沉到各自 `archive/` 子目录。
- 历史中间态若已被更高层主报告覆盖，不再回到 `docs/reports` 顶层新增平铺 Markdown。
