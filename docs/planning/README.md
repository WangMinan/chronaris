# Planning 索引

更新时间：2026-05-15

## 1. 当前应使用的文档

当前 `planning` 目录只把下面几份文档视为现行入口：

- [coding-roadmap.md](coding-roadmap.md)：仓库阶段状态与当前最高优先级工作包。
- [stage-i-thesis-mainline-roadmap-2026-05-15.md](stage-i-thesis-mainline-roadmap-2026-05-15.md)：下一阶段的指导性路线图。
- [stage-i-thesis-mainline-coding-plan-2026-05-15.md](stage-i-thesis-mainline-coding-plan-2026-05-15.md)：按文件拆解的详细编码计划。
- [thesis-coding-gap.md](thesis-coding-gap.md)：当前实现与选题报告论文主线之间的编码缺口。
- [iteration-playbook.md](iteration-playbook.md)：跨阶段通用执行模板。

## 2. 当前判断

当前已经成立的事实是：

- `Stage E / F / G(min) / H` 已完成真实数据链路、导出 contract 与测试闭环。
- `Stage I Phase 0/1/2/3` 的公开 benchmark 历史收口已完成。
- `chronaris_opt` 已完成鼎新私有 `proxy benchmark` 最优性验证，并固化 package。
- `public opt closed` 仍保留为公开支撑证据，但它不是论文双流主线已经完全收口的同义词。

当前最高优先级不再是继续扩 `public opt` 或重复跑旧 benchmark，而是：

- 把现有研究原型收敛成更贴近选题报告的“统一骨干、联合训练、真实 thesis task、checkpoint 推理、在线入口”主线。

## 3. 目录清理说明

- `planning` 根目录只保留现行路线图、gap、closure 和少量关键节点文档。
- 历史 `stage-xxx-plan` 文档已统一归档到 [archive/stage_i/README.md](archive/stage_i/README.md)。
- 归档不代表作废；它们继续保留为 `chronaris_opt`、`public opt` 和深基线阶段的历史快照与证据索引。

## 4. 分类

### 当前主线

- [coding-roadmap.md](coding-roadmap.md)
- [stage-i-thesis-mainline-roadmap-2026-05-15.md](stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [stage-i-thesis-mainline-coding-plan-2026-05-15.md](stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [thesis-coding-gap.md](thesis-coding-gap.md)
- [iteration-playbook.md](iteration-playbook.md)

### 阶段收口记录

- [stage-e-closure-2026-04-21.md](stage-e-closure-2026-04-21.md)
- [stage-f-closure-2026-04-22.md](stage-f-closure-2026-04-22.md)
- [stage-g-closure-2026-04-22.md](stage-g-closure-2026-04-22.md)
- [stage-h-closure-2026-04-27.md](stage-h-closure-2026-04-27.md)
- [stage-i-closure-2026-04-30.md](stage-i-closure-2026-04-30.md)

### 历史节点

- [stage-i-preparation.md](stage-i-preparation.md)
- [stage-i-third-party-baseline-prep-2026-04-30.md](stage-i-third-party-baseline-prep-2026-04-30.md)
- [stage-i-mainline-transition-2026-05-04.md](stage-i-mainline-transition-2026-05-04.md)
- [archive/stage_i/README.md](archive/stage_i/README.md)
