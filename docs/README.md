# 文档索引

更新时间：2026-05-15

## 目录说明

`docs` 目录按二级分类组织为五组：

1. `foundation`：项目边界、分层、架构与基础契约
2. `planning`：路线图与阶段收口记录
3. `models`：模型输入协议、模型设计与参考资料
4. `reports`：阶段主报告与可复现实验结论
5. `选题报告与基金申请书`：选题与基金原始文档

`docs/reports` 现已按一级目录整理为 `alignment / preview / stage_h / stage_i / private`，入口索引见 [reports/README.md](reports/README.md)。

## 运行约定

- 本仓库文档中的 Python 命令默认显式使用 `chronaris` 解释器：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python`
- 如果命令前带环境变量，例如 `CHRONARIS_ENABLE_*` 或 `CHRONARIS_MYSQL_HOST`，变量应直接放在该解释器前，不要退回 `base` 的 `python`

## 当前事实源

- 阶段状态只看 [coding-roadmap.md](planning/coding-roadmap.md)。
- 下一阶段指导性路线图只看 [stage-i-thesis-mainline-roadmap-2026-05-15.md](planning/stage-i-thesis-mainline-roadmap-2026-05-15.md)。
- 按文件拆解的详细编码计划只看 [stage-i-thesis-mainline-coding-plan-2026-05-15.md](planning/stage-i-thesis-mainline-coding-plan-2026-05-15.md)。
- 若要回答“为了毕业论文还差哪些编码工作”，优先看 [thesis-coding-gap.md](planning/thesis-coding-gap.md)。
- 阶段 E/F/G/H 的收口依据只看 `docs/planning/stage-*-closure-*.md`。
- 阶段 I 收口依据只看 [stage-i-closure-2026-04-30.md](planning/stage-i-closure-2026-04-30.md) 与 [stage-i-closure-2026-04-30.md](reports/stage_i/stage-i-closure-2026-04-30.md)。
- [planning/README.md](planning/README.md) 负责区分 `planning` 根目录的现行入口与历史归档。
- [archive/stage_i/README.md](planning/archive/stage_i/README.md) 负责索引已归档的 `stage-xxx-plan` 历史文档。
- [stage-i-third-party-baseline-prep-2026-04-30.md](planning/stage-i-third-party-baseline-prep-2026-04-30.md) 是第三方基线落地前置准备记录。
- [stage-i-mainline-transition-2026-05-04.md](planning/stage-i-mainline-transition-2026-05-04.md) 保留为 `chronaris_opt` 升级为鼎新私有主线时的历史节点快照。
- [stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md](reports/stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md) 是当前公开主线统一结论：`public opt closed`。
- [stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md](reports/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md) 是当前中期报告整编证据入口；它保留 `public opt closed` 作为公开主线，同时把 `NASA public_fusion` 的 2026-05-09 full confirm 作为补充证据写入。
- 当前仓库代码 contract 已把公开第二模态固定为 `context proxy / public adapter evidence`；历史 public mainline / public opt 报告继续按各自生成时快照保留。
- 当前仓库代码 contract 已把鼎新 `T1/T2/T3` 固定为 `private proxy benchmark / proxy tasks`；不要把这些 contract 直接解读成人工真值 thesis tasks。
- 当前仓库代码 contract 已补上 `stage_i_backbone_train` 与 `Stage H checkpoint inference export`；历史 `per-view training` 导出路径继续保留为 preview/research 旁路。
- 阶段 H 当前主报告只看 [stage-h-closure-2026-04-27.md](reports/stage_h/stage-h-closure-2026-04-27.md)。
- 阶段 H all-window clean 运行报告：
  - [stage-h-private-e-allwindow-clean-2026-05-02.md](reports/stage_h/stage-h-private-e-allwindow-clean-2026-05-02.md)
  - [stage-h-private-f-allwindow-clean-2026-05-02.md](reports/stage_h/stage-h-private-f-allwindow-clean-2026-05-02.md)
- 阶段 I 当前阶段主报告：
  - `Phase 2`：[stage-i-case-study-phase2-2026-04-29.md](reports/stage_i/stage-i-case-study-phase2-2026-04-29.md)
  - `Stage I closure`：[stage-i-closure-2026-04-30.md](reports/stage_i/stage-i-closure-2026-04-30.md)
  - `Enhancement batch 2 full LOSO`：[stage-i-deep-comparison-full-loso-2026-05-01.md](reports/stage_i/stage-i-deep-comparison-full-loso-2026-05-01.md)
  - 论文证据 support：
    - [stage-i-alignment-support-20260506T120000Z-stage-i-support.md](reports/stage_i/stage-i-alignment-support-20260506T120000Z-stage-i-support.md)
    - [stage-i-causal-support-20260506T120000Z-stage-i-support.md](reports/stage_i/stage-i-causal-support-20260506T120000Z-stage-i-support.md)
    - [stage-i-ablation-support-20260506T120000Z-stage-i-support.md](reports/stage_i/stage-i-ablation-support-20260506T120000Z-stage-i-support.md)
    - 当前 support 资产根：`docs/reports/assets/stage_i_support/20260506T120000Z-stage-i-support/`
  - thesis-facing 最小原型输出：
    - runtime/demo：[stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md](reports/stage_i/stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md)
    - anchor：[stage-i-anchor-20260506T165435Z-stage-i-anchor.md](reports/stage_i/stage-i-anchor-20260506T165435Z-stage-i-anchor.md)
  - 当前中期整编证据：
    - [stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md](reports/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md)
  - `chronaris public opt` 实跑：
    - NASA enhanced round 1：[stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md](reports/stage_i/stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md)
    - UAB torch auto-cuda confirm：[stage-i-public-opt-20260506T165558Z-stage-i-public-opt-uab-torch-gpu.md](reports/stage_i/stage-i-public-opt-20260506T165558Z-stage-i-public-opt-uab-torch-gpu.md)
    - UAB robust-prior adapter：[stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md](reports/stage_i/stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md)
    - unified public mainline：[stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md](reports/stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md)
  - `chronaris_public_fusion` GPU screen：
    - round 2：[stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round2.md](reports/stage_i/stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round2.md)
    - 2026-05-09 prepared-v2 balanced confirm 与 full confirm：见 [stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md](reports/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md) 引用的 `docs/reports/assets/stage_i_midterm_runtime/` 资产
  - `Private optimized package`：[private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md](reports/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md)
  - 历史 baseline / probe / 旧 public-opt / round1 screen 与私有旧支撑报告已统一下沉到：
    - [stage_i/README.md](reports/stage_i/README.md)
    - [private/README.md](reports/private/README.md)
- 顶层 `docs/reports` 只保留主报告和必要数据盘点；E/F/G 的单配置子报告 Markdown 已清理，底层 JSON/CSV/图片/checkpoint assets 继续保留作为证据。

## 分类索引

### foundation

- [project-scope.md](foundation/project-scope.md)
- [repo-layout.md](foundation/repo-layout.md)
- [architecture.md](foundation/architecture.md)
- [data-contracts.md](foundation/data-contracts.md)
- [pipeline-v1.md](foundation/pipeline-v1.md)

### planning

- [planning/README.md](planning/README.md)
- [coding-roadmap.md](planning/coding-roadmap.md)
- [iteration-playbook.md](planning/iteration-playbook.md)
- [stage-i-thesis-mainline-roadmap-2026-05-15.md](planning/stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [stage-i-thesis-mainline-coding-plan-2026-05-15.md](planning/stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [stage-e-closure-2026-04-21.md](planning/stage-e-closure-2026-04-21.md)
- [stage-f-closure-2026-04-22.md](planning/stage-f-closure-2026-04-22.md)
- [stage-g-closure-2026-04-22.md](planning/stage-g-closure-2026-04-22.md)
- [stage-h-closure-2026-04-27.md](planning/stage-h-closure-2026-04-27.md)
- [stage-i-preparation.md](planning/stage-i-preparation.md)
- [stage-i-third-party-baseline-prep-2026-04-30.md](planning/stage-i-third-party-baseline-prep-2026-04-30.md)
- [stage-i-closure-2026-04-30.md](planning/stage-i-closure-2026-04-30.md)
- [stage-i-mainline-transition-2026-05-04.md](planning/stage-i-mainline-transition-2026-05-04.md)
- [thesis-coding-gap.md](planning/thesis-coding-gap.md)
- [planning/archive/stage_i/README.md](planning/archive/stage_i/README.md)

### models

- [e0-minimal-input.md](models/e0-minimal-input.md)
- [alignment-batch-contract.md](models/alignment-batch-contract.md)
- [stage-e-prototype-design.md](models/stage-e-prototype-design.md)
- [stage-e-reference-repos.md](models/stage-e-reference-repos.md)

### reports
- [reports/README.md](reports/README.md)
- [reports/alignment/](reports/alignment)
- [reports/preview/](reports/preview)
- [reports/stage_h/](reports/stage_h)
- [reports/stage_i/](reports/stage_i)
- [reports/private/](reports/private)

### 选题报告与基金申请书

- [西北工业大学硕士学位研究生论文选题报告表.docx](选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx)
- [西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx](选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx)
