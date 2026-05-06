# 文档索引

更新时间：2026-05-06

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
- 阶段 E/F/G/H 的收口依据只看 `docs/planning/stage-*-closure-*.md`。
- 阶段 I 收口依据只看 [stage-i-closure-2026-04-30.md](planning/stage-i-closure-2026-04-30.md) 与 [stage-i-closure-2026-04-30.md](reports/stage_i/stage-i-closure-2026-04-30.md)。
- [stage-i-data-plan-2026-04-29.md](planning/stage-i-data-plan-2026-04-29.md) 继续保留为阶段 I 启动期计划，不再作为当前阶段状态判断依据。
- [stage-i-third-party-baseline-prep-2026-04-30.md](planning/stage-i-third-party-baseline-prep-2026-04-30.md) 是第三方基线落地前置准备记录。
- [stage-i-deep-baseline-plan-2026-05-01.md](planning/stage-i-deep-baseline-plan-2026-05-01.md) 是 `MulT / ContiFormer` 深基线阶段的历史执行快照，不再作为当前私有主线判断依据。
- [stage-i-private-benchmark-plan-2026-05-02.md](planning/stage-i-private-benchmark-plan-2026-05-02.md) 是当前私有双流最优性验证入口与执行顺序。
- [stage-i-mainline-transition-2026-05-04.md](planning/stage-i-mainline-transition-2026-05-04.md) 是当前“`chronaris_opt` 升级为鼎新私有主线、`E/F/G/H` 保留为历史基线”的迁移计划。
- [stage-i-public-opt-minimal-plan-2026-05-04.md](planning/stage-i-public-opt-minimal-plan-2026-05-04.md) 是 `chronaris public opt` 的历史最小起步计划；当前真实实跑已扩到 `UAB subjective regression` 与 `NASA attention_state`。
- [stage-i-public-opt-win-plan-2026-05-06.md](planning/stage-i-public-opt-win-plan-2026-05-06.md) 是当前“`public opt` 优于 `MulT / ContiFormer`”的增强执行计划与后续 `chronaris_public_fusion` 公开化预案。
- [stage-i-public-mainline-20260506T064302Z-stage-i-public-mainline.md](reports/stage_i/stage-i-public-mainline-20260506T064302Z-stage-i-public-mainline.md) 是当前公开主线统一结论：`NASA closed, UAB partial`。
- 阶段 H 当前主报告只看 [stage-h-closure-2026-04-27.md](reports/stage_h/stage-h-closure-2026-04-27.md)。
- 阶段 H all-window clean 运行报告：
  - [stage-h-private-e-allwindow-clean-2026-05-02.md](reports/stage_h/stage-h-private-e-allwindow-clean-2026-05-02.md)
  - [stage-h-private-f-allwindow-clean-2026-05-02.md](reports/stage_h/stage-h-private-f-allwindow-clean-2026-05-02.md)
- 阶段 I 当前阶段主报告：
  - `Phase 1`：[stage-i-uab-baseline-2026-04-29.md](reports/stage_i/stage-i-uab-baseline-2026-04-29.md)
  - `Phase 2`：[stage-i-case-study-phase2-2026-04-29.md](reports/stage_i/stage-i-case-study-phase2-2026-04-29.md)
  - `Phase 3` UAB window：[stage-i-uab-window-baseline-2026-04-29.md](reports/stage_i/stage-i-uab-window-baseline-2026-04-29.md)
  - `Phase 3` NASA attention：[stage-i-nasa-attention-baseline-2026-04-29.md](reports/stage_i/stage-i-nasa-attention-baseline-2026-04-29.md)
  - `Stage I closure`：[stage-i-closure-2026-04-30.md](reports/stage_i/stage-i-closure-2026-04-30.md)
  - `Enhancement batch 1 real sortie`：[stage-i-real-sortie-deep-comparison-2026-05-01.md](reports/stage_i/stage-i-real-sortie-deep-comparison-2026-05-01.md)
  - `Enhancement batch 2 public probe`：[stage-i-deep-comparison-probe-2026-05-01.md](reports/stage_i/stage-i-deep-comparison-probe-2026-05-01.md)
  - `Enhancement batch 2 full LOSO`：[stage-i-deep-comparison-full-loso-2026-05-01.md](reports/stage_i/stage-i-deep-comparison-full-loso-2026-05-01.md)
  - 论文证据 support：
    - [stage-i-alignment-support-20260506T120000Z-stage-i-support.md](reports/stage_i/stage-i-alignment-support-20260506T120000Z-stage-i-support.md)
    - [stage-i-causal-support-20260506T120000Z-stage-i-support.md](reports/stage_i/stage-i-causal-support-20260506T120000Z-stage-i-support.md)
    - [stage-i-ablation-support-20260506T120000Z-stage-i-support.md](reports/stage_i/stage-i-ablation-support-20260506T120000Z-stage-i-support.md)
    - 当前 support 资产根：`docs/reports/assets/stage_i_support/20260506T120000Z-stage-i-support/`
  - `chronaris public opt` 实跑：
    - UAB subjective historical baseline：[stage-i-public-opt-20260506T121000Z-stage-i-public-opt-uab.md](reports/stage_i/stage-i-public-opt-20260506T121000Z-stage-i-public-opt-uab.md)
    - NASA attention：[stage-i-public-opt-20260506T124500Z-stage-i-public-opt-nasa.md](reports/stage_i/stage-i-public-opt-20260506T124500Z-stage-i-public-opt-nasa.md)
    - NASA enhanced round 1：[stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md](reports/stage_i/stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md)
    - UAB torch full LOSO：[stage-i-public-opt-20260506T063146Z-stage-i-public-opt-uab-torch.md](reports/stage_i/stage-i-public-opt-20260506T063146Z-stage-i-public-opt-uab-torch.md)
    - unified public mainline：[stage-i-public-mainline-20260506T064302Z-stage-i-public-mainline.md](reports/stage_i/stage-i-public-mainline-20260506T064302Z-stage-i-public-mainline.md)
  - `chronaris_public_fusion` GPU screen：
    - round 1：[stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round1.md](reports/stage_i/stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round1.md)
    - round 2：[stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round2.md](reports/stage_i/stage-i-public-fusion-screen-20260506T-stage-i-public-fusion-screen-round2.md)
  - `Private optimized benchmark`：[private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md](reports/private/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md)
  - `Private optimized package`：[private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md](reports/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md)
  - `Thesis support assessment`：[thesis-support-assessment-2026-05-01.md](reports/stage_i/thesis-support-assessment-2026-05-01.md)
    - 注意：该文档成稿时间早于 `chronaris_opt` 私有 full LOSO，不再作为“鼎新当前最优性”判断依据
- 顶层 `docs/reports` 只保留主报告和必要数据盘点；E/F/G 的单配置子报告 Markdown 已清理，底层 JSON/CSV/图片/checkpoint assets 继续保留作为证据。

## 分类索引

### foundation

- [project-scope.md](foundation/project-scope.md)
- [repo-layout.md](foundation/repo-layout.md)
- [architecture.md](foundation/architecture.md)
- [data-contracts.md](foundation/data-contracts.md)
- [pipeline-v1.md](foundation/pipeline-v1.md)

### planning

- [coding-roadmap.md](planning/coding-roadmap.md)
- [iteration-playbook.md](planning/iteration-playbook.md)
- [stage-e-closure-2026-04-21.md](planning/stage-e-closure-2026-04-21.md)
- [stage-f-closure-2026-04-22.md](planning/stage-f-closure-2026-04-22.md)
- [stage-g-closure-2026-04-22.md](planning/stage-g-closure-2026-04-22.md)
- [stage-h-closure-2026-04-27.md](planning/stage-h-closure-2026-04-27.md)
- [stage-i-preparation.md](planning/stage-i-preparation.md)
- [stage-i-data-plan-2026-04-29.md](planning/stage-i-data-plan-2026-04-29.md)
- [stage-i-third-party-baseline-prep-2026-04-30.md](planning/stage-i-third-party-baseline-prep-2026-04-30.md)
- [stage-i-closure-2026-04-30.md](planning/stage-i-closure-2026-04-30.md)
- [stage-i-deep-baseline-plan-2026-05-01.md](planning/stage-i-deep-baseline-plan-2026-05-01.md)
- [stage-i-private-benchmark-plan-2026-05-02.md](planning/stage-i-private-benchmark-plan-2026-05-02.md)
- [stage-i-mainline-transition-2026-05-04.md](planning/stage-i-mainline-transition-2026-05-04.md)
- [stage-i-public-opt-minimal-plan-2026-05-04.md](planning/stage-i-public-opt-minimal-plan-2026-05-04.md)
- [stage-i-public-opt-win-plan-2026-05-06.md](planning/stage-i-public-opt-win-plan-2026-05-06.md)

### models

- [e0-minimal-input.md](models/e0-minimal-input.md)
- [alignment-batch-contract.md](models/alignment-batch-contract.md)
- [stage-e-prototype-design.md](models/stage-e-prototype-design.md)
- [stage-e-reference-repos.md](models/stage-e-reference-repos.md)

### reports

- [validation-overlap-preview-20251005-act4-j20-22.md](reports/preview/validation-overlap-preview-20251005-act4-j20-22.md)
- [e0-preview-20251005-act4-j20-22.md](reports/preview/e0-preview-20251005-act4-j20-22.md)
- [alignment-preview-stage-e-closure-2026-04-21.md](reports/alignment/alignment-preview-stage-e-closure-2026-04-21.md)
- [alignment-preview-stage-f-closure-2026-04-22.md](reports/alignment/alignment-preview-stage-f-closure-2026-04-22.md)
- [alignment-preview-stage-g-min-closure-2026-04-22.md](reports/alignment/alignment-preview-stage-g-min-closure-2026-04-22.md)
- [sortie-availability-preview-20251002-act8-j16-12.md](reports/preview/sortie-availability-preview-20251002-act8-j16-12.md)
- [stage-h-export-v1-2026-04-26.md](reports/stage_h/stage-h-export-v1-2026-04-26.md)
- [stage-h-closure-2026-04-27.md](reports/stage_h/stage-h-closure-2026-04-27.md)
- [stage-i-uab-baseline-2026-04-29.md](reports/stage_i/stage-i-uab-baseline-2026-04-29.md)
- [stage-i-case-study-phase2-2026-04-29.md](reports/stage_i/stage-i-case-study-phase2-2026-04-29.md)
- [stage-i-uab-window-baseline-2026-04-29.md](reports/stage_i/stage-i-uab-window-baseline-2026-04-29.md)
- [stage-i-nasa-attention-baseline-2026-04-29.md](reports/stage_i/stage-i-nasa-attention-baseline-2026-04-29.md)
- [stage-i-closure-2026-04-30.md](reports/stage_i/stage-i-closure-2026-04-30.md)
- [stage-i-real-sortie-deep-comparison-2026-05-01.md](reports/stage_i/stage-i-real-sortie-deep-comparison-2026-05-01.md)
- [stage-i-deep-comparison-probe-2026-05-01.md](reports/stage_i/stage-i-deep-comparison-probe-2026-05-01.md)
- [stage-i-deep-comparison-full-loso-2026-05-01.md](reports/stage_i/stage-i-deep-comparison-full-loso-2026-05-01.md)
- [thesis-support-assessment-2026-05-01.md](reports/stage_i/thesis-support-assessment-2026-05-01.md)
- 私有双流 benchmark 实跑报告：
  - `chronaris_opt` full LOSO run：
    - [private-alignment-support-20260502T121815Z-stage-i-private-opt-full.md](reports/private/private-alignment-support-20260502T121815Z-stage-i-private-opt-full.md)
    - [private-causal-fusion-support-20260502T121815Z-stage-i-private-opt-full.md](reports/private/private-causal-fusion-support-20260502T121815Z-stage-i-private-opt-full.md)
    - [private-optimality-summary-20260502T121815Z-stage-i-private-opt-full.md](reports/private/private-optimality-summary-20260502T121815Z-stage-i-private-opt-full.md)
    - [private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md](reports/private/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md)
    - 结论是鼎新私有 proxy benchmark 三任务全面最优：`private_optimality_supported=True`
  - `chronaris_opt` package 固化报告：
    - [private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md](reports/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md)
    - [private-optimized-package-20260504T120000Z-stage-i-private-opt-package.md](reports/private/private-optimized-package-20260504T120000Z-stage-i-private-opt-package.md)
    - 当前 package 路径：`docs/reports/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`

### 选题报告与基金申请书

- [西北工业大学硕士学位研究生论文选题报告表.docx](选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx)
- [西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx](选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx)
