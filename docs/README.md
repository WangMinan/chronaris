# 文档索引

更新时间：2026-05-02

## 目录说明

`docs` 目录按二级分类组织为五组：

1. `foundation`：项目边界、分层、架构与基础契约
2. `planning`：路线图与阶段收口记录
3. `models`：模型输入协议、模型设计与参考资料
4. `reports`：阶段主报告与可复现实验结论
5. `选题报告与基金申请书`：选题与基金原始文档

## 运行约定

- 本仓库文档中的 Python 命令默认显式使用 `chronaris` 解释器：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python`
- 如果命令前带环境变量，例如 `CHRONARIS_ENABLE_*` 或 `CHRONARIS_MYSQL_HOST`，变量应直接放在该解释器前，不要退回 `base` 的 `python`

## 当前事实源

- 阶段状态只看 [coding-roadmap.md](planning/coding-roadmap.md)。
- 阶段 E/F/G/H 的收口依据只看 `docs/planning/stage-*-closure-*.md`。
- 阶段 I 收口依据只看 [stage-i-closure-2026-04-30.md](planning/stage-i-closure-2026-04-30.md) 与 [stage-i-closure-2026-04-30.md](reports/stage-i-closure-2026-04-30.md)。
- [stage-i-data-plan-2026-04-29.md](planning/stage-i-data-plan-2026-04-29.md) 继续保留为阶段 I 启动期计划，不再作为当前阶段状态判断依据。
- [stage-i-third-party-baseline-prep-2026-04-30.md](planning/stage-i-third-party-baseline-prep-2026-04-30.md) 是第三方基线落地前置准备记录。
- [stage-i-deep-baseline-plan-2026-05-01.md](planning/stage-i-deep-baseline-plan-2026-05-01.md) 是当前增强实验执行计划与已验证状态。
- [stage-i-private-benchmark-plan-2026-05-02.md](planning/stage-i-private-benchmark-plan-2026-05-02.md) 是当前私有双流最优性验证入口与执行顺序。
- 阶段 H 当前主报告只看 [stage-h-closure-2026-04-27.md](reports/stage-h-closure-2026-04-27.md)。
- 阶段 H all-window clean 运行报告：
  - [stage-h-private-e-allwindow-clean-2026-05-02.md](reports/stage-h-private-e-allwindow-clean-2026-05-02.md)
  - [stage-h-private-f-allwindow-clean-2026-05-02.md](reports/stage-h-private-f-allwindow-clean-2026-05-02.md)
- 阶段 I 当前阶段主报告：
  - `Phase 1`：[stage-i-uab-baseline-2026-04-29.md](reports/stage-i-uab-baseline-2026-04-29.md)
  - `Phase 2`：[stage-i-case-study-phase2-2026-04-29.md](reports/stage-i-case-study-phase2-2026-04-29.md)
  - `Phase 3` UAB window：[stage-i-uab-window-baseline-2026-04-29.md](reports/stage-i-uab-window-baseline-2026-04-29.md)
  - `Phase 3` NASA attention：[stage-i-nasa-attention-baseline-2026-04-29.md](reports/stage-i-nasa-attention-baseline-2026-04-29.md)
  - `Stage I closure`：[stage-i-closure-2026-04-30.md](reports/stage-i-closure-2026-04-30.md)
  - `Enhancement batch 1 real sortie`：[stage-i-real-sortie-deep-comparison-2026-05-01.md](reports/stage-i-real-sortie-deep-comparison-2026-05-01.md)
  - `Enhancement batch 2 public probe`：[stage-i-deep-comparison-probe-2026-05-01.md](reports/stage-i-deep-comparison-probe-2026-05-01.md)
  - `Enhancement batch 2 full LOSO`：[stage-i-deep-comparison-full-loso-2026-05-01.md](reports/stage-i-deep-comparison-full-loso-2026-05-01.md)
  - `Private optimized benchmark`：[private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md](reports/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md)
  - `Thesis support assessment`：[thesis-support-assessment-2026-05-01.md](reports/thesis-support-assessment-2026-05-01.md)
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

### models

- [e0-minimal-input.md](models/e0-minimal-input.md)
- [alignment-batch-contract.md](models/alignment-batch-contract.md)
- [stage-e-prototype-design.md](models/stage-e-prototype-design.md)
- [stage-e-reference-repos.md](models/stage-e-reference-repos.md)

### reports

- [validation-overlap-preview-20251005-act4-j20-22.md](reports/validation-overlap-preview-20251005-act4-j20-22.md)
- [e0-preview-20251005-act4-j20-22.md](reports/e0-preview-20251005-act4-j20-22.md)
- [alignment-preview-stage-e-closure-2026-04-21.md](reports/alignment-preview-stage-e-closure-2026-04-21.md)
- [alignment-preview-stage-f-closure-2026-04-22.md](reports/alignment-preview-stage-f-closure-2026-04-22.md)
- [alignment-preview-stage-g-min-closure-2026-04-22.md](reports/alignment-preview-stage-g-min-closure-2026-04-22.md)
- [sortie-availability-preview-20251002-act8-j16-12.md](reports/sortie-availability-preview-20251002-act8-j16-12.md)
- [stage-h-export-v1-2026-04-26.md](reports/stage-h-export-v1-2026-04-26.md)
- [stage-h-closure-2026-04-27.md](reports/stage-h-closure-2026-04-27.md)
- [stage-i-uab-baseline-2026-04-29.md](reports/stage-i-uab-baseline-2026-04-29.md)
- [stage-i-case-study-phase2-2026-04-29.md](reports/stage-i-case-study-phase2-2026-04-29.md)
- [stage-i-uab-window-baseline-2026-04-29.md](reports/stage-i-uab-window-baseline-2026-04-29.md)
- [stage-i-nasa-attention-baseline-2026-04-29.md](reports/stage-i-nasa-attention-baseline-2026-04-29.md)
- [stage-i-closure-2026-04-30.md](reports/stage-i-closure-2026-04-30.md)
- [stage-i-real-sortie-deep-comparison-2026-05-01.md](reports/stage-i-real-sortie-deep-comparison-2026-05-01.md)
- [stage-i-deep-comparison-probe-2026-05-01.md](reports/stage-i-deep-comparison-probe-2026-05-01.md)
- [stage-i-deep-comparison-full-loso-2026-05-01.md](reports/stage-i-deep-comparison-full-loso-2026-05-01.md)
- [thesis-support-assessment-2026-05-01.md](reports/thesis-support-assessment-2026-05-01.md)
- 私有双流 benchmark 实跑报告：
  - `chronaris_opt` full LOSO run：
    - [private-alignment-support-20260502T121815Z-stage-i-private-opt-full.md](reports/private-alignment-support-20260502T121815Z-stage-i-private-opt-full.md)
    - [private-causal-fusion-support-20260502T121815Z-stage-i-private-opt-full.md](reports/private-causal-fusion-support-20260502T121815Z-stage-i-private-opt-full.md)
    - [private-optimality-summary-20260502T121815Z-stage-i-private-opt-full.md](reports/private-optimality-summary-20260502T121815Z-stage-i-private-opt-full.md)
    - [private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md](reports/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md)
    - 结论是鼎新私有 proxy benchmark 三任务全面最优：`private_optimality_supported=True`

### 选题报告与基金申请书

- [西北工业大学硕士学位研究生论文选题报告表.docx](选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx)
- [西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx](选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx)
