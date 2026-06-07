# Chronaris 产物索引

更新时间：2026-06-07

## 1. 目录定位

本目录用于组织报告、图、CSV、JSON、checkpoint、manifest 等可引用产物。

当前为了兼容已有脚本，历史 `docs/reports/...` 路径仍然可用；但 AI coding 的当前阅读入口应使用本文件和 `artifacts/stage/`、`artifacts/mid-term/`。

## 2. 阶段产物

阶段产物按 [../implementation/PLAN.md](../implementation/PLAN.md) 的阶段划分：

- [stage/stage-a/](stage/stage-a/)：仓库初始化与最小设计。
- [stage/stage-b/](stage/stage-b/)：真实元信息与数据访问接入。
- [stage/stage-c/](stage/stage-c/)：统一样本组织与数据核验。
- [stage/stage-d/](stage/stage-d/)：数据集工程化与批量构建。
- [stage/stage-e0/](stage/stage-e0/)：单架次最小训练输入适配。
- [stage/stage-e/](stage/stage-e/)：双流连续潜态对齐。
- [stage/stage-f/](stage/stage-f/)：物理一致性约束。
- [stage/stage-g/](stage/stage-g/)：因果掩码与语义融合。
- [stage/stage-h/](stage/stage-h/)：标准化融合特征导出。
- [stage/stage-i/](stage/stage-i/)：典型任务评测、论文证据与运行时。

## 3. 中期答辩产物

- [mid-term/](mid-term/)：中期答辩证据包、图件、指标表和运行日志。

当前中期主入口：

- [mid-term/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md](mid-term/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md)

## 4. 当前最常引用产物

- Stage H 收口：[stage_h/stage-h-closure-2026-04-27.md](stage_h/stage-h-closure-2026-04-27.md)
- Stage I 历史公开收口：[stage_i/stage-i-closure-2026-04-30.md](stage_i/stage-i-closure-2026-04-30.md)
- Stage I public mainline：[stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md](stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md)
- Stage I support ablation：[stage_i/stage-i-ablation-support-20260506T120000Z-stage-i-support.md](stage_i/stage-i-ablation-support-20260506T120000Z-stage-i-support.md)
- Private optimized package summary：[private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md](private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md)
- 当前 `chronaris_opt` package：[assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json](assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json)

## 5. 引用规则

- 引用当前状态先看 [../STATE.md](../STATE.md)，不要从历史报告倒推当前阶段。
- 引用计划先看 [../implementation/PLAN.md](../implementation/PLAN.md)。
- 引用论文能力要求先看 [../requirements/SPEC.md](../requirements/SPEC.md)。
- 历史报告可以引用，但必须说明是历史快照、公开适配器证据、私有代理证据还是 thesis weak-label evidence。
